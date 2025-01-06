import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sentence_transformers import SentenceTransformer


class LatentLoopTransformer(nn.Module):
    def __init__(self, latent_gpt, language_gpt, max_loops=5):
        """
        Initializes the LatentLoopTransformer with two GPT models.

        Args:
            latent_gpt (GPT): GPT model used for latent mode refinement.
            language_gpt (GPT): GPT model used for language mode generation.
            max_loops (int): Maximum number of latent refinement loops.
        """
        super(LatentLoopTransformer, self).__init__()
        self.latent_gpt = latent_gpt
        self.language_gpt = language_gpt
        self.max_loops = max_loops

    def forward(self, input_ids, return_intermediate=False):
        """
        Forward pass through the Latent Loop Transformer.

        Args:
            input_ids (torch.LongTensor): Input token IDs of shape (B, T).
            return_intermediate (bool): If True, returns the hidden states after the first latent loop.

        Returns:
            dict: Dictionary containing refined latent results, intermediate hidden states (optional), and final logits.
        """
        B, T = input_ids.size()

        # Embedding and position initialization for latent mode
        tok_emb = self.latent_gpt.token_embedding(input_ids)  # (B, T, C)
        positions = torch.arange(
            0, T, dtype=torch.long, device=input_ids.device
        ).unsqueeze(
            0
        )  # (1, T)
        pos_emb = self.latent_gpt.position_embedding(positions)  # (1, T, C)
        hidden_states = tok_emb + pos_emb  # (B, T, C)

        latent_results = []

        # Latent Mode: Iteratively refine latent states
        for loop_idx in range(self.max_loops):
            for block in self.latent_gpt.blocks:
                hidden_states = block(hidden_states)  # (B, T, C)

            # Append the refined hidden states
            latent_results.append(hidden_states)

            # Optionally return intermediate hidden states
            if return_intermediate and loop_idx == 0:
                return {"intermediate_hidden_states": hidden_states}

        # Concatenate latent results across all loops
        latent_results = torch.stack(latent_results, dim=1)  # (B, max_loops, T, C)
        latent_context = latent_results.view(
            B, -1, hidden_states.size(-1)
        )  # Flatten across loops (B, max_loops*T, C)

        # Language Mode: Combine latent context with the original input for generation
        combined_input_ids = torch.cat([input_ids, latent_context.view(B, -1)], dim=1)

        tok_emb = self.language_gpt.token_embedding(
            combined_input_ids
        )  # (B, T + latent_context_len, C)
        positions = torch.arange(
            0, combined_input_ids.size(1), dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        pos_emb = self.language_gpt.position_embedding(positions)
        hidden_states = tok_emb + pos_emb

        for block in self.language_gpt.blocks:
            hidden_states = block(hidden_states)  # (B, T, C)

        hidden_states = self.language_gpt.ln_f(hidden_states)  # (B, T, C)
        logits = self.language_gpt.lm_head(hidden_states)  # (B, T, vocab_size)

        return {
            "latent_results": latent_results,
            "logits": logits,
        }


class EmbeddingLatentSupervision(nn.Module):
    def __init__(self, embedding_dim, latent_dim, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.sentence_encoder = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # Projection layer if dimensions don't match
        self.projection = (
            nn.Linear(embedding_dim, latent_dim)
            if embedding_dim != latent_dim
            else nn.Identity()
        )

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(latent_dim)

    def get_target_embeddings(self, intermediate_answers):
        """Convert text to embeddings using the sentence encoder"""
        with torch.no_grad():
            embeddings = self.sentence_encoder.encode(
                intermediate_answers,
                convert_to_tensor=True,
                normalize_embeddings=True,  # Added normalization
            )
            projected = self.projection(embeddings)
            return self.layer_norm(projected)


class LatentLoopLoss(nn.Module):
    def __init__(
        self,
        vocab_size,
        loop_weight=0.5,
        kl_weight=0.1,
        consistency_weight=0.3,
        embedding_weight=0.4,
    ):
        """
        Enhanced Loss function with embedding supervision

        Args:
            vocab_size (int): Size of vocabulary for language generation
            loop_weight (float): Weight for latent refinement loss
            kl_weight (float): Weight for KL divergence
            consistency_weight (float): Weight for consistency loss
            embedding_weight (float): Weight for embedding supervision loss
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.loop_weight = loop_weight
        self.kl_weight = kl_weight
        self.consistency_weight = consistency_weight
        self.embedding_weight = embedding_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def kl_divergence(self, p, q):
        """Calculate KL divergence between two latent distributions"""
        p_norm = F.softmax(p, dim=-1)
        q_norm = F.softmax(q, dim=-1)
        return torch.mean(
            torch.sum(
                p_norm * (torch.log(p_norm + 1e-10) - torch.log(q_norm + 1e-10)), dim=-1
            )
        )

    def forward(self, model_output, targets):
        """
        Calculate the combined loss including embedding supervision

        Args:
            model_output (dict): Output from LatentLoopTransformer containing:
                - latent_results: Tensor of shape (B, max_loops, T, C)
                - logits: Final language model logits
            targets (dict): Target values containing:
                - answer: Target tokens for language generation
                - intermediate_states: Target latent states
                - embedding_targets: Target embeddings from EmbeddingLatentSupervision
        """
        latent_results = model_output["latent_results"]
        final_logits = model_output["logits"]
        batch_size, num_loops, seq_len, hidden_dim = latent_results.size()

        # 1. Language Generation Loss
        generation_loss = self.ce_loss(
            final_logits.view(-1, self.vocab_size), targets["answer"].view(-1)
        )

        # 2. Latent Refinement Loss
        latent_loss = 0
        if "intermediate_states" in targets:
            for loop_idx in range(num_loops):
                latent_loss += self.mse_loss(
                    latent_results[:, loop_idx],
                    targets["intermediate_states"][:, loop_idx],
                )
        latent_loss /= num_loops

        # 3. Embedding Supervision Loss
        embedding_loss = 0
        if "embedding_targets" in targets:
            target_embeddings = targets["embedding_targets"]
            # Calculate loss against final latent state
            final_latent = latent_results[:, -1].mean(
                dim=1
            )  # Average over sequence length
            embedding_loss = self.cosine_loss(
                final_latent,
                target_embeddings,
                torch.ones(batch_size, device=final_latent.device),
            )

        # 4. Inter-loop Consistency Loss
        consistency_loss = 0
        if num_loops > 1:
            for i in range(num_loops - 1):
                consistency_loss += self.mse_loss(
                    latent_results[:, i + 1], latent_results[:, i].detach()
                )
            consistency_loss /= num_loops - 1

        # 5. KL Divergence between successive latent states
        kl_loss = 0
        if num_loops > 1:
            for i in range(num_loops - 1):
                kl_loss += self.kl_divergence(
                    latent_results[:, i + 1].view(-1, hidden_dim),
                    latent_results[:, i].view(-1, hidden_dim),
                )
            kl_loss /= num_loops - 1

        # Combine all losses
        total_loss = (
            generation_loss
            + self.loop_weight * latent_loss
            + self.consistency_weight * consistency_loss
            + self.kl_weight * kl_loss
            + self.embedding_weight * embedding_loss
        )

        return {
            "total_loss": total_loss,
            "generation_loss": generation_loss.item(),
            "latent_loss": latent_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "kl_loss": kl_loss.item(),
            "embedding_loss": embedding_loss.item(),
        }

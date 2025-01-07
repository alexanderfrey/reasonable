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
        B, T = input_ids.size()
        device = input_ids.device

        # Detailed input validation
        # print(f"\nInput validation:")
        # print(f"Batch size: {B}, Sequence length: {T}")
        # print(
        #     f"Input range: min={input_ids.min().item()}, max={input_ids.max().item()}"
        # )
        # print(f"Embedding weight shape: {self.latent_gpt.token_embedding.weight.shape}")
        # print(
        #     f"Position embedding max size: {self.latent_gpt.position_embedding.weight.shape}"
        # )

        max_len = self.latent_gpt.position_embedding.weight.shape[0]
        if T > max_len:
            # print(f"Truncating sequence from {T} to {max_len}")
            input_ids = input_ids[:, :max_len]
            T = max_len

        try:
            # Token embedding with validation
            tok_emb = self.latent_gpt.token_embedding(input_ids)
            # print("Token embedding successful:", tok_emb.shape)

            # Position embedding with validation
            positions = torch.arange(0, T, dtype=torch.long, device=device)
            if positions.max() >= self.latent_gpt.position_embedding.weight.shape[0]:
                raise ValueError("Position indices exceed embedding capacity")

            positions = positions.unsqueeze(0)  # (1, T)
            pos_emb = self.latent_gpt.position_embedding(positions)
            # print("Position embedding successful:", pos_emb.shape)

            # Combine embeddings
            hidden_states = tok_emb + pos_emb
            # print("Initial hidden states shape:", hidden_states.shape)

            latent_results = []
            for loop_idx in range(self.max_loops):
                current_hidden = hidden_states

                # Process through blocks with validation
                for i, block in enumerate(self.latent_gpt.blocks):
                    try:
                        current_hidden = block(current_hidden)
                        if torch.isnan(current_hidden).any():
                            raise ValueError(f"NaN detected after block {i}")
                    except Exception as e:
                        print(f"Error in block {i}: {str(e)}")
                        raise

                latent_results.append(current_hidden)

            # Stack and process latent results
            latent_results = torch.stack(latent_results, dim=1)
            # print("Latent results shape:", latent_results.shape)

            # Language model processing
            latent_context = latent_results.mean(dim=1)
            # print("Latent context shape:", latent_context.shape)

            # Final processing
            hidden_states = latent_context
            for i, block in enumerate(self.language_gpt.blocks):
                try:
                    hidden_states = block(hidden_states)
                    if torch.isnan(hidden_states).any():
                        raise ValueError(f"NaN detected in language model block {i}")
                except Exception as e:
                    print(f"Error in language model block {i}: {str(e)}")
                    raise

            hidden_states = self.language_gpt.ln_f(hidden_states)
            logits = self.language_gpt.lm_head(hidden_states)
            # print("Final logits shape:", logits.shape)

            return {
                "latent_results": latent_results,
                "logits": logits,
            }

        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            raise


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
        self, vocab_size, loop_weight=0.1, consistency_weight=0.1
    ):  # Reduced weights
        super().__init__()
        self.vocab_size = vocab_size
        self.loop_weight = loop_weight
        self.consistency_weight = consistency_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()

    def forward(self, model_output, targets):
        # Language Generation Loss
        logits = model_output["logits"]
        generation_loss = self.ce_loss(
            logits.view(-1, self.vocab_size), targets["answer"].view(-1)
        )

        # Latent Refinement Loss with gradient clipping
        latent_loss = torch.tensor(0.0, device=logits.device)
        if "intermediate_states" in targets:
            intermediate_states = targets["intermediate_states"]
            latent_results = model_output["latent_results"]

            # Mean over sequence dimension
            latent_results_mean = latent_results.mean(dim=2)
            latent_steps = latent_results_mean[:, : intermediate_states.shape[1], :]

            # Project with gradient clipping
            projection = nn.Linear(
                intermediate_states.shape[2], latent_results.shape[-1]
            ).to(intermediate_states.device)

            projected_states = projection(intermediate_states.float())

            # Clip gradients for stability
            projected_states = torch.clamp(projected_states, -100, 100)
            latent_steps = torch.clamp(latent_steps, -100, 100)

            latent_loss = self.mse_loss(latent_steps, projected_states)
            latent_loss = torch.clamp(latent_loss, 0, 100)  # Prevent explosion

        # Consistency Loss
        consistency_loss = torch.tensor(0.0, device=logits.device)
        num_loops = model_output["latent_results"].shape[1]
        if num_loops > 1:
            latent_avg = model_output["latent_results"].mean(dim=2)
            latent_avg = torch.clamp(latent_avg, -100, 100)  # Clip values

            for i in range(num_loops - 1):
                step_loss = self.mse_loss(
                    latent_avg[:, i + 1], latent_avg[:, i].detach()
                )
                consistency_loss += torch.clamp(step_loss, 0, 100)
            consistency_loss /= num_loops - 1

        # Combine losses with checks
        total_loss = (
            generation_loss
            + self.loop_weight * latent_loss
            + self.consistency_weight * consistency_loss
        )

        # Final safety clamp
        total_loss = torch.clamp(total_loss, 0, 1000)

        return {
            "total_loss": total_loss,
            "generation_loss": generation_loss.item(),
            "latent_loss": latent_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "kl_loss": 0.0,
        }


# class LatentLoopLoss(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         loop_weight=0.5,
#         kl_weight=0.1,
#         consistency_weight=0.3,
#         embedding_weight=0.4,
#     ):
#         """
#         Enhanced Loss function with embedding supervision

#         Args:
#             vocab_size (int): Size of vocabulary for language generation
#             loop_weight (float): Weight for latent refinement loss
#             kl_weight (float): Weight for KL divergence
#             consistency_weight (float): Weight for consistency loss
#             embedding_weight (float): Weight for embedding supervision loss
#         """
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.loop_weight = loop_weight
#         self.kl_weight = kl_weight
#         self.consistency_weight = consistency_weight
#         self.embedding_weight = embedding_weight

#         self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
#         self.mse_loss = nn.MSELoss()
#         self.cosine_loss = nn.CosineEmbeddingLoss()

#     def kl_divergence(self, p, q):
#         """Calculate KL divergence between two latent distributions"""
#         p_norm = F.softmax(p, dim=-1)
#         q_norm = F.softmax(q, dim=-1)
#         return torch.mean(
#             torch.sum(
#                 p_norm * (torch.log(p_norm + 1e-10) - torch.log(q_norm + 1e-10)), dim=-1
#             )
#         )

#     def forward(self, model_output, targets):
#         """
#         Calculate the combined loss including embedding supervision

#         Args:
#             model_output (dict): Output from LatentLoopTransformer containing:
#                 - latent_results: Tensor of shape (B, max_loops, T, C)
#                 - logits: Final language model logits
#             targets (dict): Target values containing:
#                 - answer: Target tokens for language generation
#                 - intermediate_states: Target latent states
#                 - embedding_targets: Target embeddings from EmbeddingLatentSupervision
#         """
#         latent_results = model_output["latent_results"]
#         final_logits = model_output["logits"]
#         batch_size, num_loops, seq_len, hidden_dim = latent_results.size()

#         # 1. Language Generation Loss
#         generation_loss = self.ce_loss(
#             final_logits.view(-1, self.vocab_size), targets["answer"].view(-1)
#         )

#         # 2. Latent Refinement Loss
#         latent_loss = 0
#         if "intermediate_states" in targets:
#             for loop_idx in range(num_loops):
#                 latent_loss += self.mse_loss(
#                     latent_results[:, loop_idx],
#                     targets["intermediate_states"][:, loop_idx],
#                 )
#         latent_loss /= num_loops

#         # 3. Embedding Supervision Loss
#         embedding_loss = 0
#         if "embedding_targets" in targets:
#             target_embeddings = targets["embedding_targets"]
#             # Calculate loss against final latent state
#             final_latent = latent_results[:, -1].mean(
#                 dim=1
#             )  # Average over sequence length
#             embedding_loss = self.cosine_loss(
#                 final_latent,
#                 target_embeddings,
#                 torch.ones(batch_size, device=final_latent.device),
#             )

#         # 4. Inter-loop Consistency Loss
#         consistency_loss = 0
#         if num_loops > 1:
#             for i in range(num_loops - 1):
#                 consistency_loss += self.mse_loss(
#                     latent_results[:, i + 1], latent_results[:, i].detach()
#                 )
#             consistency_loss /= num_loops - 1

#         # 5. KL Divergence between successive latent states
#         kl_loss = 0
#         if num_loops > 1:
#             for i in range(num_loops - 1):
#                 kl_loss += self.kl_divergence(
#                     latent_results[:, i + 1].view(-1, hidden_dim),
#                     latent_results[:, i].view(-1, hidden_dim),
#                 )
#             kl_loss /= num_loops - 1

#         # Combine all losses
#         total_loss = (
#             generation_loss
#             + self.loop_weight * latent_loss
#             + self.consistency_weight * consistency_loss
#             + self.kl_weight * kl_loss
#             + self.embedding_weight * embedding_loss
#         )

#         return {
#             "total_loss": total_loss,
#             "generation_loss": generation_loss.item(),
#             "latent_loss": latent_loss.item(),
#             "consistency_loss": consistency_loss.item(),
#             "kl_loss": kl_loss.item(),
#             "embedding_loss": embedding_loss.item(),
#         }

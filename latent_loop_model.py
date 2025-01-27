import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sentence_transformers import SentenceTransformer


class LatentReasoningTransformer(nn.Module):
    def __init__(self, latent_gpt, language_gpt, max_loops=5, thought_processor=None):
        super(LatentReasoningTransformer, self).__init__()
        if thought_processor is None:
            raise ValueError("thought_processor must be provided")

        self.latent_gpt = latent_gpt
        self.language_gpt = language_gpt
        self.max_loops = max_loops
        self.thought_processor = thought_processor

        # Add projection layers for reasoning state
        self.reasoning_projector = nn.Linear(
            latent_gpt.config.n_embd, latent_gpt.config.n_embd
        )
        self.reasoning_predictor = nn.Linear(
            latent_gpt.config.n_embd, latent_gpt.config.n_embd
        )

        # Initialize reasoning probe
        self.reasoning_probe = LatentReasoningProbe(
            hidden_size=latent_gpt.config.n_embd,
            vocab_size=latent_gpt.config.vocab_size,
            max_seq_len=1024,
        )

        # Add thought processing layers
        self.thought_encoder = nn.Linear(
            latent_gpt.config.n_embd, latent_gpt.config.n_embd
        )
        self.thought_attention = nn.MultiheadAttention(
            embed_dim=latent_gpt.config.n_embd,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

    def process_thoughts(self, latent_states, thoughts):
        """
        Process thoughts using the thought processor
        """
        processed_thoughts = self.thought_processor.process_thought(thoughts)
        encoded_thoughts = self.thought_encoder(processed_thoughts)

        attended_states, _ = self.thought_attention(
            query=latent_states, key=encoded_thoughts, value=encoded_thoughts
        )

        return attended_states

    def interpret_reasoning_step(self, latent_state, tokenizer, max_length=50):
        """
        Reasoning step interpretation using thought processor
        """
        with torch.no_grad():
            return self.thought_processor.decode_thought(
                latent_state, self.reasoning_probe, max_length=max_length
            )

    def freeze_probe(self):
        for param in self.reasoning_probe.parameters():
            param.requires_grad = False

    def unfreeze_probe(self):
        for param in self.reasoning_probe.parameters():
            param.requires_grad = True

    def generate_intermediate_state(
        self, hidden_states, thoughts=None, prev_reasoning_state=None
    ):
        """
        Generate intermediate state with thought processing
        """
        current_hidden = hidden_states

        # Incorporate previous reasoning state if available
        if prev_reasoning_state is not None:
            reasoning_projection = self.reasoning_projector(prev_reasoning_state)
            current_hidden = current_hidden + reasoning_projection

        # Process through latent GPT blocks
        for block in self.latent_gpt.blocks:
            current_hidden = block(current_hidden)

        # Project to latent space
        latent_state = self.latent_gpt.ln_f(current_hidden)

        # Process thoughts if available
        if thoughts is not None:
            latent_state = self.process_thoughts(latent_state, thoughts)

        # Generate prediction for next reasoning state
        next_reasoning_state = self.reasoning_predictor(latent_state)

        return current_hidden, latent_state, next_reasoning_state

    def forward(
        self,
        input_ids,
        intermediate_answers=None,
        return_intermediate=True,
        target_loop_idx=None,
    ):
        """
        Enhanced forward pass with thought processing
        """
        hidden_states = self._get_embeddings(input_ids)

        # Initialize storage for intermediate results
        latent_results = []
        intermediate_hiddens = []
        reasoning_predictions = []
        probe_outputs = []
        prev_reasoning_state = None

        # Determine number of loops
        num_loops = (
            target_loop_idx + 1 if target_loop_idx is not None else self.max_loops
        )

        # Generate intermediate states with thought processing
        for loop_idx in range(num_loops):
            # Get thoughts for current loop if available
            current_thoughts = (
                intermediate_answers[:, loop_idx]
                if intermediate_answers is not None
                else None
            )

            # Generate intermediate state
            current_hidden, latent_state, next_reasoning_state = (
                self.generate_intermediate_state(
                    hidden_states,
                    thoughts=current_thoughts,
                    prev_reasoning_state=prev_reasoning_state,
                )
            )

            # Generate probe output for current state
            probe_output = self.reasoning_probe(
                latent_state.unsqueeze(1),
                target_length=(
                    intermediate_answers.size(-1)
                    if intermediate_answers is not None
                    else None
                ),
            )

            # Store results
            intermediate_hiddens.append(current_hidden)
            latent_results.append(latent_state)
            reasoning_predictions.append(next_reasoning_state)
            probe_outputs.append(probe_output)

            # Update states for next iteration
            hidden_states = current_hidden
            prev_reasoning_state = latent_state

            # Return early if targeting specific loop
            if target_loop_idx is not None and loop_idx == target_loop_idx:
                return {
                    "intermediate_state": latent_state,
                    "probe_output": probe_output,
                    "loop_idx": loop_idx,
                }

        # Stack results
        latent_results = torch.stack(latent_results, dim=1)
        intermediate_hiddens = torch.stack(intermediate_hiddens, dim=1)
        reasoning_predictions = torch.stack(reasoning_predictions, dim=1)
        probe_outputs = torch.stack(probe_outputs, dim=1)

        # Process through language model if not targeting specific loop
        if target_loop_idx is None:
            # Combine inputs
            question_context = self._get_embeddings(input_ids)
            combined_input = torch.cat(
                [question_context.unsqueeze(1), intermediate_hiddens], dim=1
            )
            combined_input = combined_input.view(
                combined_input.size(0), -1, combined_input.size(-1)
            )

            # Process through language GPT
            hidden_states = combined_input
            for block in self.language_gpt.blocks:
                hidden_states = block(hidden_states)

            hidden_states = self.language_gpt.ln_f(hidden_states)
            logits = self.language_gpt.lm_head(hidden_states)

            if return_intermediate:
                return {
                    "latent_results": latent_results,
                    "intermediate_hiddens": intermediate_hiddens,
                    "reasoning_predictions": reasoning_predictions,
                    "probe_outputs": probe_outputs,
                    "logits": logits,
                    "combined_input": combined_input,
                }
            else:
                return {"logits": logits}
        else:
            raise ValueError(
                "Unexpectedly reached end of forward pass with target_loop_idx"
            )


class LatentReasoningProbe(nn.Module):
    def __init__(self, hidden_size, vocab_size, max_seq_len=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Add special token handling
        self.start_token_embedding = nn.Parameter(
            torch.randn(1, hidden_size) / hidden_size**0.5
        )
        self.end_token_embedding = nn.Parameter(
            torch.randn(1, hidden_size) / hidden_size**0.5
        )

        # Rest of the architecture remains similar but with improved initialization
        self.token_embedding = nn.Parameter(
            torch.randn(vocab_size, hidden_size) / hidden_size**0.5
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Enhanced latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Add attention mask handling
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def generate(self, latent_states, max_length=512, temperature=0.7):
        """
        Improved generation with explicit handling of special tokens
        """
        batch_size = latent_states.size(0)
        device = latent_states.device

        # Start with start token
        current_output = torch.full(
            (batch_size, 1), self.start_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_length - 1):
            # Get predictions
            logits = self.forward(latent_states, current_output)
            next_token_logits = logits[:, -1, :] / temperature

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for end of sequence
            if (next_token == self.end_token_id).all():
                break

            # Append next token
            current_output = torch.cat([current_output, next_token], dim=1)

        return current_output


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
    def __init__(self, vocab_size, embedding_dim, latent_dim, latent_weight=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_weight = latent_weight
        self.ce_loss = nn.CrossEntropyLoss()

        # Initialize embedding supervision
        self.embedding_supervision = EmbeddingLatentSupervision(
            embedding_dim=embedding_dim, latent_dim=latent_dim
        )

        # Add projection layer for student states
        self.student_proj = nn.Linear(latent_dim, latent_dim)

    def calculate_intermediate_loss(
        self, model_output, target_state, tokenizer, loop_idx
    ):
        """Calculate loss for a specific intermediate state"""
        intermediate_state = model_output["intermediate_state"]

        # Convert target state tensor to text
        target_text = tokenizer.decode(target_state, skip_special_tokens=True)

        # Convert target state text to embedding
        with torch.no_grad():
            target_embedding = self.embedding_supervision.get_target_embeddings(
                [target_text]
            )
            target_embedding = target_embedding.to(intermediate_state.device)

        # Calculate alignment loss
        state_representation = intermediate_state.mean(
            dim=1
        )  # Average over sequence length
        student_proj = self.student_proj(state_representation)
        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(target_embedding, dim=-1)
        latent_loss = F.mse_loss(student_norm, teacher_norm)

        total_loss = self.latent_weight * latent_loss

        return {
            "total_loss": total_loss,
            "latent_loss": latent_loss.item(),
            "loop_idx": loop_idx,
        }

    def calculate_generation_loss(self, model_output, targets, tokenizer):
        """Calculate loss for final answer generation"""
        logits = model_output["logits"]
        answer = targets["answer"]

        # Truncate to matching length
        trunc_length = min(logits.size(1), answer.size(1))
        logits = logits[:, :trunc_length, :]
        answer = answer[:, :trunc_length]

        # Calculate cross entropy loss
        logits_flat = logits.view(-1, self.vocab_size)
        answer_flat = answer.view(-1)
        gen_loss = self.ce_loss(logits_flat, answer_flat)

        return gen_loss

    def forward(self, model_output, targets, tokenizer):
        # Check if this is an intermediate state loss calculation
        if "intermediate_state" in model_output:
            loop_idx = model_output["loop_idx"]
            # Get target for specific loop and ensure it's a proper tensor
            target_state = targets["intermediate_answers"][0, loop_idx].squeeze()
            loss_dict = self.calculate_intermediate_loss(
                model_output, target_state, tokenizer, loop_idx
            )
            return loss_dict

        # Calculate generation loss for final answer
        gen_loss = self.calculate_generation_loss(model_output, targets, tokenizer)

        # Return combined metrics
        metrics = {
            "total_loss": gen_loss,
            "generation_loss": gen_loss.item(),
            "latent_loss": sum(
                model_output.get("per_loop_latent_losses", [])
            ),  # Sum all loop losses
            "per_loop_latent_losses": model_output.get("per_loop_latent_losses", []),
        }

        return metrics

    def calculate_generation_loss(self, model_output, targets, tokenizer):
        """Calculate loss for final answer generation"""
        logits = model_output["logits"]
        answer = targets["answer"]

        # Truncate to matching length
        trunc_length = min(logits.size(1), answer.size(1))
        logits = logits[:, :trunc_length, :]
        answer = answer[:, :trunc_length]

        # Calculate cross entropy loss
        logits_flat = logits.view(-1, self.vocab_size)
        answer_flat = answer.view(-1)
        gen_loss = self.ce_loss(logits_flat, answer_flat)

        return gen_loss

    def forward(self, model_output, targets, tokenizer):
        # Check if this is an intermediate state loss calculation
        if "intermediate_state" in model_output:
            loop_idx = model_output["loop_idx"]
            # Get target for specific loop and ensure it's a proper tensor
            target_state = targets["intermediate_answers"][0, loop_idx].squeeze()
            return self.calculate_intermediate_loss(
                model_output, target_state, tokenizer, loop_idx
            )

        # Calculate generation loss for final answer
        gen_loss = self.calculate_generation_loss(model_output, targets, tokenizer)

        # Return combined metrics
        return {
            "total_loss": gen_loss,
            "generation_loss": gen_loss.item(),
            "latent_loss": 0.0,  # No latent loss for final generation
            "per_loop_latent_losses": [],  # Empty for final generation
        }

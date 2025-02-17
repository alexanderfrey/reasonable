from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Union, Tuple, Any


class ModelingStrategy(ABC):
    """Abstract base class for language modeling strategies"""

    @abstractmethod
    def compute_loss(self, logits, targets, **kwargs):
        """Compute the loss for this modeling strategy"""
        pass

    @abstractmethod
    def generate(self, model, input_ids, max_length, **kwargs):
        """Generate tokens using this modeling strategy"""
        pass


class MixedStrategy(ModelingStrategy):
    def __init__(self, tokenizer, mask_token_id, mixing_ratio=0.5):
        super().__init__()
        self.next_token_strategy = NextTokenStrategy(tokenizer)
        self.span_mask_strategy = SpanMaskingStrategy(
            mask_token_id=mask_token_id,
            max_span_length=5,
            min_span_length=1,
            masking_ratio=0.15,
        )
        self.mixing_ratio = mixing_ratio

    def compute_loss(self, logits, targets, **kwargs):
        next_token_loss = self.next_token_strategy.compute_loss(logits, targets)
        span_mask_loss = self.span_mask_strategy.compute_loss(logits, targets)
        return (
            self.mixing_ratio * next_token_loss
            + (1 - self.mixing_ratio) * span_mask_loss
        )

    def generate(self, model, input_ids, max_length, **kwargs):
        # Use next-token strategy for generation
        return self.next_token_strategy.generate(model, input_ids, max_length, **kwargs)


class NextTokenStrategy(ModelingStrategy):
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer

    def compute_loss(self, logits, targets, **kwargs):
        """
        Compute loss for next token prediction.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size] 
                   containing prediction for each position
            targets: Tensor of shape [batch_size, seq_len] 
                    containing target tokens for each position
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        if isinstance(logits, tuple):
            logits, _ = logits
            
        # Validate input shapes
        if len(logits.shape) != 3:
            raise ValueError(
                f"Expected logits shape [batch_size, seq_len, vocab_size], got {logits.shape}"
            )
            
        if len(targets.shape) != 2:
            raise ValueError(
                f"Expected targets shape [batch_size, seq_len], got {targets.shape}"
            )
            
        batch_size, seq_len, vocab_size = logits.shape
        target_batch, target_seq = targets.shape
        
        if batch_size != target_batch or seq_len != target_seq:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}"
            )
            
        # Reshape logits and targets for loss computation
        # From [batch_size, seq_len, vocab_size] to [batch_size * seq_len, vocab_size]
        logits_view = logits.view(-1, vocab_size)
        # From [batch_size, seq_len] to [batch_size * seq_len]
        targets_view = targets.view(-1)
        
        return F.cross_entropy(logits_view, targets_view, ignore_index=-100)

    def compute_all_losses(self, main_logits, aux_outputs, targets, future_targets):
        """
        Compute losses for main prediction and auxiliary heads.
        
        Args:
            main_logits: Tensor of shape [batch_size, seq_len, vocab_size]
            aux_outputs: List of tensors, each [batch_size, seq_len, vocab_size]
            targets: Tensor of shape [batch_size, seq_len]
            future_targets: List of tensors, each [batch_size, seq_len]
            
        Returns:
            tuple: (main_loss, list_of_aux_losses)
        """
        # Compute main loss
        main_loss = self.compute_loss(main_logits, targets)
        aux_losses = []
        
        # Process auxiliary heads if present
        if aux_outputs and future_targets:
            if len(aux_outputs) != len(future_targets):
                raise ValueError(
                    f"Number of aux_outputs ({len(aux_outputs)}) must match "
                    f"number of future_targets ({len(future_targets)})"
                )
                
            for idx, (aux_output, future_target) in enumerate(zip(aux_outputs, future_targets)):
                # Validate shapes
                if aux_output.shape != main_logits.shape:
                    raise ValueError(
                        f"Auxiliary output {idx} should have same shape as main logits, "
                        f"got {aux_output.shape} vs {main_logits.shape}"
                    )
                
                if future_target.shape != targets.shape:
                    raise ValueError(
                        f"Future target {idx} should have same shape as main targets, "
                        f"got {future_target.shape} vs {targets.shape}"
                    )
                
                aux_loss = self.compute_loss(aux_output, future_target)
                aux_losses.append(aux_loss)
                
        return main_loss, aux_losses

    def generate(self, model, input_ids, max_length, top_p=0.9, temperature=1.0, **kwargs):
        """Generate next tokens using main and auxiliary heads.
        
        Note: During generation, we only use the last position's prediction.
        """
        device = next(model.parameters()).device
        tokens = input_ids.clone().to(device)
        
        with torch.no_grad():
            outputs = model(tokens)
            
            if isinstance(outputs, tuple):
                main_logits, aux_outputs = outputs
            else:
                main_logits = outputs
                aux_outputs = []
            
            # For generation, we only care about the last position
            next_token_logits = main_logits[:, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            next_token = self.sample_top_p(next_token_logits, top_p)
            
            # Ensure next_token has sequence dimension
            if len(next_token.shape) == 1:
                next_token = next_token.unsqueeze(1)
                
            generated_sequence = torch.cat([tokens, next_token], dim=1)
        
        return generated_sequence

    def sample_top_p(self, logits, top_p):
        """Sample from the top-p probability distribution."""
        if len(logits.shape) != 2:
            raise ValueError(f"Expected 2D logits tensor [batch_size, vocab_size], got shape {logits.shape}")
            
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        mask = cumsum_probs <= top_p
        mask[:, 0] = True  # Always include top probability token
        
        filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        sample_indices = torch.multinomial(filtered_probs, num_samples=1)
        selected_indices = torch.gather(sorted_indices, -1, sample_indices)
        
        return selected_indices

class TeacherForcingStrategy(NextTokenStrategy):
    def __init__(self, tokenizer=None, teacher_forcing_ratio=0.4, generate_length=8):
        super().__init__(tokenizer=tokenizer)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.generate_length = generate_length

    def compute_all_losses(self, main_logits, aux_outputs, targets, future_targets, 
                          model=None, input_ids=None, attention_mask=None, **kwargs):
        """
        Compute losses with teacher forcing, respecting auxiliary head offsets.
        
        Args:
            main_logits: [batch_size, seq_len, vocab_size]
            aux_outputs: List of [batch_size, seq_len, vocab_size] tensors
            targets: [batch_size, seq_len]
            future_targets: List of [batch_size, seq_len] tensors, each shifted by one more position
            model: The language model
            input_ids: Original input sequence
            attention_mask: Optional attention mask
        """
        # Standard next token prediction losses
        main_loss, aux_losses = super().compute_all_losses(
            main_logits, aux_outputs, targets, future_targets
        )
        
        # Skip teacher forcing if no model provided or during validation
        if model is None or input_ids is None or not aux_outputs:
            return main_loss, aux_losses
            
        # Randomly apply teacher forcing
        if torch.rand(1).item() > self.teacher_forcing_ratio:
            return main_loss, aux_losses
            
        batch_size, seq_len = input_ids.size()
        n_aux_heads = len(aux_outputs)
        
        # Context window needs to account for the maximum future prediction
        context_window = seq_len - self.generate_length - n_aux_heads
        
        # Use first part as context
        context_ids = input_ids[:, :context_window]
        context_mask = attention_mask[:, :context_window] if attention_mask is not None else None
        
        # Generate continuation
        generated_sequences = self._generate_sequences(model, context_ids, context_mask)
        
        # Compute losses on generated sequence
        gen_main_loss, gen_aux_losses = self._compute_generation_losses(
            model,
            generated_sequences,
            targets[:, context_window:context_window + self.generate_length],
            [ft[:, context_window:context_window + self.generate_length] for ft in future_targets],
            attention_mask=attention_mask[:, context_window:context_window + self.generate_length] 
                if attention_mask is not None else None
        )
        
        # Combine losses
        combined_main_loss = main_loss + 0.5 * gen_main_loss
        combined_aux_losses = [
            aux_loss + 0.5 * gen_aux_loss 
            for aux_loss, gen_aux_loss in zip(aux_losses, gen_aux_losses)
        ]
        
        return combined_main_loss, combined_aux_losses

    def _compute_generation_losses(self, model, generated_sequences, targets, future_targets, attention_mask=None):
        """
        Compute losses on generated sequences for all heads.
        
        Args:
            model: The language model
            generated_sequences: Generated token sequences
            targets: Main target sequence
            future_targets: List of future target sequences for auxiliary heads
            attention_mask: Optional attention mask
        """
        # Get model predictions
        if attention_mask is not None:
            outputs = model(generated_sequences, attention_mask=attention_mask)
        else:
            outputs = model(generated_sequences)
            
        if isinstance(outputs, tuple):
            main_logits, aux_outputs = outputs
        else:
            main_logits = outputs
            aux_outputs = []
            
        # Compute main loss, handling padding
        main_loss = super().compute_loss(main_logits, targets)
        
        # Compute auxiliary losses, maintaining proper offsets
        aux_losses = []
        if aux_outputs and future_targets:
            for aux_output, future_target in zip(aux_outputs, future_targets):
                # Handle padding (-100) in future targets
                aux_loss = super().compute_loss(aux_output, future_target)
                aux_losses.append(aux_loss)
                
        return main_loss, aux_losses

    def _generate_sequences(self, model, context_ids, attention_mask=None):
        """Generate continuation sequences."""
        current_ids = context_ids
        current_mask = attention_mask
        
        generated_tokens = []
        
        for _ in range(self.generate_length):
            with torch.no_grad():
                # Forward pass
                if current_mask is not None:
                    outputs = model(current_ids, attention_mask=current_mask)
                else:
                    outputs = model(current_ids)
                    
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Sample next token
                next_token_logits = logits[:, -1, :]
                next_tokens = self.sample_top_p(next_token_logits, top_p=0.9)
                
                # Update sequences
                current_ids = torch.cat([current_ids, next_tokens], dim=1)
                if current_mask is not None:
                    current_mask = torch.cat([
                        current_mask, 
                        torch.ones_like(next_tokens, dtype=current_mask.dtype)
                    ], dim=1)
                    
                generated_tokens.append(next_tokens)
        
        return torch.cat(generated_tokens, dim=1)

    def generate(self, model, input_ids, max_length, top_p=0.9, temperature=1.0, **kwargs):
        """Generate tokens using main head predictions."""
        return super().generate(
            model, input_ids, max_length, top_p=top_p, temperature=temperature, **kwargs
        )
class InstructionFollowingStrategy(ModelingStrategy):
    """Strategy specialized for instruction-following tasks with proper sequence length handling"""

    def __init__(
        self,
        tokenizer,
        instruction_token="[INST]",
        response_token="[/INST]",
        end_token="</s>",
        max_length=1024,
        pad_token_id=0,
        format_type="alpaca",  # Can be "alpaca" or "thinking"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.instruction_token = instruction_token
        self.response_token = response_token
        self.end_token = end_token
        self.max_length = max_length
        self.format_type = format_type

        # Cache token IDs
        self.instruction_token_id = self.tokenizer.encode(instruction_token)[0]
        self.response_token_id = self.tokenizer.encode(response_token)[0]
        self.end_token_id = self.tokenizer.encode(end_token)[0]

        self.pad_token_id = pad_token_id
        self.device = None

    def get_padding_value(self):
        """Return the padding token ID"""
        return self.pad_token_id

    def format_instruction(self, instruction, response, input_context=None, thinking=None, system=None):
        """Format instruction including thinking tags within the response section"""
        
        # Helper function to convert various types to string
        def to_string(value):
            if value is None:
                return ""
            elif isinstance(value, (list, tuple)):
                return "\n".join(str(item) for item in value)
            elif isinstance(value, dict):
                return "\n".join(f"{k}: {v}" for k, v in value.items())
            else:
                return str(value)
        
        # Convert all inputs to strings
        instruction = to_string(instruction)
        response = to_string(response)
        input_context = to_string(input_context) if input_context is not None else None
        thinking = to_string(thinking) if thinking is not None else None
        system = to_string(system) if system is not None else None
        
        # Build formatted string
        formatted = f"{self.instruction_token}"
        
        if system:
            formatted += f"### System:\n{system}\n\n"
        
        formatted += f"### Instruction:\n{instruction}\n"
        
        if input_context:
            formatted += f"\n### Input:\n{input_context}\n"
        
        formatted += "\n### Response:\n"
        
        # Add thinking section if provided
        if thinking:
            formatted += f"<thinking>{thinking}</thinking>\n"
        
        # Add the actual response
        formatted += response
        
        # Add closing tokens
        formatted += f"{self.response_token}{self.end_token}"
        
        return formatted

    def _format_alpaca(
        self,
        instruction: str,
        response: str,
        input_context: str = None,
        system: str = None,
    ) -> str:
        """Format using Alpaca style with optional system prompt"""
        formatted = f"{self.instruction_token}"

        # Add system prompt if provided
        if system:
            formatted += f"### System:\n{system}\n\n"

        formatted += f"### Instruction:\n{instruction}\n"

        if input_context:
            formatted += f"\n### Input:\n{input_context}\n"

        formatted += f"\n### Response:\n{response}{self.response_token}{self.end_token}"
        return formatted

    def to(self, device):
        """Add device management method"""
        self.device = device
        return self

    def reset_state(self):
        """Reset any internal state"""
        self.generation_history = []
        self.current_sequence = None

    def get_state(self):
        """Return current state for debugging"""
        return {
            "device": self.device,
            "max_length": self.max_length,
            "current_sequence": self.current_sequence,
        }

    def compute_loss(self, logits, targets, instruction_mask=None, **kwargs):
        """Compute loss with proper handling of padding tokens"""
        if instruction_mask is None:
            instruction_mask = self._create_instruction_mask_vectorized(targets)

        # Create padding mask (1 for non-padding tokens, 0 for padding)
        padding_mask = (targets != self.pad_token_id).float()
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1), 
            reduction="none",
            ignore_index=self.pad_token_id  # Ignore padding tokens in loss
        ).view_as(targets)

        # Apply both instruction and padding masks
        final_mask = (~instruction_mask).float() * padding_mask
        masked_loss = loss * final_mask
        
        # Normalize by number of non-masked tokens
        return masked_loss.sum() / (final_mask.sum() + 1e-8)
    
    
    def compute_all_losses(self, main_logits, aux_outputs, targets, future_targets):
        return self.compute_loss(main_logits, targets), []

    def _create_instruction_mask_vectorized(self, token_ids):
        """Create instruction mask with padding consideration"""
        batch_size, seq_length = token_ids.shape
        inst_positions = (token_ids == self.instruction_token_id).nonzero(as_tuple=True)
        resp_positions = (token_ids == self.response_token_id).nonzero(as_tuple=True)
        pad_positions = (token_ids == self.pad_token_id).nonzero(as_tuple=True)

        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for b in range(batch_size):
            batch_insts = inst_positions[0] == b
            batch_resps = resp_positions[0] == b
            batch_pads = pad_positions[0] == b
            
            if batch_insts.any() and batch_resps.any():
                start = inst_positions[1][batch_insts][0]
                end = resp_positions[1][batch_resps][0]
                mask[b, start:end + 1] = True
            
            # Mask out padding tokens
            if batch_pads.any():
                pad_start = pad_positions[1][batch_pads][0]
                mask[b, pad_start:] = False

        return mask

    def generate(self, model, input_ids, max_length, top_p=0.9, temperature=0.7, debug=False):
        """Generate with proper handling of special tokens and removing padding"""
        self.to(input_ids.device)
        
        # Remove padding tokens from input sequence
        padding_mask = input_ids[0] != self.pad_token_id
        if not padding_mask.all():
            # Find last non-padding token
            last_token_pos = padding_mask.nonzero()[-1]
            input_ids = input_ids[:, :last_token_pos + 1]
            if debug:
                print(f"Removed padding. New input length: {input_ids.size(1)}")
        
        tokens = input_ids.clone()
        response_started = False
        
        with torch.no_grad():
            for step in range(max_length):
                # Get model outputs
                outputs = model(tokens) if not response_started else model(tokens[:, -1:])
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[:, -1, :]

                # After [/INST], prevent generating instruction-related tokens
                if response_started:
                    next_token_logits[:, self.instruction_token_id] = float('-inf')
                    next_token_logits[:, self.response_token_id] = float('-inf')
                    next_token_logits[:, self.pad_token_id] = float('-inf')
                
                # Sample next token
                scaled_logits = next_token_logits / temperature
                next_token = self.sample_top_p(scaled_logits, top_p)
                
                if debug:
                    try:
                        print(f"Step {step}: Generated token {next_token.item()} -> '{self.tokenizer.decode([next_token.item()])}'")
                    except Exception as e:
                        print(f"Step {step}: Error decoding token {next_token.item()}: {e}")
                
                # Check if we're starting the response
                if not response_started and next_token.item() == self.response_token_id:
                    response_started = True
                    if debug:
                        print("Response generation started")
                
                # Add token to sequence
                tokens = torch.cat([tokens, next_token], dim=1)
                
                # Stop if we generate end token or pad token
                if next_token.item() in [self.end_token_id, self.pad_token_id]:
                    if debug:
                        print(f"Stopping generation: Generated {'end' if next_token.item() == self.end_token_id else 'pad'} token")
                    break

                # Check sequence length
                if tokens.size(1) >= self.max_length:
                    if debug:
                        print(f"Stopping generation: Reached maximum length {self.max_length}")
                    break

        return tokens

    @staticmethod
    def sample_top_p(logits, top_p):
        """Sample from top-p (nucleus) filtered distribution"""
        # Ensure logits are 2D: [batch_size, vocab_size]
        if len(logits.shape) == 3:
            logits = logits.squeeze(1)

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = 0

        # Scatter mask back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        # Filter distribution
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample from filtered distribution
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token  # Shape: [batch_size, 1]

    def collect_metrics(self):
        """Collect generation metrics"""
        return {
            "avg_sequence_length": sum(len(s) for s in self.generation_history)
            / len(self.generation_history),
            "total_generations": len(self.generation_history),
            "unique_tokens_used": len(
                set(t for s in self.generation_history for t in s)
            ),
        }


class SpanMaskingStrategy(ModelingStrategy):
    """Strategy that masks spans of tokens for MLM-style training"""

    def __init__(
        self,
        mask_token_id=50256,  # Default to GPT's EOS as mask token
        max_span_length=5,
        min_span_length=1,
        masking_ratio=0.15,
        geometric_p=0.2,  # Parameter for geometric distribution
        mask_entire_words=True,
    ):
        self.mask_token_id = mask_token_id
        self.max_span_length = max_span_length
        self.min_span_length = min_span_length
        self.masking_ratio = masking_ratio
        self.geometric_p = geometric_p
        self.mask_entire_words = mask_entire_words

    def _sample_span_length(self):
        """Sample span length from truncated geometric distribution"""
        span_length = np.random.geometric(self.geometric_p)
        return min(max(span_length, self.min_span_length), self.max_span_length)

    def _get_word_boundaries(self, tokens):
        """Find word boundaries in token sequence (for tokenizers that can split words)"""
        # This is a simplified heuristic - adjust based on your tokenizer
        is_boundary = torch.ones_like(tokens, dtype=torch.bool)

        # Mark continuation tokens as non-boundaries
        # This assumes continuation tokens start with specific patterns
        # Adjust these patterns based on your tokenizer
        for i in range(1, len(tokens)):
            token = tokens[i].item()
            # Example: GPT-2 continuation tokens often start with non-ascii chars
            if token > 256:  # Basic heuristic for GPT-2
                is_boundary[i] = False

        return is_boundary

    def _select_spans(self, tokens, target_num_tokens):
        """Select spans to mask while respecting word boundaries if needed"""
        device = tokens.device
        seq_length = len(tokens)
        spans = []
        num_masked = 0

        if self.mask_entire_words:
            word_boundaries = self._get_word_boundaries(tokens)
        else:
            word_boundaries = torch.ones_like(tokens, dtype=torch.bool)

        # Keep selecting spans until we hit target number of tokens
        while num_masked < target_num_tokens and len(spans) < 100:  # Safety limit
            # Sample span length
            span_length = self._sample_span_length()

            # Get valid start positions (must be word boundaries if mask_entire_words)
            valid_starts = []
            for i in range(seq_length - span_length + 1):
                if word_boundaries[i]:
                    # Check if span would overlap with existing spans
                    overlap = False
                    span_end = i + span_length
                    for start, end in spans:
                        if i < end and start < span_end:
                            overlap = True
                            break
                    if not overlap:
                        valid_starts.append(i)

            if not valid_starts:
                break

            # Randomly select start position
            start = random.choice(valid_starts)
            end = start + span_length

            # Add span
            spans.append((start, end))
            num_masked += span_length

            if num_masked >= target_num_tokens:
                break

        return spans

    def prepare_masked_input(self, input_ids):
        """Create masked input and targets for training"""
        device = input_ids.device
        batch_size, seq_length = input_ids.shape

        # Initialize masked inputs as copy of inputs
        masked_inputs = input_ids.clone()

        # Create target tensor (-100 indicates ignored positions)
        targets = torch.full_like(input_ids, -100)

        # Process each sequence in batch
        for i in range(batch_size):
            # Calculate number of tokens to mask
            num_tokens = int(seq_length * self.masking_ratio)

            # Select spans to mask
            spans = self._select_spans(input_ids[i], num_tokens)

            # Apply masking
            for start, end in spans:
                # Store original tokens in targets
                targets[i, start:end] = input_ids[i, start:end]
                # Replace with mask tokens in input
                masked_inputs[i, start:end] = self.mask_token_id

        return masked_inputs.to(device), targets.to(device)

    def compute_loss(self, logits, targets, **kwargs):
        """Compute MLM loss for masked spans"""
        device = logits.device
        logits = logits.to(device)
        targets = targets.to(device)

        # Only compute loss for masked positions
        masked_positions = targets != -100

        if not masked_positions.any():
            return torch.tensor(0.0, device=device)

        # Gather logits and targets for masked positions
        masked_logits = logits[masked_positions]
        masked_targets = targets[masked_positions]

        # Compute cross entropy loss
        loss = F.cross_entropy(
            masked_logits.view(-1, logits.size(-1)), masked_targets.view(-1)
        )

        return loss

    def generate(self, model, input_ids, max_length, temperature=1.0, top_p=0.9):
        """Generate completions for both masked spans and next tokens"""
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        # First generate for any masked positions if present
        mask_positions = (tokens == self.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) > 0:
            for pos in mask_positions[1]:
                with torch.no_grad():
                    logits = model(tokens)[0]
                    next_token_logits = logits[0, pos : pos + 1, :].to(device)
                    next_token_logits = next_token_logits / temperature
                    next_token = self.sample_top_p(next_token_logits, top_p)
                    tokens[0, pos] = next_token

        # Then generate new tokens at the end
        original_length = tokens.size(1)
        for _ in range(max_length - original_length):
            with torch.no_grad():
                # Get model predictions
                logits = model(tokens)[0]
                next_token_logits = logits[0, -1:, :].to(device)

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Sample next token
                next_token = self.sample_top_p(next_token_logits, top_p)

                # Append to sequence
                tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

                # Stop if we generate an end token
                if next_token.item() == 50256:  # End token
                    break

        return tokens

    @staticmethod
    def sample_top_p(logits, top_p):
        """Sample from the top-p probability mass"""
        device = logits.device
        probs = F.softmax(logits, dim=-1)
        probs = probs.to(device)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        probs = probs.masked_fill(indices_to_remove, 0.0)
        return torch.multinomial(probs, num_samples=1)


class SpanPredictionStrategy(ModelingStrategy):
    """Predict spans of tokens at once"""

    def __init__(self, span_length=3):
        self.span_length = span_length

    def compute_loss(self, logits, targets, **kwargs):
        # Get device from input tensors
        device = logits.device

        # Ensure inputs are on the correct device
        logits = logits.to(device)
        targets = targets.to(device)

        # Reshape for span prediction
        batch_size, seq_len, vocab_size = logits.shape
        spans = seq_len // self.span_length

        # Perform reshaping operations with device awareness
        logits = (
            logits[:, : spans * self.span_length, :]
            .view(batch_size, spans, self.span_length, vocab_size)
            .to(device)
        )

        targets = (
            targets[:, : spans * self.span_length]
            .view(batch_size, spans, self.span_length)
            .to(device)
        )

        return F.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100
        )

    def generate(self, model, input_ids, max_length, top_p=0.9):
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        while tokens.size(1) < max_length:
            # Generate span_length tokens at once
            with torch.no_grad():
                # Ensure tokens are on correct device before model forward pass
                tokens = tokens.to(device)
                logits = model(tokens)[0]

                # Handle logits device placement
                span_logits = logits[0, -1:, :].to(device)
                span_logits = span_logits.expand(1, self.span_length, -1)

                # Sample tokens for the span
                span_tokens = []
                for i in range(self.span_length):
                    next_token = self.sample_top_p(span_logits[:, i, :], top_p)
                    next_token = next_token.to(device)
                    span_tokens.append(next_token)

                # Ensure consistent device placement during concatenation
                span_tensor = torch.cat(span_tokens, dim=1).to(device)
                tokens = torch.cat([tokens, span_tensor], dim=1)

                if 50256 in span_tensor:  # END_TOKEN
                    break

        return tokens

    @staticmethod
    def sample_top_p(logits, top_p):
        return NextTokenStrategy.sample_top_p(logits, top_p)


class GRPOStrategy(NextTokenStrategy):
    """
    Implements Group Relative Policy Optimization (GRPO) for token sampling.
    GRPO optimizes policies by comparing relative performance within groups
    of similar contexts/states rather than using absolute reward values.
    """

    def __init__(
        self,
        tokenizer=None,
        group_size: int = 16,
        learning_rate: float = 1e-4,
        relative_clip: float = 0.2,
        entropy_weight: float = 0.01,
        context_embedding_dim: int = 768,
        wait_token_id: Optional[int] = None,  # Add wait_token_id
        uncertainty_threshold: float = 2.0,  # Add uncertainty_threshold, tune this
        exploration_temperature_after_wait: float = 1.5,  # Add exploration temp after wait, tune
        wait_token_exploration_steps: int = 3,  # Steps to explore after wait, tune
    ):
        super().__init__(tokenizer=tokenizer)
        self.group_size = group_size
        self.relative_clip = relative_clip
        self.entropy_weight = entropy_weight
        self.wait_token_id = wait_token_id  # Store wait_token_id
        self.uncertainty_threshold = (
            uncertainty_threshold  # Store uncertainty_threshold
        )
        self.exploration_temperature_after_wait = (
            exploration_temperature_after_wait  # Store exploration temp
        )
        self.wait_token_exploration_steps = (
            wait_token_exploration_steps  # Store exploration steps
        )
        self.current_exploration_steps_remaining = (
            0  # Track exploration steps after wait
        )

        # Policy network for sampling decisions
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(
                context_embedding_dim + 2, 128
            ),  # context + entropy signals
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),  # Outputs relative advantage estimate
        )

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )

    def compute_context_similarity(
        self, context_embeddings: torch.Tensor, reference_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between a reference context and a batch of contexts.
        Used for grouping similar contexts together.
        """
        # Normalize embeddings
        context_embeddings = F.normalize(context_embeddings, dim=-1)
        reference_embedding = F.normalize(reference_embedding, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(
            context_embeddings, reference_embedding.transpose(-2, -1)
        )
        return similarity.squeeze(-1)

    def form_context_groups(
        self, context_embeddings: torch.Tensor, entropy_signals: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Form groups of similar contexts for relative optimization.
        Returns list of groups, each containing context embeddings and entropy signals.
        """
        num_contexts = context_embeddings.size(0)
        groups = []
        used_indices = set()

        while len(used_indices) < num_contexts:
            # Find unused reference context
            reference_idx = None
            for i in range(num_contexts):
                if i not in used_indices:
                    reference_idx = i
                    break

            if reference_idx is None:
                break

            # Compute similarities with reference
            similarities = self.compute_context_similarity(
                context_embeddings,
                context_embeddings[reference_idx : reference_idx + 1],
            )

            # Find most similar contexts not yet used
            _, similar_indices = similarities.topk(
                min(self.group_size, num_contexts - len(used_indices))
            )

            # Filter out already used indices
            group_indices = [
                idx.item() for idx in similar_indices if idx.item() not in used_indices
            ]

            # Form group
            group = {
                "embeddings": context_embeddings[group_indices],
                "entropy_signals": entropy_signals[group_indices],
            }
            groups.append(group)

            # Mark indices as used
            used_indices.update(group_indices)

        return groups

    def compute_relative_advantages(
        self, group: Dict[str, torch.Tensor], outcomes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative advantages within a group based on outcomes.
        """
        # Get policy values for group
        policy_input = torch.cat(
            [group["embeddings"], group["entropy_signals"]], dim=-1
        )

        policy_values = self.policy_net(policy_input).squeeze(-1)

        # Compute relative advantages
        outcome_ranks = torch.argsort(torch.argsort(outcomes))
        outcome_ranks = outcome_ranks.float() / (
            len(outcomes) - 1
        )  # Normalize to [0, 1]

        relative_advantages = outcome_ranks - policy_values
        return relative_advantages

    def update_policy(
        self, groups: List[Dict[str, torch.Tensor]], all_outcomes: List[torch.Tensor]
    ):
        """
        Update policy using GRPO algorithm on grouped contexts.
        """
        total_loss = 0

        for group, outcomes in zip(groups, all_outcomes):
            relative_advantages = self.compute_relative_advantages(group, outcomes)

            # Get policy values
            policy_input = torch.cat(
                [group["embeddings"], group["entropy_signals"]], dim=-1
            )

            policy_values = self.policy_net(policy_input).squeeze(-1)

            # Compute GRPO loss with clipping
            ratio = torch.exp(policy_values - policy_values.detach())
            surr1 = ratio * relative_advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.relative_clip, 1 + self.relative_clip)
                * relative_advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean()

            # Add entropy regularization
            probs = F.softmax(policy_values, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            loss = policy_loss - self.entropy_weight * entropy

            total_loss += loss

        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,  # Standard temperature
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens using GRPO-guided sampling with 'wait' token injection.
        """
        device = input_ids.device
        tokens = input_ids.clone()

        # Track context information for group formation
        context_embeddings = []
        entropy_signals = []
        outcomes = []

        self.current_exploration_steps_remaining = (
            0  # Reset exploration counter for each generation
        )

        while tokens.size(1) < max_length:
            with torch.no_grad():
                outputs = model(tokens)
                if isinstance(outputs, tuple):
                    logits, hidden_states = outputs
                else:
                    logits = outputs
                    hidden_states = None

                next_token_logits = logits[0, -1:, :].to(device)

                # Get context embedding and entropy signals
                if hidden_states is not None:
                    context_emb = hidden_states[-1][:, -1]  # Last layer, last position
                else:
                    context_emb = self.token_embedding(tokens)[:, -1]

                entropy = self.compute_entropy(next_token_logits)
                varentropy = self.compute_varentropy(model, tokens)
                entropy_signal = torch.cat([entropy, varentropy.unsqueeze(0)], dim=-1)

                # Store for group formation
                context_embeddings.append(context_emb)
                entropy_signals.append(entropy_signal)

                # Check for 'wait' token injection condition
                if (
                    self.wait_token_id is not None
                    and entropy > self.uncertainty_threshold
                    and self.current_exploration_steps_remaining <= 0
                ):  # Check if we are not already in exploration phase
                    next_token = torch.tensor(
                        [[self.wait_token_id]], device=device
                    )  # Force 'wait' token
                    current_temperature = (
                        self.exploration_temperature_after_wait
                    )  # Set exploration temp
                    self.current_exploration_steps_remaining = (
                        self.wait_token_exploration_steps
                    )  # Set exploration steps counter
                    print(
                        f"Wait token injected at step {tokens.size(1)}, entropy: {entropy.item()}"
                    )  # Optional print for debugging
                else:
                    # Determine temperature, using exploration temp if in exploration phase, otherwise policy-adjusted temp
                    if self.current_exploration_steps_remaining > 0:
                        current_temperature = self.exploration_temperature_after_wait
                    else:
                        policy_input = torch.cat([context_emb, entropy_signal], dim=-1)
                        policy_value = self.policy_net(policy_input)
                        current_temperature = (
                            temperature * torch.sigmoid(policy_value).item()
                        )  # Policy adjusted temp

                    next_token = self.sample_top_p(
                        next_token_logits / current_temperature, top_p
                    )
                    self.current_exploration_steps_remaining = max(
                        0, self.current_exploration_steps_remaining - 1
                    )  # Decrement exploration counter

                # Track outcome (e.g., token likelihood)
                probs = F.softmax(
                    next_token_logits / temperature, dim=-1
                )  # Use standard temperature for outcome calculation for consistency? Or current_temperature? - using standard temp for now
                outcome = torch.log(probs[0, 0, next_token.item()])
                outcomes.append(outcome)

                # Append token and check for EOS
                tokens = torch.cat([tokens, next_token], dim=1)
                if next_token.item() == 50256:  # Assuming GPT-2 tokenizer
                    break

        # Form groups and update policy
        if context_embeddings:
            context_embeddings = torch.stack(context_embeddings)
            entropy_signals = torch.stack(entropy_signals)
            outcomes = torch.stack(outcomes)

            groups = self.form_context_groups(context_embeddings, entropy_signals)
            all_outcomes = []

            # Split outcomes by group
            start_idx = 0
            for group in groups:
                group_size = group["embeddings"].size(0)
                all_outcomes.append(outcomes[start_idx : start_idx + group_size])
                start_idx += group_size

            self.update_policy(groups, all_outcomes)

        return tokens

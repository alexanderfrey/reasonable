from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Union, Tuple


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
    """Next-token prediction strategy with MoE support and multi-token prediction"""

    def __init__(self, tokenizer=None, predict_last_n=1):
        """
        Initialize the strategy
        Args:
            tokenizer: Tokenizer instance
            predict_last_n: Number of tokens at the end to predict (default: 1)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.predict_last_n = predict_last_n
        self.confidence_history = []

    def compute_loss(self, logits, targets, **kwargs):
        """
        Compute cross entropy loss for next token prediction
        Now supports predicting multiple tokens at the end
        """
        # For MoE models, logits might be packed with auxiliary loss
        if isinstance(logits, tuple):
            logits, _ = logits  # Unpack if MoE model output

        # Reshape logits and targets for loss computation
        _, _, vocab_size = logits.shape

        # Only consider the last n positions for loss computation
        if self.predict_last_n > 1:
            # Take the last n positions of logits and targets
            logits = logits[:, -self.predict_last_n :, :]
            targets = targets[:, -self.predict_last_n :]

        return F.cross_entropy(
            logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100
        )

    def generate(
        self, model, input_ids, max_length, top_p=0.9, temperature=1.0, **kwargs
    ):
        """Generate tokens with consistent device management"""
        # Get device from model and ensure input_ids is on the same device
        device = next(model.parameters()).device
        tokens = input_ids.clone().to(device)

        # Track remaining length for generation
        remaining_length = max_length - tokens.size(1)
        generated_sequence = tokens

        with torch.no_grad():
            while remaining_length > 0:
                # Model forward pass - tokens already on correct device
                outputs = model(generated_sequence)

                # Handle MoE model output
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                # Get logits for last n positions
                n_tokens = min(self.predict_last_n, remaining_length)
                next_token_logits = logits[
                    :, -n_tokens:, :
                ]  # Already on correct device

                # Apply temperature scaling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Generate tokens for all positions at once
                generated_tokens = []
                for position_logits in next_token_logits[
                    0
                ]:  # Iterate over last n positions
                    # Keep logits on device and avoid unnecessary transfers
                    position_logits = position_logits.unsqueeze(
                        0
                    )  # Add batch dimension

                    # Sample next token using top-p filtering
                    next_token = self.sample_top_p(position_logits.unsqueeze(0), top_p)

                    # Compute confidence (all operations remain on device)
                    probs = F.softmax(position_logits, dim=-1)
                    confidence = probs[
                        0, next_token.item()
                    ].item()  # Only transfer scalar to CPU
                    self.confidence_history.append(confidence)

                    generated_tokens.append(next_token)

                    # Check for EOS token - only transfer scalar for comparison
                    if next_token.item() == self.tokenizer.eos_token_id:
                        remaining_length = 0
                        break

                # Combine generated tokens efficiently
                if generated_tokens:
                    next_tokens = torch.cat(generated_tokens, dim=0).unsqueeze(
                        0
                    )  # Keep on device
                    generated_sequence = torch.cat(
                        [generated_sequence, next_tokens], dim=1
                    )
                    remaining_length -= len(generated_tokens)

        return generated_sequence

    @staticmethod
    def sample_top_p(logits, top_p):
        """Sample from top-p (nucleus) filtered distribution with consistent device handling"""
        # Use device from input logits
        device = logits.device
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities and get indices - operations stay on device
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create removal mask for tokens above top_p - stay on device
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create and scatter mask back to original order - stay on device
        unsorted_indices_to_remove = torch.zeros_like(
            sorted_indices_to_remove, dtype=torch.bool, device=device
        )
        unsorted_indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

        # Zero out filtered probabilities and sample - stay on device
        probs = probs.masked_fill(unsorted_indices_to_remove, 0.0)
        return torch.multinomial(probs, num_samples=1)

    def get_stats(self):
        """Return confidence-related metrics"""
        if self.confidence_history:
            avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
            # Add stats specific to multi-token prediction
            stats = {
                "avg_confidence": avg_confidence,
                "predict_last_n": self.predict_last_n,
            }
        else:
            stats = {"avg_confidence": 0.0, "predict_last_n": self.predict_last_n}
        return stats


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

    def format_instruction(
        self,
        instruction: str,
        response: str,
        input_context: str = None,
        thinking: str = None,
        system: str = None,
    ) -> str:
        """Format instruction based on specified format type"""
        # Ensure inputs are strings
        instruction = str(instruction) if instruction is not None else ""
        response = str(response) if response is not None else ""

        if self.format_type == "thinking":
            formatted = self._format_thinking(instruction, thinking, response)
        else:  # default to alpaca format
            formatted = self._format_alpaca(
                instruction, response, input_context, system
            )

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

    def tokenize_and_pad(self, text: str) -> List[int]:
        """Tokenize text and handle padding with our special pad token"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) < self.max_length:
            padding = [self.pad_token_id] * (self.max_length - len(tokens))
            tokens.extend(padding)
        return tokens[: self.max_length]

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
        # Vectorized mask creation if not provided
        if instruction_mask is None:
            instruction_mask = self._create_instruction_mask_vectorized(targets)

        # Use einsum for more efficient computation
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        ).view_as(targets)

        # Apply mask efficiently
        masked_loss = loss * (~instruction_mask).float()
        return masked_loss.sum() / (~instruction_mask).float().sum()

    def _create_instruction_mask_vectorized(self, token_ids):
        """Vectorized mask creation"""
        batch_size, seq_length = token_ids.shape
        inst_positions = (token_ids == self.instruction_token_id).nonzero(as_tuple=True)
        resp_positions = (token_ids == self.response_token_id).nonzero(as_tuple=True)

        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for b in range(batch_size):
            batch_insts = inst_positions[0] == b
            batch_resps = resp_positions[0] == b
            if batch_insts.any() and batch_resps.any():
                start = inst_positions[1][batch_insts][0]
                end = resp_positions[1][batch_resps][0]
                mask[b, start : end + 1] = True
        return mask

    def validate_tokens(self, tokens):
        """Validate token sequence structure"""
        has_inst = self.instruction_token_id in tokens
        has_resp = self.response_token_id in tokens
        has_end = self.end_token_id in tokens

        if not (has_inst and has_resp):
            raise ValueError("Missing instruction or response tokens")

        try:
            inst_pos = tokens.index(self.instruction_token_id)
            resp_pos = tokens.index(self.response_token_id)
        except (
            ValueError
        ):  # Should not happen now because of 'in' checks, but as a safeguard
            raise ValueError("Missing instruction or response tokens (index error)")

        if inst_pos >= resp_pos:
            raise ValueError("Invalid token sequence order")

    def validate_instruction_format(self, text: str) -> bool:
        """Validate instruction format before tokenization"""
        # Check basic structure
        has_inst_token = self.instruction_token in text
        has_resp_token = self.response_token in text
        has_end_token = self.end_token in text

        if not (has_inst_token and has_resp_token):
            print(
                f"Warning: Missing instruction/response tokens in text: {text[:100]}..."
            )
            return False

        # Check token order
        inst_pos = text.find(self.instruction_token)
        resp_pos = text.find(self.response_token)
        end_pos = text.find(self.end_token)

        if not (0 <= inst_pos < resp_pos):
            print(
                f"Warning: Invalid instruction/response token order at positions {inst_pos}, {resp_pos}"
            )
            return False

        if end_pos >= 0 and not (resp_pos < end_pos):
            print(
                f"Warning: End token appears before response token at position {end_pos}"
            )
            return False

        return True

    def prepare_input(self, instruction, response=None):
        """
        Unified method for preparing input sequences.
        Can handle both instruction-only (for inference) and instruction-response pairs (for training).

        Args:
            instruction (str): The instruction text
            response (str, optional): The response text for training

        Returns:
            torch.Tensor: Properly formatted and tokenized sequence
        """
        if response is not None:
            # Training mode - include both instruction and response
            sequence = f"{self.instruction_token}{instruction}{self.response_token}{response}{self.end_token}"
        else:
            # Inference mode - include only instruction
            sequence = f"{self.instruction_token}{instruction}{self.response_token}"

        tokens = self.tokenizer.encode(sequence)

        # Handle length constraints
        if len(tokens) > self.max_length:
            if response is None:
                # For inference, keep space for generation
                tokens = tokens[: self.max_length - 1]
            else:
                # For training, include end token
                tokens = tokens[: self.max_length - 1] + [self.end_token_id]

        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def generate(self, model, input_ids, max_length, top_p=0.9, temperature=0.7):
        self.to(input_ids.device)
        tokens = input_ids.clone()
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = (
                    model(tokens) if not generated_tokens else model(tokens[:, -1:])
                )
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                scaled_logits = next_token_logits / temperature
                next_token = self.sample_top_p(
                    scaled_logits, top_p
                )  # Shape: [batch_size, 1]
                tokens = torch.cat([tokens, next_token], dim=1)
                generated_tokens.append(next_token)

                # Check if end token was generated
                if next_token.item() == self.end_token_id:
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

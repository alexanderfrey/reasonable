from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


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


class NextTokenStrategy(ModelingStrategy):
    """Standard next-token prediction strategy"""

    def compute_loss(self, logits, targets, **kwargs):
        # Standard next token prediction loss
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )

    def generate(self, model, input_ids, max_length, top_p=0.9):
        device = input_ids.device
        tokens = input_ids.clone()

        for _ in range(max_length):
            # Get predictions
            with torch.no_grad():
                # Ensure tokens are on correct device
                tokens = tokens.to(device)
                logits = model(tokens)[0]  # Take main path logits
                next_token_logits = logits[0, -1:, :].to(device)

                # Sample next token
                next_token = self.sample_top_p(next_token_logits, top_p)
                next_token = next_token.to(device)
                tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

                if next_token.item() == 50256:  # END_TOKEN
                    break

        return tokens

    @staticmethod
    def sample_top_p(logits, top_p):
        device = logits.device
        probs = F.softmax(logits, dim=-1)

        # Ensure all tensors are on the same device
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_probs = sorted_probs.to(device)
        sorted_indices = sorted_indices.to(device)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cumulative_probs = cumulative_probs.to(device)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = sorted_indices_to_remove.to(device)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        indices_to_remove = indices_to_remove.to(device)

        probs = probs.masked_fill(indices_to_remove, 0.0)
        return torch.multinomial(probs, num_samples=1)


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
        """Generate completions for masked spans"""
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        # Find masked positions
        mask_positions = (tokens == self.mask_token_id).nonzero(as_tuple=True)

        # Generate for each masked position
        for pos in mask_positions[1]:
            with torch.no_grad():
                tokens = tokens.to(device)
                logits = model(tokens)[0]
                next_token_logits = logits[0, pos : pos + 1, :].to(device)

                # Temperature scaling
                next_token_logits = next_token_logits / temperature

                # Top-p sampling
                next_token = self.sample_top_p(next_token_logits, top_p)
                next_token = next_token.to(device)

                # Replace mask token with generated token
                tokens[0, pos] = next_token

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


class BidirectionalStrategy(ModelingStrategy):
    """Predict tokens in both directions"""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def compute_loss(self, logits, targets, **kwargs):
        device = logits.device

        # Ensure inputs are on correct device
        logits = logits.to(device)
        targets = targets.to(device)

        # Split logits for forward and backward predictions
        forward_logits, backward_logits = torch.chunk(logits, 2, dim=-1)
        forward_logits = forward_logits.to(device)
        backward_logits = backward_logits.to(device)

        # Forward loss
        forward_loss = F.cross_entropy(
            forward_logits.view(-1, forward_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        # Backward loss (reverse the targets)
        backward_targets = torch.flip(targets, [1]).to(device)
        backward_loss = F.cross_entropy(
            backward_logits.view(-1, backward_logits.size(-1)),
            backward_targets.view(-1),
            ignore_index=-100,
        )

        return (forward_loss + backward_loss) / 2

    def generate(self, model, input_ids, max_length, top_p=0.9):
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        # Generate forward
        forward_tokens = self.generate_direction(
            model, tokens, max_length // 2, top_p, forward=True
        )

        # Generate backward
        backward_tokens = self.generate_direction(
            model, tokens, max_length // 2, top_p, forward=False
        )

        # Combine tokens (remove duplicate center tokens)
        # Ensure tensors are on same device before combining
        backward_tokens = backward_tokens.to(device)
        forward_tokens = forward_tokens.to(device)
        combined = torch.cat([backward_tokens[:, :-1], forward_tokens], dim=1)

        return combined.to(device)

    def generate_direction(self, model, input_ids, max_length, top_p, forward=True):
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        for _ in range(max_length):
            with torch.no_grad():
                # Ensure tokens are on correct device before model forward pass
                tokens = tokens.to(device)
                logits = model(tokens)[0]

                # Handle logits for forward/backward direction
                if forward:
                    next_token_logits = logits[0, -1:, : self.vocab_size // 2].to(
                        device
                    )
                else:
                    next_token_logits = logits[0, -1:, self.vocab_size // 2 :].to(
                        device
                    )

                # Sample and ensure token is on correct device
                next_token = self.sample_top_p(next_token_logits, top_p)
                next_token = next_token.to(device)
                tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

                if next_token.item() == 50256:  # END_TOKEN
                    break

        if not forward:
            tokens = torch.flip(tokens, [1])

        return tokens.to(device)

    @staticmethod
    def sample_top_p(logits, top_p):
        return NextTokenStrategy.sample_top_p(logits, top_p)


class InstructionFollowingStrategy(ModelingStrategy):
    """Strategy specialized for instruction-following tasks"""

    def __init__(
        self,
        tokenizer,
        instruction_token="[INST]",
        response_token="[/INST]",
        end_token="</s>",
    ):
        self.tokenizer = tokenizer
        self.instruction_token = instruction_token
        self.response_token = response_token
        self.end_token = end_token

        # Cache token IDs
        self.instruction_token_id = self.tokenizer.encode(instruction_token)[0]
        self.response_token_id = self.tokenizer.encode(response_token)[0]
        self.end_token_id = self.tokenizer.encode(end_token)[0]

    def compute_loss(self, logits, targets, instruction_mask=None, **kwargs):
        device = logits.device
        logits = logits.to(device)
        targets = targets.to(device)

        if instruction_mask is None:
            # Create instruction mask - don't compute loss on instruction tokens
            instruction_mask = self._create_instruction_mask(targets)
            instruction_mask = instruction_mask.to(device)

        # Apply higher loss weight to response tokens
        loss_weights = torch.where(
            instruction_mask,
            torch.tensor(0.0).to(device),  # Don't train on instructions
            torch.tensor(1.0).to(device),
        )  # Full weight on responses

        # Compute cross entropy with masked targets
        masked_targets = torch.where(
            instruction_mask,
            torch.tensor(-100).to(device),  # Ignore instruction tokens
            targets,
        )

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), masked_targets.view(-1), reduction="none"
        )

        # Apply weights and average
        weighted_loss = (loss * loss_weights.view(-1)).mean()
        return weighted_loss

    def generate(self, model, input_ids, max_length, top_p=0.9, temperature=0.7):
        device = input_ids.device
        tokens = input_ids.clone().to(device)

        with torch.no_grad():
            response_started = False
            for _ in range(max_length):
                tokens = tokens.to(device)
                logits = model(tokens)[0]
                next_token_logits = logits[0, -1:, :].to(device)

                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature

                # If we haven't started response yet, force response token
                if not response_started and tokens[-1][-1] == self.instruction_token_id:
                    next_token = torch.tensor([[self.response_token_id]], device=device)
                    response_started = True
                else:
                    # Sample next token
                    next_token = self.sample_top_p(next_token_logits, top_p)
                    next_token = next_token.to(device)

                tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

                # Stop if we hit the end token
                if next_token.item() == self.end_token_id:
                    break

        return tokens

    def _create_instruction_mask(self, tokens):
        """Create mask identifying instruction tokens (True for instruction tokens)"""
        device = tokens.device
        batch_size, seq_len = tokens.shape
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        for i in range(batch_size):
            in_instruction = False
            for j in range(seq_len):
                if tokens[i, j] == self.instruction_token_id:
                    in_instruction = True
                elif tokens[i, j] == self.response_token_id:
                    in_instruction = False
                mask[i, j] = in_instruction

        return mask

    def format_prompt(self, instruction, prior_conversation=None):
        """Format an instruction into the expected prompt structure"""
        if prior_conversation:
            # Include prior conversation with proper formatting
            prompt = prior_conversation + "\n"
        else:
            prompt = ""

        # Add new instruction
        prompt += f"{self.instruction_token}{instruction}{self.response_token}"

        return prompt

    @staticmethod
    def sample_top_p(logits, top_p):
        device = logits.device
        probs = F.softmax(logits, dim=-1)
        probs = probs.to(device)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_probs = sorted_probs.to(device)
        sorted_indices = sorted_indices.to(device)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cumulative_probs = cumulative_probs.to(device)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = sorted_indices_to_remove.to(device)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        indices_to_remove = indices_to_remove.to(device)

        probs = probs.masked_fill(indices_to_remove, 0.0)
        return torch.multinomial(probs, num_samples=1)

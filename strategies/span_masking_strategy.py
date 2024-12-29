from strategies.base_strategy import DataPreparationStrategy
import torch
import random

class SpanMaskingStrategy(DataPreparationStrategy):
    def prepare(self, text, block_size, tokenizer, mask_probability=0.15, max_span_length=3, **kwargs):
        """
        Prepares data for span masking and reconstruction.

        Args:
        - text (str): Input text.
        - block_size (int): Length of each sequence.
        - tokenizer: Tokenizer object.
        - mask_probability (float): Probability of masking a token.
        - max_span_length (int): Maximum length of a masked span.

        Returns:
        - X (torch.Tensor): Input sequences with spans masked.
        - Y (torch.Tensor): Target sequences with original tokens in masked spans.
        """
        # Tokenize the text and convert to a tensor
        encoded = tokenizer.encode(text)
        data = torch.tensor(encoded.ids, dtype=torch.long)

        # Prepare sequences
        X, Y = [], []

        for i in range(0, len(data) - block_size, block_size):
            block = data[i:i + block_size]
            input_ids = block.clone()
            target_ids = block.clone()

            # Determine the number of tokens to mask
            num_tokens = len(block)
            num_masked_tokens = int(mask_probability * num_tokens)
            masked_positions = set()

            # Randomly select spans to mask
            while len(masked_positions) < num_masked_tokens:
                start = random.randint(0, num_tokens - 1)
                span_length = random.randint(1, max_span_length)
                for j in range(span_length):
                    if start + j < num_tokens:
                        masked_positions.add(start + j)

            # Apply masking
            for pos in masked_positions:
                input_ids[pos] = tokenizer.token_to_id("<mask>")  # Replace with [MASK]
                target_ids[pos] = block[pos]  # Retain original token for target

            # Ignore non-masked tokens in the target
            for j in range(len(target_ids)):
                if j not in masked_positions:
                    target_ids[j] = -100

            X.append(input_ids)
            Y.append(target_ids)

        # Convert to tensors
        X = torch.stack(X)
        Y = torch.stack(Y)
        return X, Y
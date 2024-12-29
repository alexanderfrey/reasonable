import torch
import random
from strategies.base_strategy import DataPreparationStrategy

class WholeWordMaskingStrategy(DataPreparationStrategy):
    def prepare(self, text, block_size, tokenizer, mask_probability=0.15, **kwargs):
        """
        Prepares data for Whole Word Masking (WWM).

        Args:
        - text (str): Input text.
        - block_size (int): Length of each sequence.
        - tokenizer: Tokenizer object (with <mask> as a special token in its vocabulary).
        - mask_probability (float): Probability of masking a token.

        Returns:
        - X (torch.Tensor): Input sequences with words masked.
        - Y (torch.Tensor): Target sequences with original tokens for masked words.
        """
        # Tokenize the text and convert to a tensor
        encoded = tokenizer.encode(text)
        data = torch.tensor(encoded.ids, dtype=torch.long)
        X, Y = [], []

        for i in range(0, len(data) - block_size, block_size):
            block = data[i : i + block_size]
            input_ids = block.clone()
            target_ids = torch.full_like(block, -100)  # Initialize all target positions as -100

            # Decode block into words
            # (By default, decode() omits special tokens. We'll display them later in debugging.)
            decoded_block = tokenizer.decode(block.tolist(), skip_special_tokens=False)
            words = decoded_block.split()
            # print(f"Decoded Block: {decoded_block}")

            # Calculate how many words to mask
            num_masked_words = max(1, int(mask_probability * len(words)))
            masked_word_indices = random.sample(range(len(words)), num_masked_words)

            # Mask selected words and align with input/target tokens
            for idx in masked_word_indices:
                word = words[idx]
                subwords = tokenizer.encode(word).ids  # Get subword IDs
                # print(f"Word: {word}, Subwords: {subwords}")

                for subword_id in subwords:
                    positions = (block == subword_id).nonzero(as_tuple=True)
                    for pos in positions[0]:
                        # Replace original token in input_ids with <mask>
                        mask_token_id = tokenizer.token_to_id("<mask>")
                        input_ids[pos] = mask_token_id
                        # Retain the original token ID in target_ids
                        target_ids[pos] = subword_id

            # Debugging: Inspect input and target sequences
            # Pass skip_special_tokens=False so that <mask> appears in the decoded string
            decoded_input = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
            # Filter out -100 from target and decode remaining tokens (also showing special tokens if any)
            decoded_target = tokenizer.decode(
                [token for token in target_ids.tolist() if token != -100],
                skip_special_tokens=False
            )

            # print(f"Input IDs (Masked): {decoded_input}")
            # print(f"Target IDs (Original): {decoded_target}")

            X.append(input_ids)
            Y.append(target_ids)

        # Convert to tensors
        X = torch.stack(X)
        Y = torch.stack(Y)
        return X, Y
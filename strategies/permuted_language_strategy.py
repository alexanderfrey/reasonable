from strategies.base_strategy import DataPreparationStrategy
import torch
import random

class PermutedLanguageModelingStrategy(DataPreparationStrategy):
    def prepare(self, text, block_size, tokenizer, mask_probability=0.15, **kwargs):
        """
        Prepares data for Permuted Language Modeling (PLM).
        
        Args:
        - text (str): The input text.
        - block_size (int): The size of each sequence.
        - tokenizer: Tokenizer object to encode the text.
        - mask_probability (float): Probability of masking a token.
        
        Returns:
        - X (torch.Tensor): Input sequences with tokens in permuted order.
        - Y (torch.Tensor): Target tokens based on the permuted order.
        - perm_mask (torch.Tensor): Mask indicating the permutation order.
        """
        encoded = tokenizer.encode(text)
        data = torch.tensor(encoded.ids, dtype=torch.long)

        # Prepare data
        X, Y, perm_mask_list = [], [], []
        for i in range(len(data) - block_size):
            block = data[i:i + block_size]

            # Generate a random permutation
            permuted_indices = torch.randperm(block_size)

            # Create input and target sequences
            input_ids = block.clone()
            targets = block.clone()
            perm_mask = torch.zeros(block_size, dtype=torch.float32)

            for idx in range(block_size):
                # Determine if the token will be predicted
                if random.random() < mask_probability:
                    perm_mask[idx] = 1.0  # Mark as predicted
                else:
                    targets[idx] = -100  # Ignore this token in the target

            # Reorder tokens based on the permutation
            input_ids = input_ids[permuted_indices]
            targets = targets[permuted_indices]
            perm_mask = perm_mask[permuted_indices]

            X.append(input_ids)
            Y.append(targets)
            perm_mask_list.append(perm_mask)

        # Convert to tensors
        X = torch.stack(X)
        Y = torch.stack(Y)
        perm_mask = torch.stack(perm_mask_list)
        return X, Y, perm_mask
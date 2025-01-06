from strategies.base_strategy import DataPreparationStrategy
import torch

class InstructionFollowingStrategy(DataPreparationStrategy):
    def prepare(self, text_pairs, block_size, tokenizer, **kwargs):
        """
        Prepares data for instruction-following training.

        Args:
        - text_pairs (list of tuples): List of (instruction, context, response) triples.
        - block_size (int): Maximum block size for the input sequence.
        - tokenizer: Tokenizer object.

        Returns:
        - X (torch.Tensor): Input sequences (instruction + context).
        - Y (torch.Tensor): Target sequences (response).
        """
        X, Y = [], []

        for instruction, context, response in text_pairs:
            # Construct input and target sequences
            input_text = f"Instruction: {instruction} Context: {context}"
            target_text = f"{response}"

            # Tokenize input and target sequences
            input_ids = tokenizer.encode(input_text).ids
            target_ids = tokenizer.encode(target_text).ids

            # Truncate or pad sequences to fit block size
            if len(input_ids) > block_size:
                input_ids = input_ids[:block_size]
            else:
                input_ids += [tokenizer.token_to_id("[PAD]")] * (block_size - len(input_ids))

            if len(target_ids) > block_size:
                target_ids = target_ids[:block_size]
            else:
                target_ids += [-100] * (block_size - len(target_ids))

            X.append(torch.tensor(input_ids, dtype=torch.long))
            Y.append(torch.tensor(target_ids, dtype=torch.long))

        # Stack all examples into tensors
        X = torch.stack(X)
        Y = torch.stack(Y)
        return X, Y
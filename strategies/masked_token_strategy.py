from strategies.base_strategy import DataPreparationStrategy
import torch
import random

class MaskedTokenPredictionStrategy(DataPreparationStrategy):
    def prepare(self, text, block_size, tokenizer, mask_probability=0.15, **kwargs):
        encoded = tokenizer.encode(text)
        data = torch.tensor(encoded.ids, dtype=torch.long)
        X, Y = [], []
        for i in range(0, len(data) - block_size, block_size):
            block = data[i:i + block_size]
            input_ids = block.clone()
            labels = block.clone()
            for j in range(len(block)):
                if random.random() < mask_probability:
                    input_ids[j] = tokenizer.token_to_id("<mask>")
                    labels[j] = block[j]
                else:
                    labels[j] = -100
            X.append(input_ids)
            Y.append(labels)
        X = torch.stack(X)
        Y = torch.stack(Y)
        return X, Y
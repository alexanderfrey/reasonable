from strategies.base_strategy import DataPreparationStrategy
import torch

class NextTokenPredictionStrategy(DataPreparationStrategy):
    def prepare(self, text, block_size, tokenizer, **kwargs):
        encoded = tokenizer.encode(text)
        data = torch.tensor(encoded.ids, dtype=torch.long)
        X, Y = [], []
        for i in range(len(data) - block_size):
            X.append(data[i:i + block_size])
            Y.append(data[i + 1:i + block_size + 1])
        X = torch.stack(X)
        Y = torch.stack(Y)
        return X, Y
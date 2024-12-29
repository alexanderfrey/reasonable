class DataPreparationStrategy:
    """
    Interface for all data preparation strategies.
    """
    def prepare(self, text, block_size, tokenizer, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
class StrategyRegistry:
    def __init__(self):
        self.strategies = {}

    def register(self, name, strategy):
        self.strategies[name] = strategy

    def get_strategy(self, name):
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found")
        return self.strategies[name]

# Initialize the registry
registry = StrategyRegistry()

# Import and register all strategies
from strategies.next_token_strategy import NextTokenPredictionStrategy
from strategies.masked_token_strategy import MaskedTokenPredictionStrategy
from strategies.permuted_language_strategy import PermutedLanguageModelingStrategy
from strategies.span_masking_strategy import SpanMaskingStrategy
from strategies.whole_word_masking_strategy import WholeWordMaskingStrategy
from strategies.instruction_following_strategy import InstructionFollowingStrategy



registry.register("next_token", NextTokenPredictionStrategy())
registry.register("masked_token", MaskedTokenPredictionStrategy())
registry.register("permuted_language", PermutedLanguageModelingStrategy())
registry.register("span_masking", SpanMaskingStrategy())
registry.register("whole_word_masking", WholeWordMaskingStrategy())
registry.register("instruction_following", InstructionFollowingStrategy())
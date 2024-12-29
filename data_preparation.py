from strategies.registry import registry

def prepare_data(strategy_name, text, block_size, tokenizer, **kwargs):
    strategy = registry.get_strategy(strategy_name)
    return strategy.prepare(text, block_size, tokenizer, **kwargs)
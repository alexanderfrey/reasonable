import tiktoken
import json
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import Dataset, DataLoader
from strategies import (
    SpanMaskingStrategy,
    InstructionFollowingStrategy,
    NextTokenStrategy,
    MixedStrategy,
)


def create_dataloaders(
    files_pattern,
    block_size,
    batch_size,
    rank,
    world_size,
    num_workers=4,
    prefetch_factor=2,
    pretrain=False,
    main_strategy=None,
    analysis_strategy=None,
    tokenizer=None,  # Add explicit tokenizer parameter
):
    """Create distributed dataloaders with strategy-specific data processing"""

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")  # Default to GPT-2 tokenizer

    # Create appropriate dataset based on strategy types
    if isinstance(main_strategy, NextTokenStrategy):
        dataset = TextDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=tokenizer,
        )
    elif isinstance(main_strategy, SpanMaskingStrategy):
        dataset = MaskingDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=tokenizer,  # Pass tokenizer explicitly
            main_strategy=main_strategy,
            analysis_strategy=analysis_strategy,
        )
    elif isinstance(main_strategy, InstructionFollowingStrategy):
        dataset = InstructionDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=main_strategy.tokenizer,  # Use strategy's tokenizer
            main_strategy=main_strategy,
            analysis_strategy=analysis_strategy,
        )
    else:
        dataset = TextDataset(
            files_pattern=files_pattern, block_size=block_size, tokenizer=tokenizer
        )

    # Strategy-specific collate function
    def collate_fn(examples):
        if isinstance(main_strategy, NextTokenStrategy):
            return collate_default_batch(examples)
        elif isinstance(main_strategy, SpanMaskingStrategy):
            return collate_masking_batch(
                examples, main_strategy, analysis_strategy, tokenizer
            )
        elif isinstance(main_strategy, InstructionFollowingStrategy):
            return collate_instruction_batch(examples, main_strategy, analysis_strategy)
        elif isinstance(main_strategy, MixedStrategy):
            return collate_default_batch(examples)
        else:
            return collate_default_batch(examples)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def collate_masking_batch(examples, main_strategy, analysis_strategy, tokenizer):
    """Collate function for span masking strategies"""
    # Get tokens from examples
    tokens = torch.stack([ex["tokens"] for ex in examples])

    # Apply masking strategies
    main_inputs, main_targets = main_strategy.prepare_masked_input(tokens)
    analysis_inputs, analysis_targets = analysis_strategy.prepare_masked_input(tokens)

    return {
        "main_inputs": main_inputs,
        "main_targets": main_targets,
        "analysis_inputs": analysis_inputs,
        "analysis_targets": analysis_targets,
    }


def collate_instruction_batch(examples, main_strategy, analysis_strategy):
    """Collate function for instruction-following strategies"""
    # Format instructions and responses
    main_texts = [main_strategy.format_prompt(ex["instruction"]) for ex in examples]
    analysis_texts = [
        analysis_strategy.format_prompt(ex.get("thought", "")) for ex in examples
    ]

    # Tokenize
    main_tokens = [main_strategy.tokenizer.encode(text) for text in main_texts]
    analysis_tokens = [
        analysis_strategy.tokenizer.encode(text) for text in analysis_texts
    ]

    # Pad sequences
    main_inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in main_tokens], batch_first=True, padding_value=0
    )
    analysis_inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in analysis_tokens], batch_first=True, padding_value=0
    )

    # Create targets (shifted inputs)
    main_targets = main_inputs.clone()
    main_targets = torch.roll(main_targets, shifts=-1, dims=1)
    main_targets[:, -1] = -100  # Mask last token

    analysis_targets = analysis_inputs.clone()
    analysis_targets = torch.roll(analysis_targets, shifts=-1, dims=1)
    analysis_targets[:, -1] = -100

    return {
        "main_inputs": main_inputs,
        "main_targets": main_targets,
        "analysis_inputs": analysis_inputs,
        "analysis_targets": analysis_targets,
    }


def collate_default_batch(examples):
    """Default collate function for next-token prediction"""
    tokens = [ex["tokens"].clone().detach() for ex in examples]
    padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

    targets = padded.clone()
    targets = torch.roll(targets, shifts=-1, dims=1)
    targets[:, -1] = -100

    return {
        "main_inputs": padded,
        "main_targets": targets,
        "analysis_inputs": padded.clone(),
        "analysis_targets": targets.clone(),
    }


def get_data_info(files_pattern):
    """Get information about the dataset for logging purposes"""
    dataset = BilateralDataset(
        files_pattern, block_size=1024
    )  # block_size doesn't matter here

    # Analyze a sample of examples
    sample_size = min(1000, len(dataset))
    input_lengths = []
    thought_lengths = []

    for idx in range(sample_size):
        example = dataset[idx]
        input_lengths.append(len(example["input"].split()))
        thought_lengths.append(len(example["thought"].split()))

    info = {
        "total_examples": len(dataset),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "avg_thought_length": sum(thought_lengths) / len(thought_lengths),
        "max_input_length": max(input_lengths),
        "max_thought_length": max(thought_lengths),
    }

    return info


class TextDataset(Dataset):
    """Basic dataset for next-token prediction"""

    def __init__(self, files_pattern, block_size, tokenizer):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Load and tokenize texts
        self.tokens = []
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                self.tokens.extend(self.tokenizer.encode(text))

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return {"tokens": x, "targets": y}


class InstructionDataset(Dataset):
    """Dataset for instruction-following strategies"""

    def __init__(
        self, files_pattern, block_size, tokenizer, main_strategy, analysis_strategy
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.main_strategy = main_strategy
        self.analysis_strategy = analysis_strategy

        # Load instruction-response pairs from JSONL files
        self.examples = []
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line.strip())
                    if isinstance(example, dict):
                        self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MaskingDataset(Dataset):
    """Dataset for span masking strategies"""

    def __init__(
        self, files_pattern, block_size, tokenizer, main_strategy, analysis_strategy
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.main_strategy = main_strategy
        self.analysis_strategy = analysis_strategy

        # Load and tokenize texts
        self.examples = []
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                # Tokenize the full text
                tokens = self.tokenizer.encode(text)
                # Create overlapping chunks of appropriate size
                for i in range(0, len(tokens) - block_size + 1, block_size // 2):
                    chunk = tokens[i : i + block_size]
                    if len(chunk) == block_size:  # Only keep full-size chunks
                        self.examples.append(torch.tensor(chunk))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"tokens": self.examples[idx]}


# Dataset class for finetuning with input-thought pairs
class BilateralDataset(Dataset):
    """Dataset for bilateral model training with input text and thought annotations"""

    def __init__(self, files_pattern, block_size, encoding_name="gpt2"):
        super().__init__()
        self.block_size = block_size
        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Load all JSONL files matching the pattern
        self.examples = []
        for filepath in glob(files_pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Parse each line as a separate JSON object
                        example = json.loads(line.strip())
                        if isinstance(example, dict):
                            self.examples.append(example)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {filepath}: {e}")
                        continue

        print(f"Loaded {len(self.examples)} examples from {files_pattern}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {"input": example["input"], "thought": example["thought"]}

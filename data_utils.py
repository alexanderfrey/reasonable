import tiktoken
import json
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import numpy as np
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
    main_strategy=None,
    tokenizer=None,
):
    """Create distributed dataloaders for main path data processing"""

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Select appropriate dataset based on strategy
    dataset = select_dataset(
        files_pattern=files_pattern,
        block_size=block_size,
        tokenizer=tokenizer,
        main_strategy=main_strategy,
    )

    # Select appropriate collate function based on strategy
    collate_fn = select_collate_function(main_strategy)

    # Create train/val splits
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=42,
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=42
    )

    # Create dataloaders with selected collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=lambda x: collate_fn(x),
        drop_last=True,
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=lambda x: collate_fn(x),
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    return train_loader, val_loader, train_sampler, val_sampler


def select_dataset(files_pattern, block_size, tokenizer, main_strategy):
    """Select appropriate dataset based on strategy"""

    if isinstance(main_strategy, NextTokenStrategy):
        return TextDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=tokenizer,
        )
    elif isinstance(main_strategy, SpanMaskingStrategy):
        return MaskingDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=tokenizer,
            main_strategy=main_strategy,
        )
    elif isinstance(main_strategy, InstructionFollowingStrategy):
        return InstructionDataset(
            files_pattern=files_pattern,
            block_size=block_size,
            tokenizer=main_strategy.tokenizer,
            main_strategy=main_strategy,
        )
    else:
        return TextDataset(
            files_pattern=files_pattern, block_size=block_size, tokenizer=tokenizer
        )


def select_collate_function(main_strategy):
    """Select appropriate collate function based on strategy"""

    if isinstance(main_strategy, InstructionFollowingStrategy):
        return collate_instruction_batch
    elif isinstance(main_strategy, SpanMaskingStrategy):
        return collate_masking_batch
    else:
        return collate_default_batch


def collate_default_batch(examples):
    """Default collate function with proper padding and length handling"""

    # Get max length for sequences in this batch
    x_lengths = [len(ex["x"]) for ex in examples]
    y_lengths = [len(ex["y"]) for ex in examples]
    max_len = max(max(x_lengths), max(y_lengths))

    # Pad and stack sequences
    def pad_and_stack(key):
        sequences = [ex[key] for ex in examples]
        padded_seqs = []

        for seq in sequences:
            if len(seq) < max_len:
                # Pad with zeros
                padding = [0] * (max_len - len(seq))
                padded_seq = np.concatenate([seq, padding])
            else:
                padded_seq = seq[:max_len]
            padded_seqs.append(padded_seq)

        return torch.tensor(padded_seqs)

    x = pad_and_stack("x")
    y = pad_and_stack("y")

    return {
        "inputs": x,
        "targets": y,
    }


def collate_masking_batch(examples, main_strategy, tokenizer):
    """Collate function for span masking strategies"""
    # Get tokens from examples
    tokens = torch.stack([ex["tokens"] for ex in examples])

    # Apply masking strategy
    inputs, targets = main_strategy.prepare_masked_input(tokens)

    return {
        "inputs": inputs,
        "targets": targets,
    }


def collate_instruction_batch(examples, main_strategy):
    """Collate function for instruction-following strategies"""
    # Format instructions
    texts = [main_strategy.format_prompt(ex["instruction"]) for ex in examples]

    # Tokenize
    tokens = [main_strategy.tokenizer.encode(text) for text in texts]

    # Pad sequences
    inputs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tokens], batch_first=True, padding_value=0
    )

    # Create targets (shifted inputs)
    targets = inputs.clone()
    targets = torch.roll(targets, shifts=-1, dims=1)
    targets[:, -1] = -100  # Mask last token

    return {
        "inputs": inputs,
        "targets": targets,
    }


def collate_default_batch(examples):
    """Default collate function for next-token prediction"""
    tokens = [ex["tokens"].clone().detach() for ex in examples]
    padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

    targets = padded.clone()
    targets = torch.roll(targets, shifts=-1, dims=1)
    targets[:, -1] = -100

    return {
        "inputs": padded,
        "targets": targets,
    }


def get_data_info(files_pattern):
    """Get information about the dataset for logging purposes"""
    dataset = TextDataset(
        files_pattern, block_size=1024, tokenizer=None
    )  # block_size doesn't matter here

    # Analyze a sample of examples
    sample_size = min(1000, len(dataset))
    input_lengths = []

    for idx in range(sample_size):
        example = dataset[idx]
        input_lengths.append(len(example["tokens"]))

    info = {
        "total_examples": len(dataset),
        "avg_input_length": sum(input_lengths) / len(input_lengths),
        "max_input_length": max(input_lengths),
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

    def __init__(self, files_pattern, block_size, tokenizer, main_strategy):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.main_strategy = main_strategy

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

    def __init__(self, files_pattern, block_size, tokenizer, main_strategy):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.main_strategy = main_strategy

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


class MixedStrategyDataset(Dataset):
    """Dataset that handles different strategies for main and analysis paths"""

    def __init__(
        self, files_pattern, block_size, tokenizer, main_strategy, analysis_strategy
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.main_strategy = main_strategy
        self.analysis_strategy = analysis_strategy

        # Store aligned examples where each item contains both input tokens and instruction data
        self.aligned_examples = []
        print(files_pattern, block_size)
        # Load and process data from files
        for file_path in glob(files_pattern):
            print(file_path)
            if file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line.strip())
                        # Tokenize input text for main path
                        input_text = example["input"]
                        input_tokens = np.array(self.tokenizer.encode(input_text))

                        # Store all necessary data for both paths together
                        self.aligned_examples.append(
                            {
                                "input_tokens": input_tokens,
                                "instruction": example["instruction"],
                                "thought": example["thought"],
                            }
                        )

        if not self.aligned_examples:
            raise ValueError("No valid examples found in the data files")

    def __len__(self):
        return len(self.aligned_examples)

    def __getitem__(self, idx):
        """Get item with sequence length handling and proper tokenization"""
        example = self.aligned_examples[idx]

        # Get main path data using next token prediction on input text
        if isinstance(self.main_strategy, NextTokenStrategy):
            # Randomly select a valid starting point if input is longer than block_size
            input_tokens = example["input_tokens"]
            max_start_idx = len(input_tokens) - (self.block_size + 1)
            if max_start_idx > 0:
                start_idx = random.randint(0, max_start_idx)
            else:
                start_idx = 0

            # Get the chunk for next token prediction
            main_chunk = input_tokens[start_idx : start_idx + self.block_size + 1]
            main_x = main_chunk[:-1]
            main_y = main_chunk[1:]
        else:
            raise NotImplementedError("Only NextTokenStrategy supported for main path")

        # Get analysis path data from the same example
        if isinstance(self.analysis_strategy, InstructionFollowingStrategy):
            # First decode the input tokens to text
            input_text = self.tokenizer.decode(example["input_tokens"])

            # Format the instruction with the decoded input text
            instruction = f"{example['instruction']}\n\nInput: {input_text}"

            # Tokenize and handle sequence lengths
            max_length = self.analysis_strategy.max_length
            instruction_tokens = self.analysis_strategy.tokenizer.encode(instruction)
            thought_tokens = self.analysis_strategy.tokenizer.encode(example["thought"])

            # Truncate if needed, leaving room for special tokens
            max_seq_length = (
                max_length - 3
            )  # Account for [INST], [/INST], and </s] tokens
            if len(instruction_tokens) > max_seq_length:
                instruction_tokens = instruction_tokens[:max_seq_length]
            if len(thought_tokens) > max_seq_length:
                thought_tokens = thought_tokens[:max_seq_length]

            # Decode back to text after truncation
            instruction = self.analysis_strategy.tokenizer.decode(instruction_tokens)
            thought = self.analysis_strategy.tokenizer.decode(thought_tokens)

            return {
                "main_x": main_x,
                "main_y": main_y,
                "analysis_instruction": instruction,
                "analysis_thought": thought,
            }
        else:
            raise NotImplementedError(
                "Only InstructionFollowingStrategy supported for analysis path"
            )

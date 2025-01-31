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
from typing import Dict, List, Optional, Tuple


class MultiHeadDataset(Dataset):
    """Dataset that supports multiple heads with different strategies"""

    def __init__(
        self,
        files_pattern: str,
        block_size: int,
        tokenizer,
        strategies: Dict[str, "ModelingStrategy"],
        is_instruction_data: bool = False,
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.strategies = strategies
        self.is_instruction_data = is_instruction_data

        if is_instruction_data:
            self.load_instruction_data(files_pattern)
        else:
            self.load_text_data(files_pattern)

    def load_text_data(self, files_pattern: str):
        """Load and tokenize plain text data"""
        self.tokens = []
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                self.tokens.extend(self.tokenizer.encode(text))

    def load_instruction_data(self, files_pattern: str):
        """Load instruction-response pairs from JSONL files"""
        self.examples = []
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line.strip())
                    if isinstance(example, dict):
                        self.examples.append(example)

    def __len__(self):
        if self.is_instruction_data:
            return len(self.examples)
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        if self.is_instruction_data:
            return self.get_instruction_item(idx)
        return self.get_text_item(idx)

    def get_text_item(self, idx):
        """Get item for plain text data"""
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])

        # Create data for each head using their respective strategies
        return {"tokens": x, "targets": {name: y for name in self.strategies.keys()}}

    def get_instruction_item(self, idx):
        """Get item for instruction data"""
        example = self.examples[idx]
        # Each strategy might process the instruction differently
        return {
            "instruction": example,
            "targets": {name: example for name in self.strategies.keys()},
        }


def create_dataloaders(
    files_pattern: str,
    block_size: int,
    batch_size: int,
    rank: Optional[int],
    world_size: Optional[int],
    strategies: Dict[str, "ModelingStrategy"],
    tokenizer=None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    is_instruction_data: bool = False,
):
    """Create distributed dataloaders for multi-head processing"""

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset with all strategies
    dataset = MultiHeadDataset(
        files_pattern=files_pattern,
        block_size=block_size,
        tokenizer=tokenizer,
        strategies=strategies,
        is_instruction_data=is_instruction_data,
    )

    # Create train/val splits
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create distributed samplers if needed
    train_sampler = val_sampler = None
    if world_size is not None and world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=42,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42,
        )

    # Create collate function that handles all strategies
    def collate_multi_head(examples: List[Dict]) -> Dict:
        if is_instruction_data:
            return collate_instruction_batch(examples, strategies)
        return collate_text_batch(examples, strategies)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_multi_head,
        drop_last=True,
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_multi_head,
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    return train_loader, val_loader, train_sampler, val_sampler


def collate_text_batch(
    examples: List[Dict], strategies: Dict[str, "ModelingStrategy"]
) -> Dict:
    """Collate function for text data that handles all heads"""
    # Get and pad token sequences
    tokens = [ex["tokens"] for ex in examples]
    padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

    # Create targets dictionary for each head
    targets = {}
    for name, strategy in strategies.items():
        head_targets = torch.nn.utils.rnn.pad_sequence(
            [ex["targets"][name] for ex in examples],
            batch_first=True,
            padding_value=-100,
        )
        targets[name] = (
            strategy.prepare_targets(head_targets)
            if hasattr(strategy, "prepare_targets")
            else head_targets
        )

    return {"inputs": padded, "targets": targets}


def collate_instruction_batch(
    examples: List[Dict], strategies: Dict[str, "ModelingStrategy"]
) -> Dict:
    """Collate function for instruction data that handles all heads"""
    # Process inputs and targets for each head according to its strategy
    head_inputs = {}
    head_targets = {}

    for name, strategy in strategies.items():
        if hasattr(strategy, "prepare_instruction_batch"):
            inputs, targets = strategy.prepare_instruction_batch(
                [ex["instruction"] for ex in examples]
            )
            head_inputs[name] = inputs
            head_targets[name] = targets
        else:
            # Fallback to default processing if strategy doesn't implement custom handling
            texts = [strategy.format_prompt(ex["instruction"]) for ex in examples]
            tokens = [strategy.tokenizer.encode(text) for text in texts]

            inputs = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(t) for t in tokens], batch_first=True, padding_value=0
            )

            targets = torch.roll(inputs.clone(), shifts=-1, dims=1)
            targets[:, -1] = -100

            head_inputs[name] = inputs
            head_targets[name] = targets

    return {"inputs": head_inputs, "targets": head_targets}


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

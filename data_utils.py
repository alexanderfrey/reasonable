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
    def __init__(
        self,
        files_pattern: str,
        block_size: int,
        tokenizer,
        strategies: Dict[str, "ModelingStrategy"],
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.strategies = strategies
        self.examples = []
        self.tokens = []

        self.is_mixed_strategy = (
            len(strategies) == 2
            and isinstance(strategies["primary"], NextTokenStrategy)
            and isinstance(strategies["auxiliary"], InstructionFollowingStrategy)
        )

        if self.is_mixed_strategy:
            self._load_mixed_data(files_pattern)
        elif any(
            isinstance(s, InstructionFollowingStrategy) for s in strategies.values()
        ):
            self._load_instruction_data(files_pattern)
        else:
            self._load_text_data(files_pattern)

    def _load_mixed_data(self, files_pattern: str):
        """Load data for mixed strategy training"""
        print("Using mixed strategy loading")
        for file_path in glob(files_pattern):
            print(file_path)
            if not file_path.endswith(".jsonl"):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line.strip())
                    input_tokens = self.tokenizer.encode(example["input"])
                    self.examples.append(
                        {
                            "input_tokens": input_tokens,
                            "instruction": example["instruction"],
                            "response": example.get("thought", example.get("response")),
                        }
                    )

    def _load_text_data(self, files_pattern: str):
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                self.tokens.extend(self.tokenizer.encode(text))

    def _load_instruction_data(self, files_pattern: str):
        for file_path in glob(files_pattern):
            with open(file_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    example = json.loads(line.strip())
                    if isinstance(example, dict):
                        self.examples.append(example)

    def __len__(self):
        if self.examples:
            return len(self.examples)
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        if self.is_mixed_strategy:
            return self._get_mixed_item(idx)
        elif any(
            isinstance(s, InstructionFollowingStrategy)
            for s in self.strategies.values()
        ):
            return self._get_instruction_item(idx)
        return self._get_text_item(idx)

    def _get_mixed_item(self, idx):
        example = self.examples[idx]
        input_tokens = example["input_tokens"]

        # Next token prediction for primary path
        max_start_idx = max(0, len(input_tokens) - (self.block_size + 1))
        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        chunk = input_tokens[start_idx : start_idx + self.block_size + 1]

        # Pad if necessary
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))

        # Format instruction
        input_text = self.tokenizer.decode(input_tokens)
        instruction = f"{example['instruction']}\n\nInput: {input_text}"

        strategy = self.strategies["auxiliary"]
        max_length = strategy.max_length
        instruction_tokens = strategy.tokenizer.encode(instruction)
        response_tokens = strategy.tokenizer.encode(example["response"])

        # Truncate if needed
        max_seq_length = max_length - 3  # Account for special tokens
        if len(instruction_tokens) > max_seq_length:
            instruction_tokens = instruction_tokens[:max_seq_length]
        if len(response_tokens) > max_seq_length:
            response_tokens = response_tokens[:max_seq_length]

        return {
            "primary": {
                "input": torch.tensor(chunk[:-1], dtype=torch.long),
                "target": torch.tensor(chunk[1:], dtype=torch.long),
            },
            "auxiliary": {
                "instruction": strategy.tokenizer.decode(instruction_tokens),
                "response": strategy.tokenizer.decode(response_tokens),
            },
        }

    def _get_text_item(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return {"tokens": x, "targets": {name: y for name in self.strategies.keys()}}

    def _get_instruction_item(self, idx):
        example = self.examples[idx]
        return {
            "instruction": example,
            "targets": {name: example for name in self.strategies.keys()},
        }


def get_collate_fn(strategies):
    """Determine which collate function to use based on strategy types."""
    has_instruction = any(
        isinstance(s, InstructionFollowingStrategy) for s in strategies.values()
    )
    has_next_token = any(isinstance(s, NextTokenStrategy) for s in strategies.values())

    if has_instruction and has_next_token:
        return lambda x: collate_mixed_batch(x, strategies)
    elif has_instruction:
        return lambda x: collate_instruction_batch(x, strategies)
    else:
        return lambda x: collate_text_batch(x, strategies)


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
):
    """Create distributed dataloaders for multi-head processing"""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = MultiHeadDataset(
        files_pattern=files_pattern,
        block_size=block_size,
        tokenizer=tokenizer,
        strategies=strategies,
    )

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

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

    collate_fn = get_collate_fn(strategies)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    return train_loader, val_loader, train_sampler, val_sampler


def collate_mixed_batch(examples, strategies):
    primary_inputs = []
    primary_targets = []
    aux_inputs = []
    aux_targets = []

    max_length = max(len(ex["primary"]["input"]) for ex in examples)
    aux_strategy = strategies["auxiliary"]

    for ex in examples:
        # Pad primary inputs/targets
        primary_inputs.append(
            torch.nn.functional.pad(
                ex["primary"]["input"],
                (0, max_length - len(ex["primary"]["input"])),
                value=0,
            )
        )
        primary_targets.append(
            torch.nn.functional.pad(
                ex["primary"]["target"],
                (0, max_length - len(ex["primary"]["target"])),
                value=0,
            )
        )

        # Process auxiliary data
        aux_tokens = aux_strategy.format_instruction(
            ex["auxiliary"]["instruction"], ex["auxiliary"]["response"]
        )
        aux_inputs.append(aux_tokens[:-1])
        aux_targets.append(aux_tokens[1:])

    # Stack tensors and return in standard format
    return {
        "inputs": torch.stack(primary_inputs),
        "targets": {
            "primary": torch.stack(primary_targets),
            "auxiliary": torch.stack(aux_targets),
        },
    }


def collate_text_batch(examples, strategies):
    tokens = [ex["tokens"] for ex in examples]
    padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

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

    # Return in standard format
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

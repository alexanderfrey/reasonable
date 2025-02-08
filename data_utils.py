import tiktoken
import json
from datasets import load_dataset
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
from typing import Dict, List, Optional, Tuple, Any, Union


class SingleHeadDataset(Dataset):
    def __init__(
        self,
        files_pattern: Union[str, Dict],
        block_size: int,
        tokenizer,
        strategy: "ModelingStrategy",
        n_aux_heads: int = 0,
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.n_aux_heads = n_aux_heads
        self.max_future_tokens = n_aux_heads + 1 if n_aux_heads > 0 else 1
        self.examples = []
        self.chunks_per_file = {}

        if isinstance(strategy, InstructionFollowingStrategy):
            self._load_instruction_data(files_pattern)
        else:
            self._load_text_data(files_pattern)

        # Verify we have data after loading
        total_chunks = sum(len(chunks) for chunks in self.chunks_per_file.values())
        print(
            f"Loaded total of {total_chunks} chunks across {len(self.chunks_per_file)} files"
        )
        if total_chunks == 0:
            raise ValueError("No data was loaded into the dataset!")

    def _load_instruction_data(self, dataset_source: Union[str, Dict]):
        """Load instruction data from either HuggingFace dataset or local files"""
        print("Loading instruction data...")

        if isinstance(dataset_source, str):
            if dataset_source.startswith(("hf://", "huggingface://")):
                # Extract dataset path from the URL-like string
                path = dataset_source.split("://")[1]
                self._load_huggingface_dataset(path=path)
            else:
                self._load_local_instruction_files(dataset_source)
        elif isinstance(dataset_source, dict):
            # Ensure we're using consistent parameter names
            if "dataset_source" in dataset_source:
                dataset_source["path"] = dataset_source.pop("dataset_source")
            self._load_huggingface_dataset(**dataset_source)
        else:
            raise ValueError("Invalid dataset source format")

    def _load_huggingface_dataset(
        self, path: str, split: str = "train", revision: str = "main"
    ):
        print(f"\nAttempting to load HuggingFace dataset:")
        print(f"Path: {path}")
        print(f"Split: {split}")
        print(f"Block size: {self.block_size}")
        print(f"Max future tokens: {self.max_future_tokens}")

        try:
            from datasets import load_dataset

            dataset = load_dataset(path=path, split=split, revision=revision)

            print(f"\nDataset loaded successfully:")
            print(f"Number of examples: {len(dataset)}")

            successful_chunks = 0
            total_tokens = 0
            skipped_long = 0
            skipped_short = 0

            for idx, item in enumerate(dataset):
                try:
                    # Debug the item contents
                    # if idx < 5:
                    #     print(f"\n=== Processing Example {idx} ===")
                    #     print(
                    #         "Raw item:",
                    #         {
                    #             k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                    #             for k, v in item.items()
                    #         },
                    #     )

                    formatted_text = self._extract_text_from_item(item)

                    if not formatted_text.strip():
                        continue

                    tokens = self.tokenizer.encode(formatted_text)
                    # if idx < 5:
                    #     print(f"Token length: {len(tokens)}")
                    #     print(f"First few tokens: {tokens[:10]}")
                    #     print(
                    #         f"Decoded first few tokens: {self.tokenizer.decode(tokens[:10])}"
                    #     )
                    #     print(f"Block size: {self.block_size}")
                    #     print(
                    #         f"Chunk size (block_size + max_future_tokens): {self.block_size + self.max_future_tokens}"
                    #     )

                    total_tokens += len(tokens)

                    if not tokens:
                        continue

                    # For instruction completion, we process each example as a single chunk
                    chunk_size = self.block_size + self.max_future_tokens

                    chunks = []
                    # Handle the sequence based on its length
                    # For instruction completion, we want to accept any reasonable length sequence
                    # while still respecting the maximum block size
                    chunk_size = self.block_size + self.max_future_tokens
                    min_sequence_length = (
                        32  # Minimum reasonable length for an instruction-response pair
                    )

                    chunks = []
                    # Handle the sequence based on its length
                    if len(tokens) < min_sequence_length:
                        skipped_short += 1

                    elif len(tokens) <= chunk_size:
                        # Pad sequence to chunk_size
                        padded_chunk = tokens + [self.strategy.get_padding_value()] * (
                            chunk_size - len(tokens)
                        )
                        chunks.append(padded_chunk)

                    else:
                        # For long sequences, take the first chunk_size tokens
                        chunk = tokens[:chunk_size]
                        chunks.append(chunk)
                        skipped_long += 1

                    if chunks:
                        self.chunks_per_file[f"hf_example_{idx}"] = chunks
                        successful_chunks += len(chunks)

                    if idx % 1000 == 0:
                        print(
                            f"Processed {idx} examples, created {successful_chunks} chunks"
                        )

                except Exception as e:
                    print(f"Error processing example {idx}: {str(e)}")
                    continue

            # Final statistics
            print(f"\nDataset Processing Statistics:")
            print(f"Total examples processed: {len(dataset)}")
            print(f"Total tokens across all examples: {total_tokens}")
            print(f"Average tokens per example: {total_tokens/len(dataset):.2f}")
            print(f"Successful chunks created: {successful_chunks}")
            print(f"Examples skipped (too short): {skipped_short}")
            print(f"Examples skipped (too long): {skipped_long}")
            print(
                f"Average chunks per successful example: {successful_chunks/(len(dataset)-skipped_short-skipped_long):.2f}"
            )

        except Exception as e:
            print(f"Error loading HuggingFace dataset: {str(e)}")
            raise

    def _extract_text_from_item(self, item):
        """Helper to extract text from different dataset formats"""
        if isinstance(item, dict):
            # Handle TLDR dataset format
            if "prompt" in item and "completion" in item:
                return self.strategy.format_instruction(
                    instruction=item["prompt"], response=item["completion"]
                )
            # Handle default format fields
            elif "instruction" in item and "output" in item:
                return self.strategy.format_instruction(
                    instruction=item["instruction"],
                    input_context=item["input"],
                    response=item["output"],
                )
            elif "text" in item:
                return item["text"]
            else:
                print(
                    f"Warning: Unknown data format. Available fields: {list(item.keys())}"
                )
                return " ".join(str(v) for v in item.values())
        return str(item)

    def _load_local_instruction_files(self, files_pattern: str):
        """Load instruction data from local files"""
        print("Loading instruction data from local files")
        for file_path in glob(files_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        tokens = self.tokenizer.encode(text)
                        if tokens:
                            chunks = []
                            chunk_size = self.block_size + self.max_future_tokens
                            stride = self.block_size // 24

                            for start in range(0, len(tokens) - chunk_size + 1, stride):
                                chunk = tokens[start : start + chunk_size]
                                if len(chunk) == chunk_size:
                                    chunks.append(chunk)

                            if chunks:
                                self.chunks_per_file[file_path] = chunks
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

    def _load_text_data(self, files_pattern: str):
        """Load data and split into chunks while maintaining sequence continuity"""
        print("Loading text data for next-token prediction")
        for file_path in glob(files_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        # Tokenize the entire text
                        tokens = self.tokenizer.encode(text)
                        if tokens:
                            # Split into overlapping chunks
                            chunks = []
                            chunk_size = self.block_size + self.max_future_tokens
                            stride = (
                                self.block_size // 24
                            )  # Use overlap to maintain context

                            for start in range(0, len(tokens) - chunk_size + 1, stride):
                                chunk = tokens[start : start + chunk_size]
                                if (
                                    len(chunk) == chunk_size
                                ):  # Only keep complete chunks
                                    chunks.append(chunk)

                            if chunks:
                                self.chunks_per_file[file_path] = chunks
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

    def _get_train_val_splits(self):
        """Split chunks into train/val while maintaining sequence integrity"""
        train_chunks = []
        val_chunks = []

        for file_path, chunks in self.chunks_per_file.items():
            file_hash = hash(file_path) % 10000
            generator = torch.Generator().manual_seed(file_hash)

            # Shuffle chunk indices instead of individual tokens
            n_chunks = len(chunks)
            train_size = int(0.9 * n_chunks)

            indices = torch.randperm(n_chunks, generator=generator)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Add chunks to respective splits
            train_chunks.extend([chunks[i] for i in train_indices.tolist()])
            val_chunks.extend([chunks[i] for i in val_indices.tolist()])

        return train_chunks, val_chunks

    def create_train_val_datasets(self):
        """Create separate training and validation datasets"""
        # Collect all chunks
        all_chunks = []
        for chunks in self.chunks_per_file.values():
            all_chunks.extend(chunks)

        # Create train/val split
        total_size = len(all_chunks)
        train_size = int(0.9 * total_size)

        # Use consistent random seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_size, generator=generator)

        train_chunks = [all_chunks[i] for i in indices[:train_size]]
        val_chunks = [all_chunks[i] for i in indices[train_size:]]

        print(
            f"Created train split with {len(train_chunks)} chunks and val split with {len(val_chunks)} chunks"
        )

        return (
            SingleHeadTrainDataset(
                train_chunks,
                self.block_size,
                strategy=self.strategy,
                n_aux_heads=self.n_aux_heads,
            ),
            SingleHeadValDataset(
                val_chunks,
                self.block_size,
                strategy=self.strategy,
                n_aux_heads=self.n_aux_heads,
            ),
        )


class SingleHeadTrainDataset(Dataset):
    def __init__(self, data, block_size, strategy=None, n_aux_heads=0):
        self.data = data  # Now contains complete chunks
        self.block_size = block_size
        self.strategy = strategy
        self.n_aux_heads = n_aux_heads
        self.max_future_tokens = n_aux_heads + 1 if n_aux_heads > 0 else 1

    def __len__(self):
        return len(self.data)  # Length is now number of chunks

    def __getitem__(self, idx):
        chunk = self.data[idx]

        # Input sequence
        x = torch.tensor(chunk[: self.block_size], dtype=torch.long)

        # Create targets for main and auxiliary heads
        targets = []
        for i in range(self.max_future_tokens):
            target = torch.tensor(
                chunk[i + 1 : i + 1 + self.block_size], dtype=torch.long
            )
            targets.append(target)

        return {
            "inputs": x,
            "targets": targets[0],
            "future_targets": targets[1:] if self.n_aux_heads > 0 else [],
        }

    def _get_text_item(self, idx):
        # Get the pre-made chunk
        chunk = self.data[idx]

        # Input sequence
        x = torch.tensor(chunk[: self.block_size], dtype=torch.long)

        # Create targets for main and auxiliary heads
        targets = []
        for i in range(self.max_future_tokens):
            target = torch.tensor(
                chunk[i + 1 : i + 1 + self.block_size], dtype=torch.long
            )
            targets.append(target)

        return {
            "inputs": x,
            "targets": targets[0],
            "future_targets": targets[1:] if self.n_aux_heads > 0 else [],
        }


class SingleHeadValDataset(SingleHeadTrainDataset):
    pass


def create_dataloaders(
    files_pattern: Union[str, Dict],
    block_size: int,
    batch_size: int,
    rank: Optional[int],
    world_size: Optional[int],
    strategy: "ModelingStrategy",
    tokenizer=None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    n_aux_heads: int = 0,
) -> Tuple[
    DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]
]:
    """Create distributed dataloaders with support for auxiliary heads and HuggingFace datasets

    Args:
        files_pattern: Either a string path to local files or a dict/string specifying HuggingFace dataset
            - For local files: "path/to/files/*.txt"
            - For HuggingFace: "hf://dataset_name" or {"path": "dataset_name", "split": "train"}
        block_size: Maximum length of input sequences
        batch_size: Number of samples per batch
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
        strategy: Training strategy (e.g., InstructionFollowingStrategy)
        tokenizer: Tokenizer instance (defaults to GPT-2 tiktoken)
        num_workers: Number of data loading worker processes
        prefetch_factor: Number of batches to prefetch per worker
        n_aux_heads: Number of auxiliary prediction heads

    Returns:
        Tuple containing:
        - Training dataloader
        - Validation dataloader
        - Training sampler (if distributed)
        - Validation sampler (if distributed)
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Handle HuggingFace dataset specification
    if isinstance(files_pattern, str) and files_pattern.startswith(
        ("hf://", "huggingface://")
    ):
        files_pattern = {"path": files_pattern.split("://")[1]}

    dataset = SingleHeadDataset(
        files_pattern=files_pattern,
        block_size=block_size,
        tokenizer=tokenizer,
        strategy=strategy,
        n_aux_heads=n_aux_heads,
    )

    train_dataset, val_dataset = dataset.create_train_val_datasets()

    # Set up distributed sampling if needed
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

    def collate_fn(batch):
        """Collate function supporting auxiliary heads"""
        if not batch:
            return {"inputs": [], "targets": [], "future_targets": []}

        # Stack inputs and main targets
        inputs = torch.stack([item["inputs"] for item in batch])
        targets = torch.stack([item["targets"] for item in batch])

        # Handle future targets for auxiliary heads
        future_targets = []
        if batch[0]["future_targets"]:  # Check if we have auxiliary heads
            for i in range(len(batch[0]["future_targets"])):
                future_target = torch.stack(
                    [item["future_targets"][i] for item in batch]
                )
                future_targets.append(future_target)

        return {"inputs": inputs, "targets": targets, "future_targets": future_targets}

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


class CurriculumDataset(Dataset):
    def __init__(
        self,
        file_patterns: List[str],
        block_size: int,
        tokenizer,
        active_files: Optional[List[str]] = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.file_patterns = file_patterns
        self.active_files = active_files or [file_patterns[0]]
        self.all_files = self._get_all_files()
        self.tokens_per_file = {}  # Store tokens for each file separately
        self._load_active_files()

    def _get_all_files(self) -> Dict[str, List[str]]:
        """Get all files for each pattern for future reference"""
        all_files = {}
        for pattern in self.file_patterns:
            all_files[pattern] = sorted(glob(pattern))
        return all_files

    def __len__(self):
        """Return total length of available tokens"""
        total_tokens = sum(len(tokens) for tokens in self.tokens_per_file.values())
        if total_tokens <= self.block_size:
            return 0
        return total_tokens - self.block_size

    def _load_active_files(self):
        """Load data from currently active files"""
        self.tokens_per_file.clear()

        for pattern in self.active_files:
            for file_path in self.all_files.get(pattern, []):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():  # Check if text is not empty
                            file_tokens = self.tokenizer.encode(text)
                            if file_tokens:  # Only add if we got tokens
                                self.tokens_per_file[file_path] = file_tokens
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    continue

    def _get_train_val_splits(self):
        """Create deterministic train/val splits for each file"""
        train_tokens = []
        val_tokens = []

        for file_path, tokens in self.tokens_per_file.items():
            # Use hash of filename for deterministic split
            file_hash = hash(file_path) % 10000  # Limit hash size
            generator = torch.Generator().manual_seed(file_hash)

            # Calculate splits for this file
            n_tokens = len(tokens)
            train_size = int(0.8 * n_tokens)

            # Generate permutation for this file
            indices = torch.randperm(n_tokens, generator=generator)

            # Split tokens for this file
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Add to appropriate sets
            train_tokens.extend([tokens[i] for i in train_indices.tolist()])
            val_tokens.extend([tokens[i] for i in val_indices.tolist()])

        return train_tokens, val_tokens

    def create_train_val_datasets(self):
        """Create separate training and validation datasets"""
        train_tokens, val_tokens = self._get_train_val_splits()
        return (
            SingleHeadTrainDataset(train_tokens, self.block_size),
            SingleHeadValDataset(val_tokens, self.block_size),
        )

    def rebuild_dataloaders(
        self,
        batch_size: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        """Rebuild train and val dataloaders after adding new files"""
        # Get train and val datasets with consistent splits
        train_dataset, val_dataset = self.create_train_val_datasets()

        # Create new samplers if using distributed training
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

        # Create new dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            collate_fn=self.collate_fn,
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
            collate_fn=self.collate_fn,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
        )

        return train_loader, val_loader, train_sampler, val_sampler

    def add_next_file_pattern(
        self,
    ) -> bool:
        """Add next file pattern to curriculum. Returns success boolean"""
        current_idx = self.file_patterns.index(self.active_files[-1])
        if current_idx + 1 < len(self.file_patterns):
            next_pattern = self.file_patterns[current_idx + 1]
            self.active_files.append(next_pattern)
            self._load_active_files()
            return True
        return False

    @staticmethod
    def collate_fn(batch):
        if not batch:
            return {"inputs": [], "targets": []}
        inputs = torch.stack([item["inputs"] for item in batch])
        targets = torch.stack([item["targets"] for item in batch])
        return {"inputs": inputs, "targets": targets}

    def __getitem__(self, idx):
        """Get item from the dataset"""
        all_tokens = []
        for tokens in self.tokens_per_file.values():
            all_tokens.extend(tokens)

        chunk = all_tokens[idx : idx + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {"inputs": x, "targets": y}


def create_curriculum_dataloaders(
    file_patterns: List[str],
    block_size: int,
    batch_size: int,
    rank: Optional[int],
    world_size: Optional[int],
    tokenizer=None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    active_files: Optional[List[str]] = None,
):
    """Create dataloaders with curriculum learning support"""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = CurriculumDataset(
        file_patterns=file_patterns,
        block_size=block_size,
        tokenizer=tokenizer,
        active_files=active_files,
    )

    # Create train/val datasets
    train_dataset, val_dataset = dataset.create_train_val_datasets()

    # Create samplers for distributed training
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
        collate_fn=CurriculumDataset.collate_fn,
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
        collate_fn=CurriculumDataset.collate_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
    )

    return train_loader, val_loader, train_sampler, val_sampler, dataset


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

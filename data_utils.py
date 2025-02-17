import tiktoken
import json
from datasets import load_dataset
from glob import glob
from collections import defaultdict
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
from text_normalizer import TextNormalizer


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
                    # input_context=item["input"],
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

    def _load_local_instruction_files(
        self, files_pattern: str, include_thoughts: bool = False
    ):
        """Load instruction data from local JSONL files.

        Args:
            files_pattern: Glob pattern for JSONL files containing instruction data
        """

        print(f"\nLoading local instruction files matching pattern: {files_pattern}")

        files = glob(files_pattern)
        if not files:
            raise ValueError(f"No files found matching pattern: {files_pattern}")

        print(f"Found {len(files)} files to process")

        successful_chunks = 0
        total_tokens = 0
        skipped_long = 0
        skipped_short = 0
        total_examples = 0

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            item = json.loads(line)
                            total_examples += 1

                            # Extract prompt and response from the JSONL format
                            if "prompt" not in item or "correct_response" not in item:
                                print(
                                    f"Warning: Missing required fields in line {line_num} of {file_path}"
                                )
                                continue

                            # Format the instruction using the strategy
                            formatted_text = self.strategy.format_instruction(
                                instruction=item["prompt"],
                                response=item["correct_response"],
                                input_context=item.get("context"),
                                thinking=(
                                    "\n".join(item["thoughts"])
                                    if include_thoughts
                                    else None
                                ),
                            )

                            if not formatted_text.strip():
                                continue

                            # print(formatted_text)
                            tokens = self.tokenizer.encode(formatted_text)
                            total_tokens += len(tokens)

                            if not tokens:
                                continue

                            # Process each example as a single chunk
                            chunk_size = self.block_size + self.max_future_tokens
                            min_sequence_length = 32  # Minimum reasonable length

                            chunks = []
                            if len(tokens) < min_sequence_length:
                                skipped_short += 1

                            elif len(tokens) <= chunk_size:
                                # Pad sequence to chunk_size
                                padded_chunk = tokens + [
                                    self.strategy.get_padding_value()
                                ] * (chunk_size - len(tokens))
                                chunks.append(padded_chunk)

                            else:
                                # For long sequences, take the first chunk_size tokens
                                chunk = tokens[:chunk_size]
                                chunks.append(chunk)
                                skipped_long += 1

                            if chunks:
                                self.chunks_per_file[f"{file_path}_line_{line_num}"] = (
                                    chunks
                                )
                                successful_chunks += len(chunks)

                            if total_examples % 1000 == 0:
                                print(
                                    f"Processed {total_examples} examples, created {successful_chunks} chunks"
                                )

                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON at line {line_num} in {file_path}: {str(e)}"
                            )
                            continue
                        except Exception as e:
                            print(
                                f"Error processing line {line_num} in {file_path}: {str(e)}"
                            )
                            continue

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue

        # Final statistics
        print(f"\nLocal Files Processing Statistics:")
        print(f"Total files processed: {len(files)}")
        print(f"Total examples processed: {total_examples}")
        print(f"Total tokens across all examples: {total_tokens}")
        print(
            f"Average tokens per example: {total_tokens/total_examples:.2f}"
            if total_examples > 0
            else "No examples processed"
        )
        print(f"Successful chunks created: {successful_chunks}")
        print(f"Examples skipped (too short): {skipped_short}")
        print(f"Examples skipped (too long): {skipped_long}")
        if total_examples > skipped_short + skipped_long:
            print(
                f"Average chunks per successful example: {successful_chunks/(total_examples-skipped_short-skipped_long):.2f}"
            )

    def _load_text_data(self, files_pattern: str):
        """Load and clean data, then split into chunks with streaming approach"""
        print("Loading and cleaning text data for next-token prediction")

        # Initialize text normalizer
        normalizer = TextNormalizer()
        self.chunks_per_file = defaultdict(list)
        chunk_size = self.block_size + self.max_future_tokens
        stride = self.block_size // 2

        def process_file_chunks(tokens, file_path):
            chunks = []
            for start in range(0, len(tokens) - chunk_size + 1, stride):
                chunk = tokens[start : start + chunk_size]
                if len(chunk) == chunk_size:
                    chunks.append(chunk)
            return chunks

        # Rest of the function remains the same, but add cleaning step
        for file_path in glob(files_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    buffer_size = 1024 * 1024  # 1MB buffer
                    tokens = []
                    text_buffer = ""

                    while True:
                        text = f.read(buffer_size)
                        if not text:
                            # Clean and process final buffer
                            if text_buffer:
                                cleaned_text = normalizer.clean_text(text_buffer)
                                new_tokens = self.tokenizer.encode(cleaned_text)
                                tokens.extend(new_tokens)
                            break

                        text_buffer += text

                        # Only process complete lines/paragraphs
                        if len(text_buffer) >= buffer_size:
                            # Find last newline to avoid splitting mid-paragraph
                            split_pos = text_buffer.rfind("\n")
                            if split_pos == -1:
                                split_pos = len(text_buffer)

                            # Clean and process the complete portion
                            to_process = text_buffer[:split_pos]
                            cleaned_text = normalizer.clean_text(to_process)
                            new_tokens = self.tokenizer.encode(cleaned_text)
                            tokens.extend(new_tokens)

                            # Keep remainder for next iteration
                            text_buffer = text_buffer[split_pos:]

                        # Process chunks when we have enough tokens
                        if len(tokens) >= chunk_size * 100:
                            new_chunks = process_file_chunks(tokens, file_path)
                            self.chunks_per_file[file_path].extend(new_chunks)
                            tokens = tokens[-chunk_size:]

                    # Process any remaining tokens
                    if tokens:
                        new_chunks = process_file_chunks(tokens, file_path)
                        self.chunks_per_file[file_path].extend(new_chunks)

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

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
                self.tokenizer,
                strategy=self.strategy,
                n_aux_heads=self.n_aux_heads,
            ),
            SingleHeadValDataset(
                val_chunks,
                self.block_size,
                self.tokenizer,
                strategy=self.strategy,
                n_aux_heads=self.n_aux_heads,
            ),
        )


class SingleHeadTrainDataset(Dataset):
    def __init__(self, data, block_size, tokenizer, strategy=None, n_aux_heads=0):
        self.data = data
        self.block_size = block_size
        self.strategy = strategy
        self.n_aux_heads = n_aux_heads
        self.max_future_tokens = n_aux_heads + 1 if n_aux_heads > 0 else 1
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]

        # Validate chunk size
        required_length = self.block_size + self.max_future_tokens
        if len(chunk) < required_length:
            raise ValueError(
                f"Chunk {idx} is too small: {len(chunk)} < {required_length}. "
                f"Need {self.block_size} tokens for input + {self.max_future_tokens} tokens for predictions"
            )

        # Input sequence (tokens 0 to block_size-1)
        inputs = chunk[: self.block_size]
        x = torch.tensor(inputs, dtype=torch.long)

        # Targets are the shifted sequence (tokens 1 to block_size)
        targets = chunk[1 : self.block_size + 1]
        targets = torch.tensor(targets, dtype=torch.long)

        # print(f"Last input token: {self.tokenizer.decode([inputs[-1]])}")
        # print(f"First target token: {self.tokenizer.decode([targets[0]])}")

        # Future targets for auxiliary heads
        future_targets = []
        for i in range(1, self.n_aux_heads + 1):
            start_idx = i + 1
            end_idx = self.block_size + i + 1
            if end_idx <= len(chunk):
                future_target = chunk[start_idx:end_idx]
                future_target = torch.tensor(future_target, dtype=torch.long)
                future_targets.append(future_target)
            else:
                # If we don't have enough tokens, pad with -100 (ignore_index for cross_entropy)
                pad_length = end_idx - len(chunk)
                valid_tokens = chunk[start_idx:] if start_idx < len(chunk) else []
                padding = [-100] * pad_length
                future_target = torch.tensor(valid_tokens + padding, dtype=torch.long)
                future_targets.append(future_target)

        return {"inputs": x, "targets": targets, "future_targets": future_targets}


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

import tiktoken
import json, os
from datasets import load_dataset
from glob import glob
from collections import defaultdict
import hashlib
from torch.utils.data.distributed import DistributedSampler
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from strategies import (
    InstructionFollowingStrategy,
    NextTokenStrategy,
)

from pathlib import Path
import psycopg2
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from text_normalizer import TextNormalizer
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()


def create_connection_pool():
    """Create a database connection pool using environment variables."""
    return ThreadedConnectionPool(
        1,  # minconn
        40,  # maxconn
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        port=os.getenv("DB_PORT"),
        password=os.getenv("DB_PASSWORD"),
    )


class SingleHeadDataset(Dataset):
    def __init__(
        self,
        files_pattern: Union[str, Dict],
        block_size: int,
        tokenizer,
        strategy: "ModelingStrategy",
        n_aux_heads: int = 0,
        use_news: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.n_aux_heads = n_aux_heads
        self.max_future_tokens = n_aux_heads + 1 if n_aux_heads > 0 else 1
        self.examples = []
        self.chunks_per_file = {}

        if use_news:
            conn_pool = create_connection_pool()
            self._load_news_data(conn_pool=conn_pool)
        elif isinstance(strategy, InstructionFollowingStrategy):
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

    def _load_local_instruction_files(
        self,
        files_pattern: str,
        include_thoughts: bool = False,
        train_ratio: float = 0.8,
    ):
        """Load instruction data from local JSONL files with a consistent train/val split.

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

        self.chunks_per_file = defaultdict(list)
        self.train_chunks = []
        self.val_chunks = []

        def hash_to_bucket(filename, train_ratio=0.8):
            """Deterministically assign a file to train or validation based on its hashed filename."""
            hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
            return (hash_val % 100) < (train_ratio * 100)

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
                                chunk_key = f"{file_path}_line_{line_num}"
                                self.chunks_per_file[chunk_key] = chunks
                                successful_chunks += len(chunks)

                                # Determine whether this chunk goes to train or validation
                                if hash_to_bucket(
                                    os.path.basename(file_path), train_ratio
                                ):
                                    self.train_chunks.extend(chunks)
                                else:
                                    self.val_chunks.extend(chunks)

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
        print(
            f"Train chunks: {len(self.train_chunks)}, Validation chunks: {len(self.val_chunks)}"
        )
        if total_examples > skipped_short + skipped_long:
            print(
                f"Average chunks per successful example: {successful_chunks/(total_examples-skipped_short-skipped_long):.2f}"
            )

    def _load_text_data(self, files_pattern: str, train_ratio: float = 0.8):
        """Load and clean data, then split into chunks with streaming approach using a consistent train/val split."""
        print("Loading and cleaning text data for next-token prediction")

        # Initialize text normalizer
        normalizer = TextNormalizer()
        self.chunks_per_file = defaultdict(list)
        self.train_chunks = []
        self.val_chunks = []

        chunk_size = self.block_size + self.max_future_tokens
        stride = self.block_size // 2

        def process_file_chunks(tokens):
            chunks = []
            for start in range(0, len(tokens) - chunk_size + 1, stride):
                chunk = tokens[start : start + chunk_size]
                if len(chunk) == chunk_size:
                    chunks.append(chunk)
            return chunks

        def hash_to_bucket(filename, train_ratio=0.8):
            """Deterministically assign a file to train or validation based on its hashed filename."""
            hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
            return (hash_val % 100) < (train_ratio * 100)

        # Iterate through files
        for file_path in glob(files_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    buffer_size = 1024 * 1024  # 1MB buffer
                    tokens = []
                    text_buffer = ""

                    while True:
                        text = f.read(buffer_size)
                        if not text:
                            # Process the last remaining text
                            if text_buffer:
                                cleaned_text = normalizer.clean_text(text_buffer)
                                new_tokens = self.tokenizer.encode(cleaned_text)
                                tokens.extend(new_tokens)
                            break

                        text_buffer += text

                        # Process complete paragraphs
                        if len(text_buffer) >= buffer_size:
                            split_pos = text_buffer.rfind("\n")
                            if split_pos == -1:
                                split_pos = len(text_buffer)

                            to_process = text_buffer[:split_pos]
                            cleaned_text = normalizer.clean_text(to_process)
                            new_tokens = self.tokenizer.encode(cleaned_text)
                            tokens.extend(new_tokens)

                            # Keep remainder for next iteration
                            text_buffer = text_buffer[split_pos:]

                        # Process chunks if we have enough tokens
                        if len(tokens) >= chunk_size * 100:
                            new_chunks = process_file_chunks(tokens)
                            self.chunks_per_file[file_path].extend(new_chunks)
                            tokens = tokens[-chunk_size:]

                    # Process any remaining tokens
                    if tokens:
                        new_chunks = process_file_chunks(tokens)
                        self.chunks_per_file[file_path].extend(new_chunks)

                    # Assign file to train or validation set
                    if hash_to_bucket(os.path.basename(file_path), train_ratio):
                        self.train_chunks.extend(self.chunks_per_file[file_path])
                    else:
                        self.val_chunks.extend(self.chunks_per_file[file_path])

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

        print(
            f"Train chunks: {len(self.train_chunks)}, Validation chunks: {len(self.val_chunks)}"
        )

    def _load_news_data(self, conn_pool):
        """Load alternative bodies from database, clean data, then split into chunks with streaming approach"""
        print("Loading and cleaning alternative bodies for next-token prediction")

        # Initialize text normalizer and storage
        normalizer = TextNormalizer()
        self.chunks_per_file = defaultdict(list)
        chunk_size = self.block_size + self.max_future_tokens
        stride = self.block_size // 2

        def process_text_chunks(tokens, news_id):
            chunks = []
            for start in range(0, len(tokens) - chunk_size + 1, stride):
                chunk = tokens[start : start + chunk_size]
                if len(chunk) == chunk_size:
                    chunks.append(chunk)
            return chunks

        # Load data from database
        try:
            print("Loading articles from database...")
            df = load_alternative_bodies(conn_pool)
            tqdm.pandas(desc="Normalizing texts")
            df["alternative_body"] = df["alternative_body"].progress_apply(
                lambda x: normalizer.clean_text(x)
            )

            write_alternative_bodies_to_files(df)
            print(f"Found {len(df)} articles")

            # Process each article's alternative body
            for _, row in tqdm(
                df.iterrows(), total=len(df), desc="Processing articles"
            ):
                try:
                    news_id = str(row["news_uuid"])
                    text = row["alternative_body"]

                    if not text:  # Skip if text is empty
                        continue

                    tokens = []
                    text_buffer = ""

                    # Process text in chunks to handle large articles
                    for i in tqdm(
                        range(0, len(text), 1024 * 1024),
                        total=(len(text) // (1024 * 1024)) + 1,
                        desc=f"Processing article {news_id[:8]}",
                        leave=False,
                    ):  # 1MB chunks
                        text_chunk = text[i : i + 1024 * 1024]
                        text_buffer += text_chunk

                        # Process complete portions
                        if len(text_buffer) >= 1024 * 1024:
                            # Find last sentence boundary to avoid splitting mid-sentence
                            split_pos = text_buffer.rfind(". ")
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
                            new_chunks = process_text_chunks(tokens, news_id)
                            self.chunks_per_file[news_id].extend(new_chunks)
                            tokens = tokens[-chunk_size:]

                    # Process any remaining text in buffer
                    if text_buffer:
                        cleaned_text = normalizer.clean_text(text_buffer)
                        new_tokens = self.tokenizer.encode(cleaned_text)
                        tokens.extend(new_tokens)

                    # Process any remaining tokens
                    if tokens:
                        new_chunks = process_text_chunks(tokens, news_id)
                        self.chunks_per_file[news_id].extend(new_chunks)

                except Exception as e:
                    print(f"Error processing article {news_id}: {e}")
                    continue

            total_chunks = sum(len(chunks) for chunks in self.chunks_per_file.values())
            print(
                f"Loaded {len(self.chunks_per_file)} articles with {total_chunks} total chunks"
            )

        except Exception as e:
            print(f"Error loading data from database: {e}")
            raise

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
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def create_padding_mask(self, sequence_length):
        mask = torch.ones(self.block_size)  # Start with all 1s
        if sequence_length < self.block_size:
            mask[sequence_length:] = 0  # Set padding positions to 0
        return mask

    def create_targets(self, chunk, original_length):
        """
        Create targets with instruction and padding tokens masked
        """

        # Create main target sequence (shifted by 1)
        targets = chunk[1 : self.block_size + 1]
        main_targets = torch.tensor(targets, dtype=torch.long)

        # Find response marker
        response_marker = self.tokenizer.encode("\n### Response:\n")
        response_start = None
        for i in range(len(chunk) - len(response_marker) + 1):
            if chunk[i : i + len(response_marker)] == response_marker:
                response_start = i
                break

        # Create mask for all positions that should be ignored
        mask = torch.zeros_like(main_targets, dtype=torch.bool)

        # 1. Mask instruction part
        if response_start is not None:
            cutoff = response_start + len(response_marker)
            mask_idx = min(cutoff - 1, len(main_targets))
            mask[:mask_idx] = True

        # 2. Mask padding tokens
        mask = mask | (main_targets == 50256)  # GPT-2 padding token

        # 3. Mask positions beyond original length
        if original_length < self.block_size:
            mask[original_length - 1 :] = True

        # Apply the mask
        main_targets[mask] = self.ignore_index

        # Create future targets
        future_targets = []
        if self.n_aux_heads > 0:
            for i in range(2, self.n_aux_heads + 2):
                future_target = chunk[i : self.block_size + i]
                future_target_tensor = torch.tensor(future_target, dtype=torch.long)

                # Create similar mask for future targets
                future_mask = torch.zeros_like(future_target_tensor, dtype=torch.bool)

                if response_start is not None:
                    cutoff = response_start + len(response_marker)
                    mask_idx = min(cutoff - i, len(future_target_tensor))
                    future_mask[:mask_idx] = True

                future_mask = future_mask | (future_target_tensor == 50256)

                if original_length < self.block_size + i - 1:
                    future_mask[original_length - i :] = True

                future_target_tensor[future_mask] = self.ignore_index
                future_targets.append(future_target_tensor)

        return main_targets, future_targets

    def __getitem__(self, idx):
        chunk = self.data[idx]
        original_length = len(chunk)
        required_length = self.block_size + self.max_future_tokens

        if len(chunk) < required_length:
            chunk = self.pad_sequence(chunk, required_length)

        # Input sequence is the first block_size tokens
        inputs = chunk[: self.block_size]
        x = torch.tensor(inputs, dtype=torch.long)

        # Create targets with instruction tokens masked out
        targets, future_targets = self.create_targets(chunk, original_length)

        # Use strategy-specific attention mask if available
        if self.strategy is not None and hasattr(
            self.strategy, "create_instruction_attention_mask"
        ):
            x_batch = x.unsqueeze(0)
            attention_mask = self.strategy.create_instruction_attention_mask(x_batch)
            attention_mask = attention_mask.squeeze(0)

            # Create instruction mask using targets tensor
            targets_batch = targets.unsqueeze(0)
            instruction_mask = self.strategy._create_instruction_mask_vectorized(
                targets_batch
            ).squeeze(0)
        else:
            padding_mask = self.create_padding_mask(original_length)
            causal_mask = torch.tril(torch.ones(self.block_size, self.block_size))
            attention_mask = padding_mask.unsqueeze(1) * causal_mask
            instruction_mask = None

        return {
            "inputs": x,
            "targets": targets,
            "future_targets": future_targets,
            "attention_mask": attention_mask,
            "instruction_mask": instruction_mask,
        }

    def pad_sequence(self, sequence, target_length):
        """Pad sequence with strategy-specific padding token"""
        padding_value = self.strategy.get_padding_value()
        return sequence + [padding_value] * (target_length - len(sequence))


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
    use_news: bool = False,
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
        use_news=use_news,
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
        """Collate function supporting auxiliary heads and attention masks."""
        if not batch:
            return {
                "inputs": [],
                "targets": [],
                "future_targets": [],
                "attention_mask": [],
                "instruction_mask": [],
            }

        # Stack inputs, targets, and attention masks
        inputs = torch.stack([item["inputs"] for item in batch])
        targets = torch.stack([item["targets"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        instruction_mask = torch.stack([item["instruction_mask"] for item in batch])

        # Handle future targets for auxiliary heads
        future_targets = []
        if batch[0]["future_targets"]:  # Check if we have auxiliary heads
            for i in range(len(batch[0]["future_targets"])):
                future_target = torch.stack(
                    [item["future_targets"][i] for item in batch]
                )
                future_targets.append(future_target)

        return {
            "inputs": inputs,
            "targets": targets,
            "future_targets": future_targets,
            "attention_mask": attention_mask,
            "instruction_mask": instruction_mask,
        }

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


def load_alternative_bodies(conn_pool) -> pd.DataFrame:
    """
    Load all alternative bodies from msh_news_embeddings table that are not empty.

    Args:
        conn_pool: Database connection pool

    Returns:
        DataFrame containing news_uuid, link, title, and alternative_body
    """
    conn = None
    try:
        conn = conn_pool.getconn()
        cur = conn.cursor()

        query = """
        SELECT 
            news_uuid,
            link,
            title,
            alternative_body
        FROM 
            msh_news_embeddings
        WHERE 
            updated_at >= '2025-02-18'
            AND alternative_body IS NOT NULL 
            AND alternative_body != ''
        ORDER BY 
            created_at DESC limit 500000
        """

        cur.execute(query)
        rows = cur.fetchall()

        df = pd.DataFrame(
            rows, columns=["news_uuid", "link", "title", "alternative_body"]
        )

        df["word_count"] = df["alternative_body"].apply(lambda x: len(str(x).split()))
        df = df[df["word_count"] >= 100]

        return df

    except psycopg2.Error as e:
        print(f"Database error while loading alternative bodies: {e}")
        raise
    finally:
        if conn is not None:
            conn_pool.putconn(conn)


def write_alternative_bodies_to_files(
    df: pd.DataFrame, output_dir: str = "/home/alexander/Tank3/training_data/news_texts"
) -> None:
    """
    Write alternative bodies to individual text files named by news_uuid.

    Args:
        df: DataFrame containing news_uuid and normalized alternative_body columns
        output_dir: Directory where text files will be stored (default: 'news_texts')
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through DataFrame and write files with tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing files"):
        filename = os.path.join(output_dir, f"{row['news_uuid']}.txt")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(row["alternative_body"]))
        except Exception as e:
            print(f"Error writing file {filename}: {e}")
            continue

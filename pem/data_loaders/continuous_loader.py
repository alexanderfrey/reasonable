"""
Continuous Data Loader for Persistent Sync Training.

Key design decisions (from the plan):
- Data format: Continuous stream of pages (no batch complexity)
- Reset policy: Never reset (accumulate knowledge across all books)
- Natural reading flow: Stream pages sequentially across books

This loader is designed for training the emergent world model via persistent
sync patterns. The world state (S_world) accumulates across all pages, building
a representation of the reader's knowledge over time.

Usage:
    stream = ContinuousFeatureStream(
        feature_paths=['book1_features.pt', 'book2_features.pt'],
        context_size=512,
    )

    for page_features in stream:
        output = model(page_features, world_state=world_state)
        loss.backward()
        optimizer.step()
        model.commit_world_state(output.world_state.detach())
"""

from typing import List, Iterator, Optional, Union
from pathlib import Path
from itertools import cycle

import torch
from torch.utils.data import IterableDataset


class ContinuousPageStream(IterableDataset):
    """
    Yields pages continuously: book1_p1, book1_p2, ..., book2_p1, ...

    For text-based training with tokenization.

    Args:
        book_paths: List of paths to book files (text or pre-tokenized)
        context_size: Number of tokens per page
        tokenizer: Optional tokenizer (if passing text files)
        loop_forever: Whether to cycle through books indefinitely
    """

    def __init__(
        self,
        book_paths: List[Union[str, Path]],
        context_size: int = 512,
        tokenizer=None,
        loop_forever: bool = True,
    ):
        self.book_paths = [Path(p) for p in book_paths]
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.loop_forever = loop_forever

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield pages continuously from all books."""
        book_iter = cycle(self.book_paths) if self.loop_forever else iter(self.book_paths)

        for book_path in book_iter:
            for page in self._load_book_pages(book_path):
                yield page

    def _load_book_pages(self, book_path: Path) -> Iterator[torch.Tensor]:
        """Load and yield pages from a single book."""
        if book_path.suffix == '.pt':
            # Pre-tokenized tensor file
            tokens = torch.load(book_path)
            yield from self._chunk_tokens(tokens)
        elif book_path.suffix in ('.txt', '.md'):
            # Text file - tokenize on the fly
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for text files")
            with open(book_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = self.tokenizer.encode(text, return_tensors='pt')[0]
            yield from self._chunk_tokens(tokens)
        else:
            raise ValueError(f"Unsupported file type: {book_path.suffix}")

    def _chunk_tokens(self, tokens: torch.Tensor) -> Iterator[torch.Tensor]:
        """Chunk tokens into pages of context_size."""
        total_tokens = tokens.shape[0]
        for i in range(0, total_tokens - self.context_size, self.context_size):
            yield tokens[i:i + self.context_size]


class ContinuousFeatureStream(IterableDataset):
    """
    Yields pre-computed feature tensors continuously.

    For training on pre-extracted Janus/backbone features. This is the primary
    loader for persistent sync training - features are pre-computed and stored,
    allowing the model to focus on learning sync patterns.

    Args:
        feature_paths: List of paths to feature files (.pt or .safetensors)
        context_size: Number of positions per page (for chunking if needed)
        loop_forever: Whether to cycle through books indefinitely
        shuffle_books: Whether to shuffle book order (default False for reproducibility)
    """

    def __init__(
        self,
        feature_paths: List[Union[str, Path]],
        context_size: int = 512,
        loop_forever: bool = True,
        shuffle_books: bool = False,
    ):
        self.feature_paths = [Path(p) for p in feature_paths]
        self.context_size = context_size
        self.loop_forever = loop_forever
        self.shuffle_books = shuffle_books

        # Validate paths
        for path in self.feature_paths:
            if not path.exists():
                raise FileNotFoundError(f"Feature file not found: {path}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield feature pages continuously from all books."""
        paths = list(self.feature_paths)

        if self.shuffle_books:
            import random
            random.shuffle(paths)

        book_iter = cycle(paths) if self.loop_forever else iter(paths)

        for book_path in book_iter:
            for page in self._load_book_features(book_path):
                yield page

    def _load_book_features(self, book_path: Path) -> Iterator[torch.Tensor]:
        """Load and yield feature pages from a single book."""
        if book_path.suffix == '.pt':
            features = torch.load(book_path, weights_only=True)
        elif book_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            data = load_file(str(book_path))
            # Assume 'features' key or first tensor
            features = data.get('features', list(data.values())[0])
        else:
            raise ValueError(f"Unsupported file type: {book_path.suffix}")

        # features: (seq_len, d_model) or (num_pages, seq_len, d_model)
        if features.dim() == 2:
            # Single long sequence - chunk into pages
            yield from self._chunk_features(features)
        elif features.dim() == 3:
            # Already chunked into pages
            for i in range(features.shape[0]):
                yield features[i]  # (seq_len, d_model)
        else:
            raise ValueError(f"Expected 2D or 3D features, got {features.dim()}D")

    def _chunk_features(self, features: torch.Tensor) -> Iterator[torch.Tensor]:
        """Chunk features into pages of context_size."""
        seq_len, d_model = features.shape
        for i in range(0, seq_len - self.context_size, self.context_size):
            yield features[i:i + self.context_size]


class ContinuousHuggingFaceStream(IterableDataset):
    """
    Yields pages continuously from a HuggingFace dataset.

    For training on streaming datasets like 'pg19' (Project Gutenberg).

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (e.g., 'train')
        text_column: Column containing the text
        context_size: Number of tokens per page
        tokenizer: Tokenizer for encoding text
        loop_forever: Whether to cycle indefinitely
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = 'train',
        text_column: str = 'text',
        context_size: int = 512,
        tokenizer=None,
        loop_forever: bool = True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.context_size = context_size
        self.tokenizer = tokenizer
        self.loop_forever = loop_forever

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield pages continuously from the dataset."""
        from datasets import load_dataset

        while True:
            dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)

            for item in dataset:
                text = item[self.text_column]
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(text, return_tensors='pt')[0]
                    for i in range(0, len(tokens) - self.context_size, self.context_size):
                        yield tokens[i:i + self.context_size]
                else:
                    # Yield raw text (caller handles tokenization)
                    yield text

            if not self.loop_forever:
                break


def create_continuous_loader(
    feature_paths: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    context_size: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    **kwargs,
):
    """
    Factory function to create a continuous data loader.

    Args:
        feature_paths: List of paths to pre-computed feature files
        dataset_name: HuggingFace dataset name (alternative to feature_paths)
        context_size: Number of positions per page
        batch_size: Batch size (typically 1 for continuous streaming)
        num_workers: Number of data loading workers

    Returns:
        DataLoader wrapping the appropriate continuous stream
    """
    from torch.utils.data import DataLoader

    if feature_paths is not None:
        stream = ContinuousFeatureStream(
            feature_paths=feature_paths,
            context_size=context_size,
            **kwargs,
        )
    elif dataset_name is not None:
        stream = ContinuousHuggingFaceStream(
            dataset_name=dataset_name,
            context_size=context_size,
            **kwargs,
        )
    else:
        raise ValueError("Either feature_paths or dataset_name must be provided")

    return DataLoader(
        stream,
        batch_size=batch_size,
        num_workers=num_workers,
        # No shuffle - streaming is sequential for world model continuity
    )

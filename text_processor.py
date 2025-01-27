import spacy
from typing import List, Optional, Iterator
import re
from pathlib import Path
import logging
import multiprocessing as mp
from tqdm.auto import tqdm

from itertools import chain
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


class TextProcessor:
    """
    A class to handle text processing with parallel sentence segmentation.
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        cache_dir: Optional[Path] = None,
        batch_size: int = 1000,
        n_process: Optional[int] = None,
    ):
        """
        Initialize the text processor.

        Args:
            model: spaCy model to use
            cache_dir: Directory to cache the model
            batch_size: Number of characters per batch for parallel processing
            n_process: Number of processes to use (defaults to CPU count - 1)
        """
        self.batch_size = batch_size
        self.n_process = n_process or max(1, 6)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load spaCy model
        try:
            self.nlp = spacy.load(model)
            self.nlp.add_pipe("sentencizer")
            self.nlp.disable_pipes(
                ["ner", "tagger", "parser", "attribute_ruler", "lemmatizer"]
            )
        except OSError:
            self.logger.info(f"Downloading spaCy model {model}...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

        # Compile regex patterns
        self.cleanup_patterns = {
            "multiple_spaces": re.compile(r"\s+"),
            "multiple_newlines": re.compile(r"\n+"),
            "special_chars": re.compile(r'[^\w\s.,!?"\'-]'),
            "standalone_numbers": re.compile(r"^\d+\.?\d*$"),
        }

        # Define sentence quality criteria
        self.min_words = 3
        self.max_words = 100
        self.min_chars = 10
        self.max_chars = 512

    def clean_text(self, text: str) -> str:
        """Clean raw text before processing."""
        text = text.strip()
        # Preserve colons for character names
        text = self.cleanup_patterns["multiple_newlines"].sub("\n", text)
        text = self.cleanup_patterns["multiple_spaces"].sub(" ", text)
        # Modified special chars pattern to preserve colons
        text = re.sub(r'[^\w\s.,!?:"\'-]', " ", text)
        return text

    def is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence meets quality criteria for dramatic text."""
        sentence = sentence.strip()
        words = sentence.split()
        word_count = len(words)

        # Remove strict punctuation requirement
        if len(sentence) > self.max_chars:
            return False
        if all(
            self.cleanup_patterns["standalone_numbers"].match(word) for word in words
        ):
            return False

        return True

    def _split_text_into_chunks(
        self, text: str, show_progress: bool = True
    ) -> List[str]:
        """Split text into roughly equal chunks for parallel processing."""
        # Find sentence boundary candidates
        if show_progress:
            print("Finding sentence boundaries...")
        boundary_positions = [m.start() for m in re.finditer(r"[.!?:]\s+", text)]

        if not boundary_positions:
            return [text]

        # Calculate optimal chunk size
        n_chunks = max(1, len(text) // self.batch_size)
        chunk_size = len(text) // n_chunks

        chunks = []
        current_pos = 0

        if show_progress:
            chunk_iterator = tqdm(range(n_chunks), desc="Splitting text into chunks")
        else:
            chunk_iterator = range(n_chunks)

        for i in chunk_iterator:
            if i == n_chunks - 1:
                chunks.append(text[current_pos:])
                break

            # Find the nearest sentence boundary
            target_pos = current_pos + chunk_size
            nearest_boundary = min(
                boundary_positions, key=lambda x: abs(x - target_pos)
            )

            # Add chunk and update position
            chunks.append(text[current_pos : nearest_boundary + 1])
            current_pos = nearest_boundary + 1

        return chunks

    def _process_chunk(self, chunk: str) -> List[str]:
        """Process a single chunk of text."""
        chunk = self.clean_text(chunk)
        try:
            doc = self.nlp(chunk)
            sentences = []
            for sent in doc.sents:
                cleaned_sent = sent.text.strip()
                if self.is_valid_sentence(cleaned_sent):
                    sentences.append(cleaned_sent)
            return sentences
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            return []

    def process_text_parallel(self, text: str, show_progress: bool = True) -> List[str]:
        """
        Process text into sentences using parallel processing.

        Args:
            text: Input text to be processed
            show_progress: Whether to show progress bar

        Returns:
            List of processed sentences
        """
        if show_progress:
            print("Starting parallel text processing...")

        # Split text into chunks with progress
        chunks = self._split_text_into_chunks(text, show_progress)

        # Process chunks in parallel
        sentences = []
        with ProcessPoolExecutor(max_workers=self.n_process) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]

            # Handle progress bar for chunk processing
            if show_progress:
                futures_iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing chunks",
                    position=0,
                    leave=True,
                )
            else:
                futures_iterator = as_completed(futures)

            # Collect results
            for future in futures_iterator:
                try:
                    chunk_sentences = future.result()
                    sentences.extend(chunk_sentences)
                except Exception as e:
                    self.logger.error(f"Error processing chunk: {str(e)}")

        if show_progress:
            print(f"Processed {len(sentences)} sentences from {len(chunks)} chunks")

        return sentences

    def merge_short_sentences(
        self, sentences: List[str], min_length: int = 50
    ) -> List[str]:
        """Merge short sentences with neighboring sentences."""
        if not sentences:
            return []

        merged = []
        current = sentences[0]

        for next_sent in sentences[1:]:
            if len(current) < min_length:
                current = current[:-1] + " " + next_sent
            else:
                merged.append(current)
                current = next_sent

        merged.append(current)
        return merged

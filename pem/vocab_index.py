"""
Vocabulary Embedding Index for PredictionCTM inference.

Pre-computes embeddings for all vocabulary tokens to enable
fast similarity search for predicted embeddings.

Usage:
    index = VocabEmbeddingIndex(feature_extractor, device='cuda')

    # Get top-k similar tokens for a prediction
    top_k = index.top_k_similar(prediction_embedding, k=5)
    # Returns: [("mat", 0.87), ("floor", 0.82), ...]
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VocabEmbeddingIndex:
    """
    Pre-computed embedding index for vocabulary tokens.

    Embeds all tokens in the vocabulary using the feature extractor,
    then provides fast cosine similarity search.
    """

    def __init__(
        self,
        feature_extractor,
        device: str = 'cuda',
        cache_path: Optional[str] = None,
        batch_size: int = 128,
    ):
        """
        Initialize the vocabulary embedding index.

        Args:
            feature_extractor: JanusProFeatureExtractor instance
            device: Device to store embeddings on
            cache_path: Optional path to cache embeddings
            batch_size: Batch size for embedding computation
        """
        self.feature_extractor = feature_extractor
        self.device = device
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None

        # Get tokenizer from feature extractor
        self.tokenizer = feature_extractor.tokenizer
        self.vocab_size = len(self.tokenizer)

        # Embeddings will be computed lazily or loaded from cache
        self._embeddings: Optional[torch.Tensor] = None
        self._token_strings: Optional[List[str]] = None

        logger.info(f"VocabEmbeddingIndex initialized for vocab size {self.vocab_size}")

    def _build_index(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Build the embedding index by encoding all vocabulary tokens.

        Returns:
            embeddings: (vocab_size, d_model) tensor of normalized embeddings
            token_strings: List of decoded token strings
        """
        logger.info(f"Building vocabulary embedding index ({self.vocab_size} tokens)...")

        # Decode all tokens to strings for display
        token_strings = []
        for token_id in range(self.vocab_size):
            try:
                token_str = self.tokenizer.decode([token_id])
                token_strings.append(token_str)
            except Exception:
                token_strings.append(f"<unk_{token_id}>")

        # Compute embeddings in batches
        all_embeddings = []

        with torch.no_grad():
            for start_idx in range(0, self.vocab_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.vocab_size)
                batch_ids = list(range(start_idx, end_idx))

                # Create input tensors - each token as a single-token sequence
                # We'll embed them and take the last (only) hidden state
                input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device).unsqueeze(1)

                # Get features from the feature extractor
                features = self.feature_extractor(input_ids)  # (B, 1, d_model)

                # Take the embedding at position 0 (the only position)
                embeddings = features[:, 0, :]  # (B, d_model)

                all_embeddings.append(embeddings.cpu())

                if (start_idx // self.batch_size) % 50 == 0:
                    logger.info(f"  Embedded {end_idx}/{self.vocab_size} tokens...")

        # Stack all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)  # (vocab_size, d_model)

        # Normalize for cosine similarity
        embeddings = F.normalize(embeddings, dim=-1)

        logger.info(f"Built embedding index: {embeddings.shape}")

        return embeddings, token_strings

    def _ensure_loaded(self):
        """Ensure embeddings are loaded (from cache or computed)."""
        if self._embeddings is not None:
            return

        # Try to load from cache
        if self.cache_path and self.cache_path.exists():
            logger.info(f"Loading vocab embeddings from cache: {self.cache_path}")
            cached = torch.load(self.cache_path, map_location='cpu')
            self._embeddings = cached['embeddings']
            self._token_strings = cached['token_strings']
            logger.info(f"Loaded {self._embeddings.shape[0]} embeddings from cache")
        else:
            # Build the index
            self._embeddings, self._token_strings = self._build_index()

            # Save to cache if path provided
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'embeddings': self._embeddings,
                    'token_strings': self._token_strings,
                }, self.cache_path)
                logger.info(f"Saved vocab embeddings to cache: {self.cache_path}")

        # Convert to float32 once for fast similarity computation
        self._embeddings = self._embeddings.float()
        logger.info(f"Embeddings ready: {self._embeddings.shape}, dtype={self._embeddings.dtype}")

    @property
    def embeddings(self) -> torch.Tensor:
        """Get the embedding matrix (vocab_size, d_model)."""
        self._ensure_loaded()
        return self._embeddings

    @property
    def token_strings(self) -> List[str]:
        """Get the list of decoded token strings."""
        self._ensure_loaded()
        return self._token_strings

    def top_k_similar(
        self,
        prediction: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find the top-k most similar tokens to a prediction embedding.

        Args:
            prediction: (d_model,) or (1, d_model) prediction embedding
            k: Number of top results to return

        Returns:
            List of (token_string, similarity_score) tuples
        """
        self._ensure_loaded()

        # Ensure prediction is the right shape
        if prediction.dim() == 2:
            prediction = prediction.squeeze(0)

        # Normalize prediction for cosine similarity
        prediction = F.normalize(prediction, dim=-1)

        # Move to CPU and convert to float32 for comparison with cached embeddings
        prediction = prediction.cpu().float()

        # Compute cosine similarity with all vocab embeddings (already float32)
        similarities = torch.matmul(self._embeddings, prediction)  # (vocab_size,)

        # Get top-k
        top_k_values, top_k_indices = similarities.topk(k)

        results = []
        for idx, sim in zip(top_k_indices.tolist(), top_k_values.tolist()):
            token_str = self._token_strings[idx]
            results.append((token_str, sim))

        return results

    def top_k_for_sequence(
        self,
        predictions: torch.Tensor,
        k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """
        Find top-k similar tokens for each position in a sequence.

        Args:
            predictions: (S, d_model) or (B, S, d_model) prediction embeddings
            k: Number of top results per position

        Returns:
            List of lists of (token_string, similarity_score) tuples
        """
        self._ensure_loaded()

        # Handle batch dimension
        if predictions.dim() == 3:
            predictions = predictions[0]  # Take first batch item

        results = []
        for pos in range(predictions.shape[0]):
            pos_results = self.top_k_similar(predictions[pos], k=k)
            results.append(pos_results)

        return results

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Get the embedding for a specific token ID."""
        self._ensure_loaded()
        return self._embeddings[token_id]

    def find_token_rank(
        self,
        prediction: torch.Tensor,
        target_token_id: int,
    ) -> Tuple[int, float]:
        """
        Find where a target token ranks in similarity to a prediction.

        Args:
            prediction: (d_model,) prediction embedding
            target_token_id: Token ID to find rank for

        Returns:
            (rank, similarity) tuple where rank is 0-indexed
        """
        self._ensure_loaded()

        # Ensure prediction is the right shape
        if prediction.dim() == 2:
            prediction = prediction.squeeze(0)

        # Normalize prediction and convert to float32
        prediction = F.normalize(prediction, dim=-1).cpu().float()

        # Compute all similarities (embeddings already float32)
        similarities = torch.matmul(self._embeddings, prediction)

        # Get the similarity for target token
        target_sim = similarities[target_token_id].item()

        # Count how many tokens have higher similarity
        rank = (similarities > target_sim).sum().item()

        return rank, target_sim

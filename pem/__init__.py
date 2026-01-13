"""
Predictive Experience Machine (PEM) - Neural architecture that experiences what it reads.

A system that constantly generates expectations, experiences surprise when wrong,
and maintains episodic memory of those surprises.
"""

from .feature_extractor import (
    FeatureExtractorConfig,
    FeatureExtractor,
    Qwen3VLFeatureExtractor,
)

__all__ = [
    "FeatureExtractorConfig",
    "FeatureExtractor",
    "Qwen3VLFeatureExtractor",
]

"""
Predictive Experience Machine (PEM) - Neural architecture that experiences what it reads.

A system that constantly generates expectations, experiences surprise when wrong,
and maintains episodic memory of those surprises.
"""

from .feature_extractor import (
    FeatureExtractorConfig,
    FeatureExtractor,
    Qwen3VLFeatureExtractor,
    LearningMode,
    create_feature_extractor,
)

from .prediction_module import (
    PredictionConfig,
    PredictionModule,
    PredictionTargets,
    PredictionLoss,
    create_prediction_module,
)

__all__ = [
    # Feature extraction
    "FeatureExtractorConfig",
    "FeatureExtractor",
    "Qwen3VLFeatureExtractor",
    "LearningMode",
    "create_feature_extractor",
    # Prediction
    "PredictionConfig",
    "PredictionModule",
    "PredictionTargets",
    "PredictionLoss",
    "create_prediction_module",
]

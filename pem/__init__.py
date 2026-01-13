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

from .surprise_module import (
    SurpriseConfig,
    SurpriseModule,
    SurpriseLoss,
    create_surprise_module,
)

from .sync_module import (
    SyncModuleConfig,
    SyncModule,
    ChangePointDetector,
    MemoryBank,
    PersonalityModule,
    IntentionConfig,
    IntentionModule,
    create_sync_module,
)

from .perception_attention import (
    PerceptionConfig,
    PerceptionAttention,
    PerceptionKVCache,
    OscillationQueryBuilder,
    PerceptionCrossAttention,
    PerceptionSynapse,
    PerceptionOutput,
    create_perception_attention,
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
    # Surprise
    "SurpriseConfig",
    "SurpriseModule",
    "SurpriseLoss",
    "create_surprise_module",
    # Sync
    "SyncModuleConfig",
    "SyncModule",
    "ChangePointDetector",
    "MemoryBank",
    "PersonalityModule",
    "IntentionConfig",
    "IntentionModule",
    "create_sync_module",
    # Perception Attention
    "PerceptionConfig",
    "PerceptionAttention",
    "PerceptionKVCache",
    "OscillationQueryBuilder",
    "PerceptionCrossAttention",
    "PerceptionSynapse",
    "PerceptionOutput",
    "create_perception_attention",
]

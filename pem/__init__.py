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
    # Valence (affective dimension of experience)
    ValenceConfig,
    ValenceModule,
    ValenceLoss,
    create_valence_module,
)

from .curiosity_module import (
    CuriosityConfig,
    CuriosityModule,
    CuriosityOutput,
    CuriosityLoss,
    UncertaintyEstimator,
    NoveltyMemory,
    InformationGainComputer,
    create_curiosity_module,
)

from .activation_module import (
    ActivationConfig,
    ActivationModule,
    ActivationOutput,
    ActivationLoss,
    ArousalComputer,
    ModulationComputer,
    create_activation_module,
)

from .imagination_module import (
    ImaginationConfig,
    ImaginationModule,
    ImaginationOutput,
    ImaginationLoss,
    SceneGenerator,
    MindModeler,
    CounterfactualGenerator,
    create_imagination_module,
)

from .generative_core import (
    GenerativeCoreConfig,
    GenerativeCore,
    GenerativeEncoder,
    GenerativeDecoder,
    ModeFusion,
    create_generative_core,
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
    # Valence (affective dimension)
    "ValenceConfig",
    "ValenceModule",
    "ValenceLoss",
    "create_valence_module",
    # Curiosity (epistemic drive)
    "CuriosityConfig",
    "CuriosityModule",
    "CuriosityOutput",
    "CuriosityLoss",
    "UncertaintyEstimator",
    "NoveltyMemory",
    "InformationGainComputer",
    "create_curiosity_module",
    # Activation (arousal/engagement)
    "ActivationConfig",
    "ActivationModule",
    "ActivationOutput",
    "ActivationLoss",
    "ArousalComputer",
    "ModulationComputer",
    "create_activation_module",
    # Imagination (mental simulation)
    "ImaginationConfig",
    "ImaginationModule",
    "ImaginationOutput",
    "ImaginationLoss",
    "SceneGenerator",
    "MindModeler",
    "CounterfactualGenerator",
    "create_imagination_module",
    # Generative Core (shared prediction/imagination)
    "GenerativeCoreConfig",
    "GenerativeCore",
    "GenerativeEncoder",
    "GenerativeDecoder",
    "ModeFusion",
    "create_generative_core",
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

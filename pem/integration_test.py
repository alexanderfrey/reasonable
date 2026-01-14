#!/usr/bin/env python3
"""
Gradual Integration Test for PEM.

Tests modules incrementally and visualizes their behavior:
1. Feature Extractor alone
2. + Prediction Module
3. + Surprise Module
4. + Valence Module
5. + Curiosity Module
6. + Full pipeline

Usage:
    PYTHONPATH=. python pem/integration_test.py
    PYTHONPATH=. python pem/integration_test.py --text "Your custom text here"
    PYTHONPATH=. python pem/integration_test.py --file data/eval_corpus.txt
"""

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default test texts with varying complexity
DEFAULT_TEXTS = [
    # Simple, predictable
    "The cat sat on the mat. The dog sat on the rug. The bird sat on the branch.",
    # More complex narrative
    "The old house at the end of the street had been abandoned for years. Nobody knew what happened to the family that once lived there. Some said they moved away. Others whispered darker stories.",
    # Surprising twist
    "She opened the door expecting to see her husband. Instead, standing in the rain, was a stranger holding a photograph of her grandmother.",
    # Abstract/philosophical
    "Time flows like a river, yet we cannot step into the same moment twice. Each second brings new possibilities, new chances for change.",
    # Technical/factual
    "The neural network processes input through multiple layers. Each layer transforms the representation, extracting increasingly abstract features.",
]


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class IntegrationTester:
    """Gradual integration tester for PEM modules."""

    def __init__(self, device: torch.device):
        self.device = device
        self.results = {}

        # Module instances (created lazily)
        self._feature_extractor = None
        self._prediction_module = None
        self._surprise_module = None
        self._valence_module = None
        self._curiosity_module = None
        self._activation_module = None
        self._sync_module = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from pem import create_feature_extractor
            logger.info("Loading Janus Pro feature extractor...")
            self._feature_extractor = create_feature_extractor(
                model_name_or_path="deepseek-ai/Janus-Pro-1B",
                output_dim=1536,
                learning_mode="frozen",
            )
        return self._feature_extractor

    @property
    def prediction_module(self):
        if self._prediction_module is None:
            from pem import PredictionModule, PredictionConfig
            config = PredictionConfig(
                sync_pairs=512,
                d_model=1536,
                n_head=8,
                immediate_horizon=8,
                shortterm_horizon=64,
                longterm_horizon=256,
            )
            self._prediction_module = PredictionModule(config).to(self.device)
        return self._prediction_module

    @property
    def surprise_module(self):
        if self._surprise_module is None:
            from pem import SurpriseModule, SurpriseConfig
            config = SurpriseConfig(
                d_model=1536,
                hidden_dim=768,
                scales=["immediate", "shortterm", "longterm"],
            )
            self._surprise_module = SurpriseModule(config).to(self.device)
        return self._surprise_module

    @property
    def valence_module(self):
        if self._valence_module is None:
            from pem import ValenceModule, ValenceConfig
            config = ValenceConfig(
                d_model=1536,
                personality_dim=512,
                hidden_dim=768,
            )
            self._valence_module = ValenceModule(config).to(self.device)
        return self._valence_module

    @property
    def curiosity_module(self):
        if self._curiosity_module is None:
            from pem import CuriosityModule, CuriosityConfig
            config = CuriosityConfig(
                d_model=1536,
                hidden_dim=768,
            )
            self._curiosity_module = CuriosityModule(config).to(self.device)
        return self._curiosity_module

    def extract_features(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Extract features from text using Janus Pro."""
        self.feature_extractor._ensure_loaded()
        tokenizer = self.feature_extractor.tokenizer

        # Tokenize
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        # Get token strings for visualization
        tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_ids)
            # Ensure float32 for downstream modules
            features = features.float()

        return features, tokens

    def test_prediction_only(self, texts: List[str]) -> Dict:
        """Test prediction module alone."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Feature Extraction + Prediction")
        logger.info("="*60)

        from pem import PredictionTargets

        all_results = []

        for i, text in enumerate(texts):
            logger.info(f"\nText {i+1}: {text[:50]}...")

            # Extract features
            features, tokens = self.extract_features(text)
            B, S, D = features.shape
            logger.info(f"  Tokens: {S}, Features: {D}")

            # Create mock sync (since we don't have full pipeline yet)
            # In real use, this comes from SyncModule
            sync = torch.randn(B, S, 512, device=self.device)

            # Run prediction
            with torch.no_grad():
                predictions = self.prediction_module(sync, features)

            # Compute targets (what we should have predicted)
            target_computer = PredictionTargets(
                immediate_horizon=8,
                shortterm_horizon=64,
                longterm_horizon=256,
            )
            targets = target_computer.compute_targets(features)

            # Compute prediction errors at each position
            errors = {}
            for scale in ["immediate", "shortterm", "longterm"]:
                pred = predictions[scale]
                target = targets[scale]
                valid = targets[f"{scale}_valid"]

                # Cosine distance (1 - cosine_similarity)
                cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
                cos_dist = 1 - cos_sim  # 0 = perfect, 2 = opposite

                # Mask invalid positions
                cos_dist = cos_dist * valid.float()

                errors[scale] = cos_dist[0].cpu().numpy()  # First batch

            result = {
                "text": text,
                "tokens": tokens,
                "num_tokens": S,
                "errors": errors,
                "features_mean": features.mean().item(),
                "features_std": features.std().item(),
            }
            all_results.append(result)

            # Log summary
            for scale in ["immediate", "shortterm", "longterm"]:
                valid_errors = errors[scale][errors[scale] > 0]
                if len(valid_errors) > 0:
                    logger.info(f"  {scale}: mean_error={valid_errors.mean():.4f}, std={valid_errors.std():.4f}")

        return {"stage": "prediction", "results": all_results}

    def test_with_surprise(self, texts: List[str]) -> Dict:
        """Test prediction + surprise modules."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: + Surprise Module")
        logger.info("="*60)

        from pem import PredictionTargets

        all_results = []

        for i, text in enumerate(texts):
            logger.info(f"\nText {i+1}: {text[:50]}...")

            features, tokens = self.extract_features(text)
            B, S, D = features.shape

            sync = torch.randn(B, S, 512, device=self.device)

            with torch.no_grad():
                predictions = self.prediction_module(sync, features)
                target_computer = PredictionTargets(
                    immediate_horizon=8, shortterm_horizon=64, longterm_horizon=256
                )
                targets = target_computer.compute_targets(features)

                # Compute surprise
                surprises = self.surprise_module(predictions, targets, features)

            result = {
                "text": text,
                "tokens": tokens,
                "num_tokens": S,
                "surprises": {},
            }

            for scale in ["immediate", "shortterm", "longterm"]:
                surprise = surprises[scale]
                magnitude = surprise["magnitude"][0, :, 0].cpu().numpy()
                raw = surprise["raw"][0, :, 0].cpu().numpy()

                result["surprises"][scale] = {
                    "magnitude": magnitude,
                    "raw": raw,
                }

                logger.info(f"  {scale} surprise: mean={magnitude.mean():.4f}, max={magnitude.max():.4f}")

            all_results.append(result)

        return {"stage": "surprise", "results": all_results}

    def test_with_valence(self, texts: List[str]) -> Dict:
        """Test prediction + surprise + valence modules."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: + Valence Module")
        logger.info("="*60)

        from pem import PredictionTargets

        all_results = []

        # Create a mock personality embedding
        personality = torch.randn(512, device=self.device)
        personality = personality / personality.norm()  # Normalize

        for i, text in enumerate(texts):
            logger.info(f"\nText {i+1}: {text[:50]}...")

            features, tokens = self.extract_features(text)
            B, S, D = features.shape

            sync = torch.randn(B, S, 512, device=self.device)

            with torch.no_grad():
                predictions = self.prediction_module(sync, features)
                target_computer = PredictionTargets(
                    immediate_horizon=8, shortterm_horizon=64, longterm_horizon=256
                )
                targets = target_computer.compute_targets(features)
                surprises = self.surprise_module(predictions, targets, features)

                # Compute valence for each scale
                valences = {}
                for scale in ["immediate", "shortterm", "longterm"]:
                    direction = surprises[scale]["direction"]
                    valence = self.valence_module(direction, personality, features)
                    valences[scale] = valence[0, :, 0].cpu().numpy()

            result = {
                "text": text,
                "tokens": tokens,
                "num_tokens": S,
                "surprises": {s: {"magnitude": surprises[s]["magnitude"][0, :, 0].cpu().numpy()}
                             for s in ["immediate", "shortterm", "longterm"]},
                "valences": valences,
            }

            for scale in ["immediate", "shortterm", "longterm"]:
                v = valences[scale]
                logger.info(f"  {scale} valence: mean={v.mean():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")

            all_results.append(result)

        return {"stage": "valence", "results": all_results}

    def test_with_curiosity(self, texts: List[str]) -> Dict:
        """Test prediction + surprise + valence + curiosity modules."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: + Curiosity Module")
        logger.info("="*60)

        from pem import PredictionTargets

        all_results = []
        personality = torch.randn(512, device=self.device)
        personality = personality / personality.norm()

        for i, text in enumerate(texts):
            logger.info(f"\nText {i+1}: {text[:50]}...")

            features, tokens = self.extract_features(text)
            B, S, D = features.shape

            sync = torch.randn(B, S, 512, device=self.device)

            with torch.no_grad():
                predictions = self.prediction_module(sync, features)
                target_computer = PredictionTargets(
                    immediate_horizon=8, shortterm_horizon=64, longterm_horizon=256
                )
                targets = target_computer.compute_targets(features)
                surprises = self.surprise_module(predictions, targets, features)

                # Valence
                valences = {}
                for scale in ["immediate", "shortterm", "longterm"]:
                    direction = surprises[scale]["direction"]
                    valence = self.valence_module(direction, personality, features)
                    valences[scale] = valence[0, :, 0].cpu().numpy()

                # Curiosity
                curiosity_out = self.curiosity_module(features, predictions["immediate"])

            result = {
                "text": text,
                "tokens": tokens,
                "num_tokens": S,
                "surprises": {s: {"magnitude": surprises[s]["magnitude"][0, :, 0].cpu().numpy()}
                             for s in ["immediate", "shortterm", "longterm"]},
                "valences": valences,
                "curiosity": curiosity_out.curiosity[0, :, 0].cpu().numpy(),
                "uncertainty": curiosity_out.uncertainty[0, :, 0].cpu().numpy(),
                "info_gain": curiosity_out.information_gain[0, :, 0].cpu().numpy(),
            }

            logger.info(f"  curiosity: mean={result['curiosity'].mean():.4f}")
            logger.info(f"  uncertainty: mean={result['uncertainty'].mean():.4f}")
            logger.info(f"  info_gain: mean={result['info_gain'].mean():.4f}")

            all_results.append(result)

        return {"stage": "curiosity", "results": all_results}

    def run_all_stages(self, texts: List[str]) -> Dict:
        """Run all integration stages."""
        results = {}

        results["prediction"] = self.test_prediction_only(texts)
        results["surprise"] = self.test_with_surprise(texts)
        results["valence"] = self.test_with_valence(texts)
        results["curiosity"] = self.test_with_curiosity(texts)

        return results


def plot_prediction_errors(results: Dict, output_path: str = "prediction_errors.png"):
    """Plot prediction errors across texts."""
    fig, axes = plt.subplots(len(results["results"]), 1, figsize=(14, 3*len(results["results"])))
    if len(results["results"]) == 1:
        axes = [axes]

    for idx, result in enumerate(results["results"]):
        ax = axes[idx]
        tokens = result["tokens"]
        errors = result["errors"]

        x = np.arange(len(tokens))
        width = 0.25

        for i, (scale, color) in enumerate([("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]):
            err = errors[scale]
            ax.bar(x + i*width, err, width, label=scale, alpha=0.7, color=color)

        ax.set_ylabel("Prediction Error\n(cosine distance)")
        ax.set_title(f"Text {idx+1}: {result['text'][:60]}...")
        ax.set_xticks(x + width)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved prediction errors plot to {output_path}")
    plt.close()


def plot_surprise_over_text(results: Dict, output_path: str = "surprise_analysis.png"):
    """Plot surprise magnitude across text positions."""
    n_texts = len(results["results"])
    fig, axes = plt.subplots(n_texts, 1, figsize=(14, 3*n_texts))
    if n_texts == 1:
        axes = [axes]

    for idx, result in enumerate(results["results"]):
        ax = axes[idx]
        tokens = result["tokens"]
        x = np.arange(len(tokens))

        for scale, color in [("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]:
            magnitude = result["surprises"][scale]["magnitude"]
            ax.plot(x, magnitude, label=f"{scale} surprise", color=color, alpha=0.8)

        ax.set_ylabel("Surprise Magnitude")
        ax.set_title(f"Text {idx+1}: {result['text'][:60]}...")
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved surprise analysis plot to {output_path}")
    plt.close()


def plot_valence_analysis(results: Dict, output_path: str = "valence_analysis.png"):
    """Plot valence (good/bad) across text positions."""
    n_texts = len(results["results"])
    fig, axes = plt.subplots(n_texts, 2, figsize=(16, 3*n_texts))
    if n_texts == 1:
        axes = axes.reshape(1, 2)

    for idx, result in enumerate(results["results"]):
        tokens = result["tokens"]
        x = np.arange(len(tokens))

        # Left: Surprise
        ax1 = axes[idx, 0]
        for scale, color in [("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]:
            magnitude = result["surprises"][scale]["magnitude"]
            ax1.plot(x, magnitude, label=f"{scale}", color=color, alpha=0.8)
        ax1.set_ylabel("Surprise")
        ax1.set_title(f"Text {idx+1} - Surprise")
        ax1.legend(loc="upper right")
        ax1.set_ylim(0, 1)

        # Right: Valence
        ax2 = axes[idx, 1]
        for scale, color in [("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]:
            valence = result["valences"][scale]
            ax2.plot(x, valence, label=f"{scale}", color=color, alpha=0.8)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax2.set_ylabel("Valence (+ good, - bad)")
        ax2.set_title(f"Text {idx+1} - Valence")
        ax2.legend(loc="upper right")
        ax2.set_ylim(-1, 1)

        # Shared x-axis labels
        for ax in [ax1, ax2]:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved valence analysis plot to {output_path}")
    plt.close()


def plot_curiosity_analysis(results: Dict, output_path: str = "curiosity_analysis.png"):
    """Plot curiosity, uncertainty, and information gain."""
    n_texts = len(results["results"])
    fig, axes = plt.subplots(n_texts, 3, figsize=(18, 3*n_texts))
    if n_texts == 1:
        axes = axes.reshape(1, 3)

    for idx, result in enumerate(results["results"]):
        tokens = result["tokens"]
        x = np.arange(len(tokens))

        # Curiosity
        ax1 = axes[idx, 0]
        ax1.plot(x, result["curiosity"], color="purple", linewidth=2)
        ax1.fill_between(x, result["curiosity"], alpha=0.3, color="purple")
        ax1.set_ylabel("Curiosity")
        ax1.set_title(f"Text {idx+1} - Curiosity")
        ax1.set_ylim(0, 1)

        # Uncertainty
        ax2 = axes[idx, 1]
        ax2.plot(x, result["uncertainty"], color="orange", linewidth=2)
        ax2.fill_between(x, result["uncertainty"], alpha=0.3, color="orange")
        ax2.set_ylabel("Uncertainty")
        ax2.set_title(f"Text {idx+1} - Uncertainty")
        ax2.set_ylim(0, 1)

        # Information Gain
        ax3 = axes[idx, 2]
        ax3.plot(x, result["info_gain"], color="green", linewidth=2)
        ax3.fill_between(x, result["info_gain"], alpha=0.3, color="green")
        ax3.set_ylabel("Expected Info Gain")
        ax3.set_title(f"Text {idx+1} - Information Gain")

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved curiosity analysis plot to {output_path}")
    plt.close()


def plot_combined_dashboard(all_results: Dict, text_idx: int = 0, output_path: str = "pem_dashboard.png"):
    """Create a combined dashboard for a single text."""

    # Get data from each stage
    pred_result = all_results["prediction"]["results"][text_idx]
    surp_result = all_results["surprise"]["results"][text_idx]
    val_result = all_results["valence"]["results"][text_idx]
    cur_result = all_results["curiosity"]["results"][text_idx]

    tokens = pred_result["tokens"]
    x = np.arange(len(tokens))
    text = pred_result["text"]

    fig = plt.figure(figsize=(16, 12))

    # Title
    fig.suptitle(f"PEM Analysis: \"{text[:80]}...\"", fontsize=12, fontweight="bold")

    # 1. Prediction Errors
    ax1 = fig.add_subplot(4, 1, 1)
    width = 0.25
    for i, (scale, color) in enumerate([("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]):
        ax1.bar(x + i*width, pred_result["errors"][scale], width, label=scale, alpha=0.7, color=color)
    ax1.set_ylabel("Prediction Error")
    ax1.set_title("1. Prediction Errors (how wrong was the prediction?)")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 2)

    # 2. Surprise
    ax2 = fig.add_subplot(4, 1, 2)
    for scale, color in [("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]:
        ax2.plot(x, surp_result["surprises"][scale]["magnitude"], label=scale, color=color, linewidth=2)
    ax2.set_ylabel("Surprise Magnitude")
    ax2.set_title("2. Surprise (learned surprise signal)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 1)

    # 3. Valence
    ax3 = fig.add_subplot(4, 1, 3)
    for scale, color in [("immediate", "blue"), ("shortterm", "green"), ("longterm", "red")]:
        ax3.plot(x, val_result["valences"][scale], label=scale, color=color, linewidth=2)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax3.set_ylabel("Valence")
    ax3.set_title("3. Valence (good vs bad for personality goals)")
    ax3.legend(loc="upper right")
    ax3.set_ylim(-1, 1)

    # 4. Curiosity & Uncertainty
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.plot(x, cur_result["curiosity"], label="Curiosity", color="purple", linewidth=2)
    ax4.plot(x, cur_result["uncertainty"], label="Uncertainty", color="orange", linewidth=2)
    ax4.plot(x, cur_result["info_gain"], label="Info Gain", color="green", linewidth=2)
    ax4.set_ylabel("Value")
    ax4.set_title("4. Curiosity, Uncertainty, Information Gain")
    ax4.legend(loc="upper right")

    # X-axis labels for bottom plot only
    ax4.set_xticks(x)
    ax4.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)

    # Remove x labels from upper plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved combined dashboard to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PEM Integration Test with Visualization")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="File with text to analyze")
    parser.add_argument("--output-dir", type=str, default="pem_analysis", help="Output directory for plots")
    parser.add_argument("--max-texts", type=int, default=3, help="Maximum number of texts to analyze")
    args = parser.parse_args()

    # Setup
    device = setup_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get texts
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            content = f.read()
            # Split into sentences/paragraphs
            texts = [t.strip() for t in content.split("\n\n") if t.strip()][:args.max_texts]
    else:
        texts = DEFAULT_TEXTS[:args.max_texts]

    logger.info(f"Analyzing {len(texts)} text(s)")

    # Run tests
    tester = IntegrationTester(device)
    all_results = tester.run_all_stages(texts)

    # Generate plots
    logger.info("\n" + "="*60)
    logger.info("Generating Visualizations")
    logger.info("="*60)

    plot_prediction_errors(
        all_results["prediction"],
        str(output_dir / "1_prediction_errors.png")
    )

    plot_surprise_over_text(
        all_results["surprise"],
        str(output_dir / "2_surprise_analysis.png")
    )

    plot_valence_analysis(
        all_results["valence"],
        str(output_dir / "3_valence_analysis.png")
    )

    plot_curiosity_analysis(
        all_results["curiosity"],
        str(output_dir / "4_curiosity_analysis.png")
    )

    # Combined dashboard for first text
    plot_combined_dashboard(
        all_results,
        text_idx=0,
        output_path=str(output_dir / "5_combined_dashboard.png")
    )

    logger.info(f"\nAll plots saved to {output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()

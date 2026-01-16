#!/usr/bin/env python3
"""
PredictionCTM Inference Demo

Showcases the PredictionCTM module capabilities with three visualizations:
1. Token Similarity Ranking - Shows predicted next tokens
2. Surprise Heatmap - Color-coded text by prediction surprise
3. Tick Evolution Plot - Shows CTM "thinking" process

Usage:
    python -m pem.inference_demo --checkpoint checkpoints/best.pt --text "The cat sat on the"

    # Or with a text file
    python -m pem.inference_demo --checkpoint checkpoints/best.pt --text_file input.txt

    # Stream through a book with live updates
    python -m pem.inference_demo --checkpoint checkpoints/best.pt --book /path/to/book.txt --pages 10

    # Save tick evolution plot
    python -m pem.inference_demo --checkpoint checkpoints/best.pt --text "..." --save_plot tick_evolution.png
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Background colors for surprise heatmap
    BG_GREEN = '\033[42m'      # Low surprise
    BG_YELLOW = '\033[43m'     # Medium surprise
    BG_RED = '\033[41m'        # High surprise
    BG_LIGHT_GREEN = '\033[102m'
    BG_LIGHT_YELLOW = '\033[103m'
    BG_LIGHT_RED = '\033[101m'

    # Foreground colors
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    WHITE = '\033[97m'
    BLACK = '\033[30m'


def get_surprise_color(surprise: float) -> str:
    """Get ANSI color code based on surprise level (0-1).

    Thresholds calibrated for typical model output:
    - <0.1: green (highly predictable, ~80% of tokens)
    - 0.1-0.2: yellow (mid-range, ~15% of tokens)
    - >=0.2: red (surprising, ~5% of tokens)
    """
    if surprise < 0.1:
        return Colors.BG_GREEN + Colors.BLACK
    elif surprise < 0.2:
        return Colors.BG_YELLOW + Colors.BLACK
    else:
        return Colors.BG_RED + Colors.WHITE


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print()
    print(Colors.BOLD + "=" * width + Colors.RESET)
    print(Colors.BOLD + title.center(width) + Colors.RESET)
    print(Colors.BOLD + "=" * width + Colors.RESET)
    print()


def print_section(title: str):
    """Print a section header."""
    width = 70
    print()
    print(Colors.CYAN + title + Colors.RESET)
    print(Colors.DIM + "-" * width + Colors.RESET)


def compute_targets(features: torch.Tensor, config) -> Dict[str, torch.Tensor]:
    """
    Compute prediction targets from features.

    For each position t, the target is the MEAN of features from t+1 to t+1+horizon.
    This matches how targets are computed during training.

    Args:
        features: (B, S, d_model) input features
        config: Configuration with horizon settings

    Returns:
        Dict with 'immediate', 'shortterm', 'longterm' targets and validity masks
    """
    from .prediction_module import PredictionTargets

    # Get horizons from config
    immediate_horizon = getattr(config, 'immediate_horizon', 8)
    shortterm_horizon = getattr(config, 'shortterm_horizon', 64)
    longterm_horizon = getattr(config, 'longterm_horizon', 256)

    # Use the same target computer as training
    target_computer = PredictionTargets(
        immediate_horizon=immediate_horizon,
        shortterm_horizon=shortterm_horizon,
        longterm_horizon=longterm_horizon,
    )

    return target_computer.compute_targets_efficient(features)


def compute_surprise(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """
    Compute surprise as 1 - cosine_similarity.

    Args:
        predictions: (B, S, d_model) predicted features
        targets: (B, S, d_model) target features
        valid: (B, S) validity mask

    Returns:
        surprise: (B, S) surprise values (0 = perfect prediction, 1 = orthogonal)
    """
    # Compute cosine similarity
    pred_norm = F.normalize(predictions, dim=-1)
    target_norm = F.normalize(targets, dim=-1)

    cos_sim = (pred_norm * target_norm).sum(dim=-1)  # (B, S)

    # Convert to surprise
    surprise = 1 - cos_sim

    # Mask invalid positions
    surprise = torch.where(valid, surprise, torch.zeros_like(surprise))

    return surprise


def print_surprise_heatmap(
    tokens: List[str],
    surprise_scores: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """
    Print tokens with colored backgrounds based on surprise.

    Args:
        tokens: List of token strings
        surprise_scores: (S,) surprise values
        valid_mask: (S,) validity mask
    """
    print_section("SURPRISE HEATMAP (green=predicted, red=surprising)")

    # Print colored tokens
    line = ""
    for i, token in enumerate(tokens):
        if i < len(surprise_scores) and valid_mask[i]:
            surprise = surprise_scores[i].item()
            color = get_surprise_color(surprise)
            line += f"{color}{token}{Colors.RESET}"
        else:
            line += Colors.DIM + token + Colors.RESET

    print(line)
    print()

    # Print surprise values below
    print(Colors.DIM + "Surprise values:" + Colors.RESET)
    values_line = ""
    for i, token in enumerate(tokens):
        if i < len(surprise_scores) and valid_mask[i]:
            surprise = surprise_scores[i].item()
            if surprise > 0.6:
                values_line += Colors.RED + f"{surprise:.2f}" + Colors.RESET + " "
            elif surprise > 0.3:
                values_line += Colors.YELLOW + f"{surprise:.2f}" + Colors.RESET + " "
            else:
                values_line += Colors.GREEN + f"{surprise:.2f}" + Colors.RESET + " "
        else:
            values_line += Colors.DIM + "----" + Colors.RESET + " "

    print(values_line)


def print_token_predictions(
    token_rankings: List[List[Tuple[str, float]]],
    tokens: List[str],
    actual_token_ids: List[int],
    tokenizer,
    surprise_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    max_positions: int = 5,
):
    """
    Print token predictions for each position.

    Args:
        token_rankings: List of (token, similarity) tuples per position
        tokens: Original token strings
        actual_token_ids: Actual next token IDs
        tokenizer: Tokenizer for decoding
        surprise_scores: Surprise values per position
        valid_mask: Validity mask
        max_positions: Maximum positions to show
    """
    print_section("TOKEN PREDICTIONS (immediate horizon)")

    num_to_show = min(max_positions, len(token_rankings))

    for pos in range(num_to_show):
        if not valid_mask[pos]:
            continue

        context_token = tokens[pos] if pos < len(tokens) else "?"
        predictions = token_rankings[pos]

        # Get actual next token
        if pos < len(actual_token_ids) - 1:
            actual_id = actual_token_ids[pos + 1]
            actual_token = tokenizer.decode([actual_id])
        else:
            actual_token = "N/A"
            actual_id = -1

        surprise = surprise_scores[pos].item() if pos < len(surprise_scores) else 0

        print(f"\nPosition {pos}: \"{context_token}\" -> Predicting next token...")

        # Check if actual is in predictions
        actual_in_predictions = False
        actual_rank = None
        for rank, (token, sim) in enumerate(predictions):
            marker = ""
            if token.strip() == actual_token.strip():
                actual_in_predictions = True
                actual_rank = rank + 1
                marker = Colors.GREEN + " *" + Colors.RESET

            print(f"  {rank + 1}. {repr(token):15s} ({sim:.3f}){marker}")

        # Show actual
        surprise_color = Colors.RED if surprise > 0.6 else (Colors.YELLOW if surprise > 0.3 else Colors.GREEN)
        if actual_in_predictions:
            print(f"  {Colors.GREEN}Actual: {repr(actual_token)} | Rank: {actual_rank} | Surprise: {surprise:.3f}{Colors.RESET}")
        else:
            print(f"  {surprise_color}Actual: {repr(actual_token)} | Rank: >5 | Surprise: {surprise:.3f}{Colors.RESET}")


def print_tick_evolution_bars(
    tick_similarities: List[float],
    certainty: float,
):
    """
    Print ASCII progress bars showing tick evolution.

    Args:
        tick_similarities: Cosine similarity at each tick
        certainty: Final certainty value
    """
    print_section("TICK EVOLUTION (prediction refinement)")

    bar_width = 40
    num_ticks = len(tick_similarities)

    for t, sim in enumerate(tick_similarities):
        # Create progress bar
        filled = int(sim * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Determine color based on similarity
        if sim > 0.7:
            color = Colors.GREEN
        elif sim > 0.4:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        # Mark final tick
        marker = ""
        if t == num_ticks - 1:
            marker = f"  <- Final (certainty: {certainty:.2f})"

        print(f"Tick {t}: {color}{bar}{Colors.RESET} {sim:.3f}{marker}")


def plot_tick_evolution(
    all_tick_outputs: List[torch.Tensor],
    targets: Dict[str, torch.Tensor],
    readout_immediate,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, List[float]]:
    """
    Create a plot showing how predictions improve over internal ticks.

    Args:
        all_tick_outputs: List of y_t tensors at each tick
        targets: Target dict with 'immediate' key
        readout_immediate: Readout head for immediate predictions
        save_path: Optional path to save the plot

    Returns:
        Figure and list of similarity values
    """
    tick_sims = []
    target = targets['immediate']  # (B, S, d_model)
    valid = targets['immediate_valid']  # (B, S)

    with torch.no_grad():
        for t, y_t in enumerate(all_tick_outputs):
            # Normalize and apply readout (matching training)
            y_t_norm = F.normalize(y_t, dim=-1)
            pred = readout_immediate(y_t_norm)

            # Compute cosine similarity for valid positions
            if valid.any():
                pred_valid = pred[valid]
                target_valid = target[valid]
                sim = F.cosine_similarity(pred_valid, target_valid, dim=-1).mean().item()
            else:
                sim = 0.0

            tick_sims.append(sim)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ticks = range(len(tick_sims))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tick_sims)))

    # Plot bars
    bars = ax.bar(ticks, tick_sims, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, sim in zip(bars, tick_sims):
        height = bar.get_height()
        ax.annotate(f'{sim:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Styling
    ax.set_xlabel('Internal Tick', fontsize=11)
    ax.set_ylabel('Cosine Similarity to Target', fontsize=11)
    ax.set_title('Prediction Refinement Over Thinking Ticks', fontsize=13, fontweight='bold')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f't{t}' for t in ticks])
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add trend line
    if len(tick_sims) > 1:
        z = np.polyfit(range(len(tick_sims)), tick_sims, 1)
        p = np.poly1d(z)
        ax.plot(ticks, p(ticks), '--', color='red', alpha=0.7, label=f'Trend (slope: {z[0]:.3f})')
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved tick evolution plot to: {save_path}")

    return fig, tick_sims


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load PredictionCTM model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config, feature_extractor)
    """
    import sys
    from .prediction_ctm import PredictionCTM, PredictionCTMConfig
    from .janus_pro_feature_extractor import JanusProFeatureExtractor, JanusProConfig, LearningMode

    # Import TrainingConfig and add to __main__ for unpickling
    # (checkpoint was saved from __main__ context)
    from .train_prediction import TrainingConfig
    sys.modules['__main__'].TrainingConfig = TrainingConfig

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint - handle both TrainingConfig and dict
    if 'config' in checkpoint:
        config_obj = checkpoint['config']
        # If it's a TrainingConfig, extract the relevant fields
        if hasattr(config_obj, '__dataclass_fields__'):
            config_dict = {k: getattr(config_obj, k) for k in config_obj.__dataclass_fields__}
        else:
            config_dict = config_obj

        # Map TrainingConfig field names to PredictionCTMConfig field names
        field_mapping = {
            'feature_dim': 'd_input',
            'd_sync_action': 'd_sync_internal',
        }
        mapped_dict = {}
        for k, v in config_dict.items():
            mapped_key = field_mapping.get(k, k)
            mapped_dict[mapped_key] = v

        # Also set d_output to match d_input
        if 'd_input' in mapped_dict and 'd_output' not in mapped_dict:
            mapped_dict['d_output'] = mapped_dict['d_input']

        config = PredictionCTMConfig(**{k: v for k, v in mapped_dict.items()
                                        if k in PredictionCTMConfig.__dataclass_fields__})
    else:
        # Default config
        config = PredictionCTMConfig()

    # Create model
    model = PredictionCTM(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model with config: T={config.T}, d_neurons={config.d_neurons}")

    # Load feature extractor
    logger.info("Loading feature extractor...")
    janus_config = JanusProConfig(
        model_name_or_path="deepseek-ai/Janus-Pro-1B",
        output_dim=config.d_input,
        learning_mode=LearningMode.FROZEN,
    )
    feature_extractor = JanusProFeatureExtractor(janus_config)

    # Load feature extractor projection if saved in checkpoint
    if "fe_projection_state" in checkpoint and checkpoint["fe_projection_state"] is not None:
        # Ensure feature extractor is loaded (creates projection layer)
        feature_extractor._ensure_loaded()
        if feature_extractor.projection is not None:
            feature_extractor.projection.load_state_dict(checkpoint["fe_projection_state"])
            logger.info("Loaded feature extractor projection from checkpoint")
    else:
        logger.warning("No feature extractor projection in checkpoint - predictions may be inaccurate!")
        logger.warning("Re-train with updated training script to fix this.")

    feature_extractor.eval()

    return model, config, feature_extractor


def run_inference(
    text: str,
    model,
    feature_extractor,
    device: str = 'cuda',
):
    """
    Run inference on input text.

    Args:
        text: Input text
        model: PredictionCTM model
        feature_extractor: Feature extractor
        device: Device

    Returns:
        Dict with all inference outputs
    """
    # Tokenize
    tokenizer = feature_extractor.tokenizer
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Decode tokens for display
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_ids, attention_mask=attention_mask)
        features = features.float()  # Cast to float32

    # Run prediction model
    with torch.no_grad():
        output = model(features)

    # Compute targets
    targets = compute_targets(features, model.prediction_config)

    return {
        'input_ids': input_ids,
        'tokens': tokens,
        'features': features,
        'output': output,
        'targets': targets,
        'tokenizer': tokenizer,
    }


def main():
    parser = argparse.ArgumentParser(
        description='PredictionCTM Inference Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                        help='Input text to process')
    parser.add_argument('--text_file', type=str, default=None,
                        help='Path to file containing input text')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save tick evolution plot')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predicted tokens to show')
    parser.add_argument('--max_positions', type=int, default=5,
                        help='Maximum positions to show for token predictions')
    parser.add_argument('--vocab_cache', type=str, default=None,
                        help='Path to cache vocabulary embeddings')
    parser.add_argument('--skip_vocab', action='store_true',
                        help='Skip vocabulary index building (faster but no token predictions)')
    parser.add_argument('--rebuild_vocab', action='store_true',
                        help='Force rebuild vocab cache (needed if checkpoint changed)')

    # Book streaming mode
    parser.add_argument('--book', type=str, default=None,
                        help='Path to book file for streaming analysis')
    parser.add_argument('--pages', type=int, default=10,
                        help='Number of pages to process (for --book mode)')
    parser.add_argument('--chars_per_page', type=int, default=2000,
                        help='Characters per page (for --book mode)')
    parser.add_argument('--update_interval', type=float, default=0.5,
                        help='Seconds between display updates (for --book mode)')
    parser.add_argument('--text_only', action='store_true',
                        help='Show only colored text output without statistics panels')

    args = parser.parse_args()

    # Check for book mode
    if args.book:
        # Book streaming mode
        print_header("PredictionCTM Book Analysis")
        print(f"Book: {args.book}")
        print(f"Pages to process: {args.pages}")
        print(f"Device: {args.device}")
        print()

        # Load model
        try:
            model, config, feature_extractor = load_model(args.checkpoint, args.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

        # Build vocab index if not skipped
        vocab_index = None
        if not args.skip_vocab:
            from .vocab_index import VocabEmbeddingIndex
            # Delete cache if rebuild requested
            if args.rebuild_vocab and args.vocab_cache:
                cache_path = Path(args.vocab_cache)
                if cache_path.exists():
                    logger.info(f"Removing old vocab cache: {cache_path}")
                    cache_path.unlink()
            logger.info("Building vocabulary embedding index...")
            vocab_index = VocabEmbeddingIndex(
                feature_extractor,
                device=args.device,
                cache_path=args.vocab_cache,
            )

        # Run book streaming
        run_book_streaming(
            book_path=args.book,
            model=model,
            feature_extractor=feature_extractor,
            vocab_index=vocab_index,
            device=args.device,
            max_pages=args.pages,
            chars_per_page=args.chars_per_page,
            update_interval=args.update_interval,
            text_only=args.text_only,
        )
        return

    # Single text mode (original behavior)
    # Get input text
    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            text = f.read()
    else:
        # Default demo text
        text = "The cat sat on the mat. It was a warm sunny day."

    # Print header
    print_header("PredictionCTM Inference Demo")

    print(f"Input text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
    print(f"Device: {args.device}")
    print()

    # Load model
    try:
        model, config, feature_extractor = load_model(args.checkpoint, args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Running in demo mode with mock outputs...")
        # Demo mode - show what the output would look like
        print("\n[Demo mode - showing example output format]\n")
        demo_mode(text, args)
        return

    # Run inference
    logger.info("Running inference...")
    results = run_inference(text, model, feature_extractor, args.device)

    # Build vocab index if not skipped
    vocab_index = None
    if not args.skip_vocab:
        from .vocab_index import VocabEmbeddingIndex
        # Delete cache if rebuild requested
        if args.rebuild_vocab and args.vocab_cache:
            cache_path = Path(args.vocab_cache)
            if cache_path.exists():
                logger.info(f"Removing old vocab cache: {cache_path}")
                cache_path.unlink()
        logger.info("Building vocabulary embedding index...")
        vocab_index = VocabEmbeddingIndex(
            feature_extractor,
            device=args.device,
            cache_path=args.vocab_cache,
        )

    output = results['output']
    targets = results['targets']
    tokens = results['tokens']
    tokenizer = results['tokenizer']
    input_ids = results['input_ids']

    # === Compute Surprise ===
    surprise_scores = compute_surprise(
        output.predictions['immediate'],
        targets['immediate'],
        targets['immediate_valid'],
    )[0]  # Take first (only) batch item

    # === Print Surprise Heatmap ===
    print_surprise_heatmap(
        tokens,
        surprise_scores,
        targets['immediate_valid'][0],
    )

    # === Print Token Predictions ===
    if vocab_index is not None:
        logger.info("Computing token similarity rankings...")
        token_rankings = vocab_index.top_k_for_sequence(
            output.predictions['immediate'][0],
            k=args.top_k,
        )
        print_token_predictions(
            token_rankings,
            tokens,
            input_ids[0].tolist(),
            tokenizer,
            surprise_scores,
            targets['immediate_valid'][0],
            max_positions=args.max_positions,
        )
    else:
        print_section("TOKEN PREDICTIONS (skipped - use --vocab_cache to enable)")

    # === Plot Tick Evolution ===
    if output.all_tick_outputs:
        fig, tick_sims = plot_tick_evolution(
            output.all_tick_outputs,
            targets,
            model.readout_immediate,
            save_path=args.save_plot,
        )
        plt.close(fig)

        # Print ASCII version
        print_tick_evolution_bars(tick_sims, output.certainty.item())
    else:
        print_section("TICK EVOLUTION (no tick outputs available)")

    # === Summary Statistics ===
    print_section("SUMMARY")

    valid_mask = targets['immediate_valid'][0]
    if valid_mask.any():
        mean_surprise = surprise_scores[valid_mask].mean().item()
        max_surprise = surprise_scores[valid_mask].max().item()
        min_surprise = surprise_scores[valid_mask].min().item()
    else:
        mean_surprise = max_surprise = min_surprise = 0.0

    print(f"Total tokens: {len(tokens)}")
    print(f"Valid positions: {valid_mask.sum().item()}")
    print(f"Mean surprise: {mean_surprise:.3f}")
    print(f"Max surprise: {max_surprise:.3f} (most unexpected)")
    print(f"Min surprise: {min_surprise:.3f} (most predictable)")
    print(f"Final certainty: {output.certainty.item():.3f}")

    if output.all_tick_outputs and len(tick_sims) > 1:
        improvement = tick_sims[-1] - tick_sims[0]
        print(f"Tick improvement: {improvement:+.3f} (from t0 to t{len(tick_sims)-1})")

    if args.save_plot:
        print(f"\nPlot saved to: {args.save_plot}")


# =============================================================================
# STREAMING BOOK MODE
# =============================================================================

@dataclass
class BookStats:
    """Running statistics for book processing."""
    chunks_processed: int = 0
    total_tokens: int = 0
    total_valid_positions: int = 0

    # Surprise statistics
    surprise_sum: float = 0.0
    surprise_count: int = 0
    surprise_min: float = float('inf')
    surprise_max: float = float('-inf')
    surprise_histogram: List[int] = field(default_factory=lambda: [0] * 10)  # 10 bins from 0-1

    # Prediction accuracy (top-k hits from vocab lookup)
    top1_hits: int = 0
    top5_hits: int = 0
    total_predictions: int = 0

    # Direct next-token feature similarity (compares prediction to actual features[t+1])
    # This is more meaningful than vocab lookup since embeddings are contextual
    next_token_sim_sum: float = 0.0
    next_token_sim_count: int = 0

    # Tick evolution
    tick_improvements: List[float] = field(default_factory=list)

    # Tokens by surprise level
    most_surprising: List[Tuple[str, float, str]] = field(default_factory=list)  # (token, surprise, context)
    mid_surprising: List[Tuple[str, float, str]] = field(default_factory=list)   # mid-range (0.1-0.2)
    least_surprising: List[Tuple[str, float, str]] = field(default_factory=list) # most predictable

    @property
    def mean_next_token_sim(self) -> float:
        return self.next_token_sim_sum / self.next_token_sim_count if self.next_token_sim_count > 0 else 0.0

    def update_surprise(self, surprise_scores: torch.Tensor, tokens: List[str], valid_mask: torch.Tensor):
        """Update surprise statistics."""
        for i, (surprise, valid) in enumerate(zip(surprise_scores.tolist(), valid_mask.tolist())):
            if not valid:
                continue

            self.surprise_sum += surprise
            self.surprise_count += 1
            self.surprise_min = min(self.surprise_min, surprise)
            self.surprise_max = max(self.surprise_max, surprise)

            # Update histogram
            bin_idx = min(int(surprise * 10), 9)
            self.surprise_histogram[bin_idx] += 1

            # Track tokens by surprise level
            if i < len(tokens):
                context_start = max(0, i - 3)
                context_end = min(len(tokens), i + 4)
                context = ''.join(tokens[context_start:context_end])
                entry = (tokens[i], surprise, context)

                # Categorize by surprise level
                if surprise >= 0.2:
                    self.most_surprising.append(entry)
                elif surprise >= 0.1:
                    self.mid_surprising.append(entry)
                else:
                    self.least_surprising.append(entry)

        # Keep top entries for each category
        self.most_surprising.sort(key=lambda x: x[1], reverse=True)
        self.most_surprising = self.most_surprising[:20]

        # For mid-range, keep a representative sample (sorted by surprise)
        self.mid_surprising.sort(key=lambda x: x[1], reverse=True)
        self.mid_surprising = self.mid_surprising[:20]

        # For least surprising, keep the most predictable ones
        self.least_surprising.sort(key=lambda x: x[1])
        self.least_surprising = self.least_surprising[:20]

    @property
    def mean_surprise(self) -> float:
        return self.surprise_sum / self.surprise_count if self.surprise_count > 0 else 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_hits / self.total_predictions if self.total_predictions > 0 else 0.0

    @property
    def top5_accuracy(self) -> float:
        return self.top5_hits / self.total_predictions if self.total_predictions > 0 else 0.0


def estimate_pages(text: str, chars_per_page: int = 2000) -> int:
    """Estimate number of pages in text."""
    return max(1, len(text) // chars_per_page)


def get_page_chunk(text: str, page: int, chars_per_page: int = 2000) -> str:
    """Get a specific page from text."""
    start = page * chars_per_page
    end = start + chars_per_page
    return text[start:end]


def create_live_display(stats: BookStats, current_chunk: str, current_tokens: List[str],
                        current_surprise: torch.Tensor, valid_mask: torch.Tensor,
                        top_predictions: Optional[List[List[Tuple[str, float]]]] = None,
                        tick_sims: Optional[List[float]] = None):
    """Create rich display panels for live update."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.layout import Layout
        from rich.text import Text
        from rich.progress import Progress, BarColumn, TextColumn
    except ImportError:
        return None

    console = Console()

    # Build surprise heatmap text
    # Thresholds: <0.1 green, 0.1-0.2 yellow, >=0.2 red
    heatmap_text = Text()
    for i, token in enumerate(current_tokens[:50]):  # Show first 50 tokens
        if i < len(current_surprise) and i < len(valid_mask) and valid_mask[i]:
            surprise = current_surprise[i].item()
            if surprise < 0.1:
                heatmap_text.append(token, style="black on green")
            elif surprise < 0.2:
                heatmap_text.append(token, style="black on yellow")
            else:
                heatmap_text.append(token, style="white on red")
        else:
            heatmap_text.append(token, style="dim")

    if len(current_tokens) > 50:
        heatmap_text.append("...", style="dim")

    # Build stats table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Pages processed", f"{stats.chunks_processed}")
    stats_table.add_row("Total tokens", f"{stats.total_tokens:,}")
    stats_table.add_row("Valid positions", f"{stats.total_valid_positions:,}")
    stats_table.add_row("", "")
    stats_table.add_row("Mean surprise", f"{stats.mean_surprise:.3f}")
    stats_table.add_row("Min surprise", f"{stats.surprise_min:.3f}" if stats.surprise_min != float('inf') else "N/A")
    stats_table.add_row("Max surprise", f"{stats.surprise_max:.3f}" if stats.surprise_max != float('-inf') else "N/A")

    # Next-token similarity (direct feature comparison, more meaningful)
    if stats.next_token_sim_count > 0:
        stats_table.add_row("", "")
        stats_table.add_row("Next-token sim", f"{stats.mean_next_token_sim:.3f}")

    # Vocab lookup accuracy (often low due to contextual embeddings)
    if stats.total_predictions > 0:
        stats_table.add_row("Top-1 accuracy", f"{stats.top1_accuracy:.1%}")
        stats_table.add_row("Top-5 accuracy", f"{stats.top5_accuracy:.1%}")

    # Build histogram
    # Histogram colors match heatmap: <0.1 green, 0.1-0.2 yellow, >=0.2 red
    histogram_text = Text()
    max_count = max(stats.surprise_histogram) if any(stats.surprise_histogram) else 1
    for i, count in enumerate(stats.surprise_histogram):
        bar_len = int(count / max_count * 20) if max_count > 0 else 0
        label = f"{i/10:.1f}-{(i+1)/10:.1f}"
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if i < 1:  # 0.0-0.1
            histogram_text.append(f"{label}: {bar} {count}\n", style="green")
        elif i < 2:  # 0.1-0.2
            histogram_text.append(f"{label}: {bar} {count}\n", style="yellow")
        else:  # 0.2+
            histogram_text.append(f"{label}: {bar} {count}\n", style="red")

    # Build tick evolution if available
    tick_text = Text()
    if tick_sims:
        for t, sim in enumerate(tick_sims):
            bar_len = int(sim * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            if sim > 0.7:
                tick_text.append(f"t{t}: {bar} {sim:.3f}\n", style="green")
            elif sim > 0.4:
                tick_text.append(f"t{t}: {bar} {sim:.3f}\n", style="yellow")
            else:
                tick_text.append(f"t{t}: {bar} {sim:.3f}\n", style="red")

    # Build most surprising tokens
    surprising_text = Text()
    for token, surprise, context in stats.most_surprising[:10]:
        surprising_text.append(f"{surprise:.3f} ", style="red bold")
        surprising_text.append(f"{repr(token):12s} ", style="white")
        surprising_text.append(f"...{context}...\n", style="dim")

    # Build token predictions display
    predictions_text = Text()
    if top_predictions and len(current_tokens) > 1:
        # Show predictions for first 5 valid positions
        shown = 0
        for pos in range(min(len(top_predictions), len(current_tokens) - 1)):
            if pos < len(valid_mask) and valid_mask[pos] and shown < 5:
                context_token = current_tokens[pos]
                actual_token = current_tokens[pos + 1] if pos + 1 < len(current_tokens) else "?"
                preds = top_predictions[pos][:3]  # Top 3 predictions

                predictions_text.append(f"{repr(context_token):10s} -> ", style="cyan")
                for i, (pred_tok, sim) in enumerate(preds):
                    style = "green bold" if pred_tok.strip() == actual_token.strip() else "white"
                    predictions_text.append(f"{repr(pred_tok)}", style=style)
                    predictions_text.append(f"({sim:.2f})", style="dim")
                    if i < len(preds) - 1:
                        predictions_text.append(" | ", style="dim")
                predictions_text.append(f"  [actual: {repr(actual_token)}]\n", style="yellow")
                shown += 1

    return {
        'heatmap': heatmap_text,
        'stats': stats_table,
        'histogram': histogram_text,
        'ticks': tick_text,
        'surprising': surprising_text,
        'predictions': predictions_text,
    }


def run_book_streaming(
    book_path: str,
    model,
    feature_extractor,
    vocab_index,
    device: str = 'cuda',
    max_pages: int = 10,
    chars_per_page: int = 2000,
    update_interval: float = 0.5,
    text_only: bool = False,
):
    """
    Stream through a book with live updating display.

    Args:
        book_path: Path to book text file
        model: PredictionCTM model
        feature_extractor: Feature extractor
        vocab_index: Optional vocabulary index for token predictions
        device: Device to run on
        max_pages: Maximum pages to process
        chars_per_page: Characters per page
        update_interval: Seconds between display updates
        text_only: If True, show only colored text without statistics panels
    """
    # Read book
    with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
        book_text = f.read()

    total_pages = estimate_pages(book_text, chars_per_page)
    pages_to_process = min(max_pages, total_pages)

    logger.info(f"Book: {book_path}")
    logger.info(f"Total estimated pages: {total_pages}, processing: {pages_to_process}")

    # Initialize stats
    stats = BookStats()
    tokenizer = feature_extractor.tokenizer

    # Try to use rich for live display
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.layout import Layout
        from rich.table import Table
        use_rich = True
        console = Console()
    except ImportError:
        use_rich = False
        logger.warning("Install 'rich' for live display: pip install rich")

    def process_page(page_num: int) -> dict:
        """Process a single page and return results."""
        chunk = get_page_chunk(book_text, page_num, chars_per_page)
        if not chunk.strip():
            return None

        # Tokenize
        tokenized = tokenizer(
            chunk,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        input_ids = tokenized["input_ids"].to(device)

        # Get tokens for display
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Extract features and run model
        with torch.no_grad():
            features = feature_extractor(input_ids)
            features = features.float()
            output = model(features)

        # Compute targets and surprise
        targets = compute_targets(features, model.prediction_config)
        surprise_scores = compute_surprise(
            output.predictions['immediate'],
            targets['immediate'],
            targets['immediate_valid'],
        )[0]

        valid_mask = targets['immediate_valid'][0]

        # Compute tick evolution
        tick_sims = []
        if output.all_tick_outputs:
            for y_t in output.all_tick_outputs:
                y_t_norm = F.normalize(y_t, dim=-1)
                pred = model.readout_immediate(y_t_norm)
                if valid_mask.any():
                    sim = F.cosine_similarity(
                        pred[0][valid_mask],
                        targets['immediate'][0][valid_mask],
                        dim=-1
                    ).mean().item()
                else:
                    sim = 0.0
                tick_sims.append(sim)

        # Compute direct next-token similarity (prediction vs actual features[t+1])
        # This is more meaningful than vocab lookup because LLM embeddings are contextual
        next_token_sims = []
        predictions = output.predictions['immediate'][0]  # (S, d_model)
        seq_len = features.shape[1]
        for pos in range(seq_len - 1):  # Can't predict beyond last position
            pred_norm = F.normalize(predictions[pos], dim=-1)
            next_feat_norm = F.normalize(features[0, pos + 1], dim=-1)
            sim = F.cosine_similarity(pred_norm.unsqueeze(0), next_feat_norm.unsqueeze(0)).item()
            next_token_sims.append(sim)
            stats.next_token_sim_sum += sim
            stats.next_token_sim_count += 1

        # Get token predictions if vocab index available
        top_predictions = None
        if vocab_index is not None:
            # NOTE: The model predicts mean(features[t+1:t+1+horizon]) where horizon=8 by default
            # This means predictions won't match single-token embeddings exactly
            # For better token accuracy, retrain with immediate_horizon=1
            top_predictions = vocab_index.top_k_for_sequence(
                output.predictions['immediate'][0],
                k=5,
            )

            # Check prediction accuracy against actual next token
            for pos in range(min(len(top_predictions), len(input_ids[0]) - 1)):
                if pos < len(valid_mask) and valid_mask[pos]:
                    actual_id = input_ids[0, pos + 1].item()
                    actual_token = tokenizer.decode([actual_id]).strip()
                    predicted_tokens = [t[0].strip() for t in top_predictions[pos]]

                    stats.total_predictions += 1
                    if predicted_tokens and predicted_tokens[0] == actual_token:
                        stats.top1_hits += 1
                    if actual_token in predicted_tokens:
                        stats.top5_hits += 1

        return {
            'tokens': tokens,
            'surprise': surprise_scores,
            'valid_mask': valid_mask,
            'tick_sims': tick_sims,
            'top_predictions': top_predictions,
            'chunk': chunk,
            'next_token_sims': next_token_sims,
        }

    def render_display(stats: BookStats, page_results: dict, page_num: int, pages_to_process: int) -> str:
        """Render display without rich."""
        lines = []
        lines.append("\033[2J\033[H")  # Clear screen
        lines.append("=" * 70)
        lines.append(f"  PredictionCTM Book Analysis - Page {page_num + 1}/{pages_to_process}")
        lines.append("=" * 70)
        lines.append("")

        # Stats
        lines.append(f"Tokens: {stats.total_tokens:,}  |  Valid: {stats.total_valid_positions:,}  |  Mean surprise: {stats.mean_surprise:.3f}")

        if stats.next_token_sim_count > 0:
            lines.append(f"Next-token similarity: {stats.mean_next_token_sim:.3f}")

        if stats.total_predictions > 0:
            lines.append(f"Vocab lookup - Top-1: {stats.top1_accuracy:.1%}  |  Top-5: {stats.top5_accuracy:.1%}")

        lines.append("")

        # Current chunk surprise heatmap (simplified)
        lines.append("Current chunk (first 50 tokens):")
        tokens = page_results['tokens'][:50]
        surprise = page_results['surprise']
        valid = page_results['valid_mask']

        heatmap_line = ""
        for i, token in enumerate(tokens):
            if i < len(surprise) and i < len(valid) and valid[i]:
                s = surprise[i].item()
                if s < 0.1:
                    heatmap_line += f"\033[42m\033[30m{token}\033[0m"
                elif s < 0.2:
                    heatmap_line += f"\033[43m\033[30m{token}\033[0m"
                else:
                    heatmap_line += f"\033[41m\033[97m{token}\033[0m"
            else:
                heatmap_line += f"\033[2m{token}\033[0m"

        lines.append(heatmap_line)
        lines.append("")

        # Histogram (colors: <0.1 green, 0.1-0.2 yellow, >=0.2 red)
        lines.append("Surprise distribution:")
        max_count = max(stats.surprise_histogram) if any(stats.surprise_histogram) else 1
        for i, count in enumerate(stats.surprise_histogram):
            bar_len = int(count / max_count * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            color = "\033[32m" if i < 1 else ("\033[33m" if i < 2 else "\033[31m")
            lines.append(f"  {i/10:.1f}-{(i+1)/10:.1f}: {color}{bar}\033[0m {count}")

        lines.append("")

        # Top surprising tokens
        if stats.most_surprising:
            lines.append("Most surprising tokens:")
            for token, s, ctx in stats.most_surprising[:5]:
                lines.append(f"  \033[31m{s:.3f}\033[0m {repr(token):12s} \033[2m{ctx[:40]}\033[0m")

        return "\n".join(lines)

    # Main processing loop
    if text_only:
        # Text-only mode: just print colored text
        print("\n" + "=" * 70)
        print("  COLORED TEXT OUTPUT (green=predictable, yellow=mid, red=surprising)")
        print("=" * 70 + "\n")

        for page_num in range(pages_to_process):
            result = process_page(page_num)
            if result is None:
                continue

            # Update stats
            stats.chunks_processed += 1
            stats.total_tokens += len(result['tokens'])
            stats.total_valid_positions += result['valid_mask'].sum().item()
            stats.update_surprise(result['surprise'], result['tokens'], result['valid_mask'])

            if result['tick_sims'] and len(result['tick_sims']) > 1:
                stats.tick_improvements.append(result['tick_sims'][-1] - result['tick_sims'][0])

            # Print colored text for all tokens in this chunk
            tokens = result['tokens']
            surprise = result['surprise']
            valid = result['valid_mask']

            for i, token in enumerate(tokens):
                if i < len(surprise) and i < len(valid) and valid[i]:
                    s = surprise[i].item()
                    if s < 0.1:
                        print(f"\033[42m\033[30m{token}\033[0m", end="")
                    elif s < 0.2:
                        print(f"\033[43m\033[30m{token}\033[0m", end="")
                    else:
                        print(f"\033[41m\033[97m{token}\033[0m", end="")
                else:
                    print(f"\033[2m{token}\033[0m", end="")

            # No delay in text-only mode - just continuous output
        print("\n")  # Final newline

    elif use_rich:
        with Live(console=console, refresh_per_second=2) as live:
            for page_num in range(pages_to_process):
                result = process_page(page_num)
                if result is None:
                    continue

                # Update stats
                stats.chunks_processed += 1
                stats.total_tokens += len(result['tokens'])
                stats.total_valid_positions += result['valid_mask'].sum().item()
                stats.update_surprise(result['surprise'], result['tokens'], result['valid_mask'])

                if result['tick_sims'] and len(result['tick_sims']) > 1:
                    stats.tick_improvements.append(result['tick_sims'][-1] - result['tick_sims'][0])

                # Create display
                display = create_live_display(
                    stats, result['chunk'], result['tokens'],
                    result['surprise'], result['valid_mask'],
                    result['top_predictions'], result['tick_sims']
                )

                if display:
                    # Build layout
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel(f"Page {page_num + 1}/{pages_to_process}", title="Progress"), size=3),
                        Layout(name="main"),
                    )
                    layout["main"].split_row(
                        Layout(Panel(display['stats'], title="Statistics"), ratio=1),
                        Layout(name="right", ratio=2),
                    )
                    layout["right"].split_column(
                        Layout(Panel(display['heatmap'], title="Current Chunk Surprise"), ratio=1),
                        Layout(name="middle", ratio=1),
                        Layout(Panel(display['histogram'], title="Surprise Distribution"), ratio=2),
                    )
                    # Add predictions panel if available
                    if display.get('predictions'):
                        layout["middle"].update(Panel(display['predictions'], title="Token Predictions (top-3)"))
                    else:
                        layout["middle"].update(Panel("No predictions available", title="Token Predictions"))

                    live.update(layout)

                time.sleep(update_interval)
    else:
        # Fallback without rich
        for page_num in range(pages_to_process):
            result = process_page(page_num)
            if result is None:
                continue

            # Update stats
            stats.chunks_processed += 1
            stats.total_tokens += len(result['tokens'])
            stats.total_valid_positions += result['valid_mask'].sum().item()
            stats.update_surprise(result['surprise'], result['tokens'], result['valid_mask'])

            # Print simple display
            display = render_display(stats, result, page_num, pages_to_process)
            print(display)

            time.sleep(update_interval)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\nPages processed: {stats.chunks_processed}")
    print(f"Total tokens: {stats.total_tokens:,}")
    print(f"Valid positions: {stats.total_valid_positions:,}")
    print(f"\nMean surprise: {stats.mean_surprise:.3f}")
    print(f"Min surprise: {stats.surprise_min:.3f}")
    print(f"Max surprise: {stats.surprise_max:.3f}")

    # Next-token similarity is the key metric for token prediction quality
    if stats.next_token_sim_count > 0:
        print(f"\nNext-token similarity: {stats.mean_next_token_sim:.3f}")
        print("  (cosine similarity between prediction and actual next token embedding)")

    # Vocab lookup accuracy (typically low due to contextual embedding mismatch)
    if stats.total_predictions > 0:
        print(f"\nVocab lookup accuracy:")
        print(f"  Top-1: {stats.top1_accuracy:.1%}")
        print(f"  Top-5: {stats.top5_accuracy:.1%}")
        print("  (low accuracy expected: model predicts mean of next 8 tokens,")
        print("   and vocab embeddings are non-contextual)")

    if stats.tick_improvements:
        print(f"\nMean tick improvement: {np.mean(stats.tick_improvements):.4f}")

    # Show tokens across surprise spectrum
    print("\n" + "-" * 70)
    print("TOKENS BY SURPRISE LEVEL")
    print("-" * 70)

    print("\nMost surprising (>0.2) - hard to predict:")
    for token, surprise, context in stats.most_surprising[:10]:
        print(f"  {surprise:.3f} {repr(token):15s} ...{context[:50]}...")

    print("\nMid-range (0.1-0.2) - moderately predictable:")
    for token, surprise, context in stats.mid_surprising[:10]:
        print(f"  {surprise:.3f} {repr(token):15s} ...{context[:50]}...")

    print("\nLeast surprising (<0.1) - highly predictable:")
    for token, surprise, context in stats.least_surprising[:10]:
        print(f"  {surprise:.3f} {repr(token):15s} ...{context[:50]}...")

    return stats


def demo_mode(text: str, args):
    """Show demo output when model can't be loaded."""
    # Simulate tokens
    tokens = text.split()

    # Demo surprise scores
    np.random.seed(42)
    surprise_scores = np.random.uniform(0.1, 0.9, len(tokens))
    surprise_scores = torch.tensor(surprise_scores)
    valid_mask = torch.ones(len(tokens), dtype=torch.bool)

    print_surprise_heatmap(tokens, surprise_scores, valid_mask)

    # Demo token predictions
    print_section("TOKEN PREDICTIONS (demo mode)")
    demo_predictions = [
        ("mat", 0.87), ("floor", 0.82), ("couch", 0.79), ("bed", 0.76), ("ground", 0.74)
    ]
    for rank, (token, sim) in enumerate(demo_predictions):
        print(f"  {rank + 1}. {repr(token):15s} ({sim:.3f})")
    print(f"  {Colors.GREEN}Actual: 'mat' | Rank: 1 | Surprise: 0.13{Colors.RESET}")

    # Demo tick evolution
    tick_sims = [0.42, 0.68, 0.79, 0.84, 0.89]
    print_tick_evolution_bars(tick_sims, 0.91)

    print_section("SUMMARY (demo mode)")
    print("This is a demonstration of the output format.")
    print("To see real predictions, provide a valid checkpoint with --checkpoint")


if __name__ == '__main__':
    main()

"""
CTM Loss Functions with Tick Selection.

Implements loss computation strategies that leverage CTM's multiple tick outputs
to select the best prediction at each position.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMLoss(nn.Module):
    """
    Loss computation with tick selection for CTM.

    Strategies:
    - "all": Average loss across all ticks (DEFAULT - most faithful to CTM paper)
    - "min_loss": Select tick with minimum loss per position (can discourage exploration)
    - "max_certainty": Select tick with highest confidence
    - "weighted": Soft attention over ticks based on inverse loss
    - "last": Always use final tick (for warmup/debugging)

    Per the CTM paper, the model should be allowed to develop rich dynamics across
    ticks without being pressured to produce correct answers early. The "all"
    strategy treats all ticks equally, allowing emergent behaviors to develop
    naturally without explicit bias toward early stopping.

    The loss is computed per position across the sequence, respecting
    the autoregressive language modeling objective.

    Note: This module assumes labels are ALREADY shifted (i.e., labels[i] is the
    target for logits[i]). The dataset should provide input_ids = tokens[:-1]
    and labels = tokens[1:].
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = -100,
        ignore_index: int = -100,
        selection: str = "all",  # Changed from "min_loss" - more faithful to CTM
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            pad_token_id: Token ID used for padding (used to create mask for metrics)
            ignore_index: Index to ignore in loss computation (default -100).
                          Set to -100 to include all tokens including EOS in loss.
            selection: Tick selection strategy
            tau: Temperature for soft selection (used in "weighted" mode)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.selection = selection
        self.tau = tau
        self.label_smoothing = label_smoothing

        # Base criterion with no reduction for per-position losses
        # Use ignore_index (default -100) to control what's excluded from loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='none',
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        all_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss with tick selection.

        Args:
            all_logits: List of (B, S, V) logits, one per tick
            labels: (B, S) ground truth token IDs (already shifted by dataset)

        Returns:
            loss: Scalar loss for backpropagation
            metrics: Dict with per-tick losses, selected ticks, etc.
        """
        B, S = labels.shape
        T = len(all_logits)
        V = all_logits[0].size(-1)
        device = labels.device

        # Compute loss for each tick: (T, B, S)
        # NO shifting here - dataset already provides shifted labels
        tick_losses = []
        for logits in all_logits:
            loss_per_pos = self.criterion(
                logits.view(-1, V),
                labels.view(-1)
            ).view(B, S)
            tick_losses.append(loss_per_pos)

        tick_losses = torch.stack(tick_losses, dim=0)  # (T, B, S)

        # Create mask for non-padding positions (for metrics and loss aggregation)
        # Use ignore_index if set, otherwise use pad_token_id
        mask_token = self.ignore_index if self.ignore_index != -100 else self.pad_token_id
        if mask_token == -100:
            # No masking - all positions are valid
            mask = torch.ones(B, S, device=device)
        else:
            mask = (labels != mask_token).float()  # (B, S)
        num_valid = mask.sum()

        if self.selection == "min_loss":
            # Select tick with minimum loss per position
            # Set padded positions to inf so they don't affect selection
            masked_losses = tick_losses + (1 - mask.unsqueeze(0)) * 1e9
            min_losses, selected_ticks = masked_losses.min(dim=0)  # (B, S)
            loss = (min_losses * mask).sum() / (num_valid + 1e-8)

        elif self.selection == "max_certainty":
            # Select tick with highest max probability (most confident)
            certainties = []
            for logits in all_logits:
                probs = F.softmax(logits, dim=-1)
                cert = probs.max(dim=-1).values  # (B, S)
                certainties.append(cert)
            certainties = torch.stack(certainties, dim=0)  # (T, B, S)

            # Mask padded positions
            masked_cert = certainties * mask.unsqueeze(0)
            selected_ticks = masked_cert.argmax(dim=0)  # (B, S)

            # Gather losses for selected ticks
            selected_losses = tick_losses.gather(
                0, selected_ticks.unsqueeze(0)
            ).squeeze(0)  # (B, S)
            loss = (selected_losses * mask).sum() / (num_valid + 1e-8)

        elif self.selection == "weighted":
            # Soft attention over ticks based on inverse loss
            # Lower loss -> higher weight
            masked_losses = tick_losses + (1 - mask.unsqueeze(0)) * 1e9
            weights = F.softmax(-masked_losses / self.tau, dim=0)  # (T, B, S)
            weighted_loss = (weights * tick_losses).sum(dim=0)  # (B, S)
            loss = (weighted_loss * mask).sum() / (num_valid + 1e-8)
            selected_ticks = weights.argmax(dim=0)

        elif self.selection == "all":
            # Average across all ticks (most stable)
            avg_loss_per_pos = tick_losses.mean(dim=0)  # (B, S)
            loss = (avg_loss_per_pos * mask).sum() / (num_valid + 1e-8)
            # For metrics, report which tick was best
            masked_losses = tick_losses + (1 - mask.unsqueeze(0)) * 1e9
            _, selected_ticks = masked_losses.min(dim=0)

        else:  # "last"
            loss_per_pos = tick_losses[-1]  # (B, S)
            loss = (loss_per_pos * mask).sum() / (num_valid + 1e-8)
            selected_ticks = torch.full((B, S), T - 1, device=device)

        # Compute metrics
        with torch.no_grad():
            # Per-tick average loss (for logging)
            per_tick_loss = []
            for t in range(T):
                tick_loss = (tick_losses[t] * mask).sum() / (num_valid + 1e-8)
                per_tick_loss.append(tick_loss)
            per_tick_loss = torch.stack(per_tick_loss)

            # Average selected tick (measure of "thinking time")
            avg_selected = (selected_ticks.float() * mask).sum() / (num_valid + 1e-8)

            # Tick distribution (how often each tick is selected)
            tick_counts = torch.zeros(T, device=device)
            for t in range(T):
                tick_counts[t] = ((selected_ticks == t).float() * mask).sum()
            tick_dist = tick_counts / (num_valid + 1e-8)

        metrics = {
            "per_tick_loss": per_tick_loss,          # (T,)
            "avg_selected_tick": avg_selected,        # scalar
            "tick_distribution": tick_dist,           # (T,)
            "selected_ticks": selected_ticks,         # (B, S)
            "num_valid_tokens": num_valid,            # scalar
        }

        return loss, metrics


class CTMPerplexity:
    """
    Compute perplexity for CTM models.

    Can compute perplexity at each tick or with tick selection.

    Note: This class assumes labels are ALREADY shifted (i.e., labels[i] is the
    target for logits[i]). The dataset should provide input_ids = tokens[:-1]
    and labels = tokens[1:].
    """

    def __init__(self, pad_token_id: int = -100, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(
        self,
        all_logits: List[torch.Tensor],
        labels: torch.Tensor,
        selection: str = "last"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perplexity metrics.

        Args:
            all_logits: List of (B, S, V) logits
            labels: (B, S) labels (already shifted by dataset)
            selection: Selection strategy. Supports same values as CTMLoss:
                       "min_loss", "max_certainty", "weighted", "last", "all"

        Returns:
            Dict with per-tick perplexity and selected perplexity
        """
        B, S = labels.shape
        T = len(all_logits)
        V = all_logits[0].size(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')

        # Determine mask for valid tokens
        mask_token = self.ignore_index if self.ignore_index != -100 else self.pad_token_id
        if mask_token == -100:
            mask = torch.ones(B * S, device=labels.device, dtype=torch.bool)
        else:
            mask = (labels.view(-1) != mask_token)

        perplexities = []
        for logits in all_logits:
            # NO shifting - dataset already provides shifted labels
            loss = criterion(
                logits.view(-1, V),
                labels.view(-1)
            )

            # Compute average loss over valid tokens
            avg_loss = loss[mask].mean() if mask.any() else loss.mean()
            ppl = torch.exp(avg_loss)
            perplexities.append(ppl)

        perplexities = torch.stack(perplexities)

        # Support same selection strings as CTMLoss
        if selection in ("min_loss", "min"):
            selected_ppl = perplexities.min()
        elif selection == "last":
            selected_ppl = perplexities[-1]
        elif selection == "max_certainty":
            # For perplexity, use the tick with lowest perplexity (most confident)
            selected_ppl = perplexities.min()
        elif selection == "weighted":
            # Weight by inverse perplexity
            weights = F.softmax(-perplexities, dim=0)
            selected_ppl = (weights * perplexities).sum()
        elif selection == "all":
            selected_ppl = perplexities.mean()
        else:
            # Default fallback
            selected_ppl = perplexities.mean()

        return {
            "per_tick_perplexity": perplexities,
            "selected_perplexity": selected_ppl,
            "best_tick": perplexities.argmin(),
        }

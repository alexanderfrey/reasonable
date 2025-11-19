import torch
import torch.nn as nn
from typing import Optional


class IdentityBlock(nn.Module):
    """
    Maintains a persistent identity vector and produces bias vectors that can be
    injected into the main transformer. The module learns an internal state and
    optionally conditions it on summarized hidden states from the current batch.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim or d_model
        self.hidden_dim = hidden_dim or max(self.state_dim * 2, d_model)

        self.identity_state = nn.Parameter(torch.zeros(self.state_dim))
        self.context_to_state = nn.Linear(d_model, self.state_dim)

        self.controller = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

        self.state_norm = nn.LayerNorm(self.state_dim)
        self.to_model = nn.Linear(self.state_dim, d_model)

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Produce the identity vector for the given batch.

        Args:
            context: Optional tensor of shape [B, d_model] used to nudge the
                persistent identity state toward the current batch.
            batch_size: Number of samples when no context is provided.

        Returns:
            Tensor of shape [B, d_model] representing the identity bias.
        """
        if context is not None:
            batch = context.size(0)
        elif batch_size is not None:
            batch = batch_size
        else:
            batch = 1

        base_state = self.identity_state.unsqueeze(0).expand(batch, -1)

        if context is not None:
            ctx_state = self.context_to_state(context)
            ctrl_input = torch.cat([base_state, ctx_state], dim=-1)
            delta = self.controller(ctrl_input)
            state = base_state + delta
        else:
            state = base_state

        state = self.state_norm(state)
        return self.to_model(state)

    @staticmethod
    def summarize_token_context(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reduce token-wise representations to a batch-level summary vector.

        Args:
            hidden_states: Tensor of shape [B, S, D].
            attention_mask: Optional mask [B, S] with 1 for valid tokens.

        Returns:
            Tensor of shape [B, D] summarizing the sequence.
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        weights = attention_mask.unsqueeze(-1).type_as(hidden_states)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        summed = (hidden_states * weights).sum(dim=1)
        return summed / denom

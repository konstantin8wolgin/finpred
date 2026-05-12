"""TCN discriminator / WGAN critic: (T, 5) sequence -> scalar score.

ARCHITECTURE (pinned — see CLAUDE.md): dilated-causal TCN mirroring the generator, then
global average pool over time -> linear -> scalar (no sigmoid; WGAN critic). Higher score =
"more real". At inference the quiz picks argmax over candidates.
"""

from __future__ import annotations

import torch
from torch import nn

from finpred.models.tcn_blocks import TCNBlock


class Discriminator(nn.Module):
    """QuantGAN-style TCN critic."""

    def __init__(
        self,
        hidden_channels: int,
        n_blocks: int,
        kernel_size: int,
        dilations: list[int],
        n_features: int = 5,
    ) -> None:
        super().__init__()
        if len(dilations) != n_blocks:
            raise ValueError(
                f"len(dilations)={len(dilations)} must equal n_blocks={n_blocks}"
            )

        # Project input features to channel dimension; operate in (B, C, T) throughout.
        self.input_proj = nn.Conv1d(n_features, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                TCNBlock(hidden_channels, hidden_channels, kernel_size, dilations[i])
                for i in range(n_blocks)
            ]
        )

        # Global average pool over time, then linear head to scalar.
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features) -> (B,) critic scores."""
        # (B, T, n_features) -> (B, n_features, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # Global average pool over time: (B, hidden_channels, T) -> (B, hidden_channels)
        x = x.mean(dim=-1)

        # Linear head: (B, hidden_channels) -> (B, 1) -> (B,)
        x = self.head(x)
        return x.squeeze(-1)

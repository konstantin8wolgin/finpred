"""TCN discriminator / WGAN critic: (T, 5) sequence -> scalar score.

ARCHITECTURE (pinned — see CLAUDE.md): dilated-causal TCN mirroring the generator, then
global average pool over time -> linear -> scalar (no sigmoid; WGAN critic). Higher score =
"more real". At inference the quiz picks argmax over candidates.

STUB (Phase B).
"""

from __future__ import annotations

import torch
from torch import nn


class Discriminator(nn.Module):
    """QuantGAN-style TCN critic. STUB."""

    def __init__(
        self,
        hidden_channels: int,
        n_blocks: int,
        kernel_size: int,
        dilations: list[int],
        n_features: int = 5,
    ) -> None:
        super().__init__()
        raise NotImplementedError("models.discriminator.Discriminator — implement in Phase B")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features) -> (B,) critic scores."""
        raise NotImplementedError

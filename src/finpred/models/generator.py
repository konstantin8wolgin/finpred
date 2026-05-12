"""TCN generator: noise -> synthetic (T, 5) OHLCV log-return-style sequence.

ARCHITECTURE (pinned — see CLAUDE.md): QuantGAN-style dilated-causal TCN.
  Input  noise z: (B, T, z_dim).  Optional `cond`: accepted but IGNORED in v1.
  Output: (B, T, 5) = log-returns of Open/High/Low/Close + log-diff Volume.

OHLC CONSISTENCY CONTRACT (must hold for EVERY emitted bar, by construction):
    low <= min(open, close) <= max(open, close) <= high
The output head enforces this via a 5-channel projection from the TCN backbone:
  channel 0  dlog_close   — log-return of close for this bar
  channel 1  dlog_volume  — log-difference of volume for this bar
  channel 2  open_offset  — log(open_t / close_t): same-bar log spread; positive ↔ open > close
  channel 3  hi_raw       — unconstrained; softplus gives the high margin above max(open, close)
  channel 4  lo_raw       — unconstrained; softplus gives the low margin below min(open, close)

Reconstruction in price space (per bar):
  close_price = prev_close * exp(dlog_close)
  open_price  = close_price * exp(open_offset)
  high_price  = max(open_price, close_price) + softplus(hi_raw)   # always >= both
  low_price   = min(open_price, close_price) - softplus(lo_raw)   # always <= both
Never emit O/H/L as free values. A test asserts no invalid candle is ever produced.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from finpred.models.tcn_blocks import TCNBlock


class Generator(nn.Module):
    """QuantGAN-style TCN generator with OHLC-consistent output head."""

    def __init__(
        self,
        z_dim: int,
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

        # Project noise to channel dimension; operate in (B, C, T) throughout.
        self.input_proj = nn.Conv1d(z_dim, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                TCNBlock(hidden_channels, hidden_channels, kernel_size, dilations[i])
                for i in range(n_blocks)
            ]
        )

        # Raw 5-channel head — softplus applied to channels 3 & 4 in forward.
        self.output_proj = nn.Conv1d(hidden_channels, n_features, kernel_size=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, T, z_dim) -> (B, T, 5). `cond` reserved for v2; ignored in v1."""
        # (B, T, z_dim) -> (B, z_dim, T) for Conv1d
        x = z.permute(0, 2, 1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        # (B, 5, T) raw logits
        x = self.output_proj(x)

        # Apply softplus to the high and low margin channels so they are strictly positive.
        # Channels 0, 1, 2 are passed through unchanged (log-return / log-spread values).
        x = torch.cat(
            [
                x[:, :3, :],
                F.softplus(x[:, 3:4, :]),
                F.softplus(x[:, 4:5, :]),
            ],
            dim=1,
        )

        # (B, 5, T) -> (B, T, 5)
        return x.permute(0, 2, 1)

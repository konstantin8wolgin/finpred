"""TCN generator: noise -> synthetic (T, 5) OHLCV log-return-style sequence.

ARCHITECTURE (pinned — see CLAUDE.md): QuantGAN-style dilated-causal TCN.
  Input  noise z: (B, T, z_dim).  Optional `cond`: accepted but IGNORED in v1.
  Output: (B, T, 5) — same 5-channel format as ingest.py / WindowDataset:
    channel 0  logret_open   = log(Open_t  / Close_{t-1})
    channel 1  logret_high   = log(high_margin + 1e-8)  where high_margin = High - max(Open, Close)
    channel 2  logret_low    = log(low_margin  + 1e-8)  where low_margin  = min(Open, Close) - Low
    channel 3  logret_close  = log(Close_t / Close_{t-1})
    channel 4  logdiff_volume = log(Volume_t) - log(Volume_{t-1})

OHLC CONSISTENCY CONTRACT (must hold for EVERY emitted bar, by construction):
    low <= min(open, close) <= max(open, close) <= high
The output head enforces this by parameterizing channels 1 and 2 as
    log(softplus(raw) + 1e-8)
so that high_margin = softplus(raw_hi) > 0 and low_margin = softplus(raw_lo) > 0 always,
making high = max(open, close) + high_margin >= max(open, close) and
      low  = min(open, close) - low_margin  <= min(open, close).
Invalid candles are structurally impossible regardless of network weights.

Price reconstruction (render module only — the discriminator sees log-returns, not prices):
  close_t  = close_{t-1} * exp(logret_close)
  open_t   = close_{t-1} * exp(logret_open)
  high_margin = exp(logret_high) - 1e-8          # = softplus(raw_hi) > 0
  high_t   = max(open_t, close_t) + high_margin
  low_margin  = exp(logret_low) - 1e-8           # = softplus(raw_lo) > 0
  low_t    = min(open_t, close_t) - low_margin
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

        # Transform channels 1 and 2 (high/low margins) to match ingest.py format:
        # logret_high = log(softplus(raw_hi) + 1e-8), logret_low = log(softplus(raw_lo) + 1e-8).
        # softplus ensures margin > 0; taking log matches log(clip(margin,0)+1e-8) in ingest.py.
        x = torch.cat(
            [
                x[:, 0:1, :],
                torch.log(F.softplus(x[:, 1:2, :]) + 1e-8),
                torch.log(F.softplus(x[:, 2:3, :]) + 1e-8),
                x[:, 3:4, :],
                x[:, 4:5, :],
            ],
            dim=1,
        )

        # (B, 5, T) -> (B, T, 5)
        return x.permute(0, 2, 1)

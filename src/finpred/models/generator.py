"""TCN generator: noise -> synthetic (T, 5) OHLCV log-return-style sequence.

ARCHITECTURE (pinned — see CLAUDE.md): QuantGAN-style dilated-causal TCN.
  Input  noise z: (B, T, z_dim).  Optional `cond`: accepted but IGNORED in v1.
  Output: (B, T, 5) = log-returns of Open/High/Low/Close + log-diff Volume.

OHLC CONSISTENCY CONTRACT (must hold for EVERY emitted bar, by construction):
    low <= min(open, close) <= max(open, close) <= high
The output head enforces this: the net emits (dlog_close, dlog_volume, open_offset, hi_raw,
lo_raw); then in price space  high = max(open, close) + softplus(hi_raw),
low = min(open, close) - softplus(lo_raw). Never emit O/H/L as free values. A test asserts no
invalid candle is ever produced.

STUB (Phase B).
"""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """QuantGAN-style TCN generator. STUB."""

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
        raise NotImplementedError("models.generator.Generator — implement in Phase B")

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, T, z_dim) -> (B, T, n_features). `cond` reserved for v2; ignored in v1."""
        raise NotImplementedError

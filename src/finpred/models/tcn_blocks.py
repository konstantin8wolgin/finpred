"""Dilated causal convolution residual block — the shared building block of G and D.

STUB (Phase B). Implement a standard TCN residual block:
  - two dilated *causal* 1D convs (left-pad by (kernel_size-1)*dilation, no future leakage),
  - weight-norm or layer-norm, GELU/ReLU, dropout,
  - 1x1 conv on the residual path when channel counts differ.
A stack of these with dilations {1,2,4,8,16,32} gives a receptive field >= the longest window.
"""

from __future__ import annotations

import torch
from torch import nn


class TCNBlock(nn.Module):
    """One dilated causal residual block. STUB."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        raise NotImplementedError("models.tcn_blocks.TCNBlock — implement in Phase B")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        raise NotImplementedError

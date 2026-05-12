"""Dilated causal convolution residual block — the shared building block of G and D.

Causal padding: each Conv1d is left-padded by (kernel_size - 1) * dilation samples so that
the output at time t depends only on inputs at times <= t (no future leakage).

Residual connection: when in_ch != out_ch, a 1x1 Conv1d projects x to out_ch before the add.
When in_ch == out_ch, the identity is used directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class _CausalConv1d(nn.Module):
    """Conv1d with causal left-padding baked in."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """One dilated causal residual block.

    Layout:
        causal_conv1 -> GELU -> dropout -> causal_conv2 -> dropout
        + residual_skip(x)
        -> GELU
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.dropout = nn.Dropout(0.1)
        # 1x1 projection only when channel counts differ
        self.skip: nn.Conv1d | None = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, C, T)
        residual = x if self.skip is None else self.skip(x)
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return self.activation(out + residual)

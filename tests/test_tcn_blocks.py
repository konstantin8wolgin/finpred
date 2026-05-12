from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from finpred.models.tcn_blocks import TCNBlock


def test_output_shape():
    block = TCNBlock(in_ch=16, out_ch=32, kernel_size=3, dilation=1)
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (2, 32, 64)


def test_causal_no_future_leakage():
    torch.manual_seed(0)
    block = TCNBlock(in_ch=8, out_ch=8, kernel_size=3, dilation=4)
    block.eval()
    x = torch.randn(1, 8, 50, dtype=torch.float32)
    with torch.no_grad():
        y_full = block(x.clone())
    x_masked = x.clone()
    x_masked[:, :, 25:] = 0.0
    with torch.no_grad():
        y_masked = block(x_masked)
    # output at t=24 must be identical — no future leakage
    assert torch.allclose(y_full[:, :, 24], y_masked[:, :, 24], atol=1e-5), (
        "Future leakage detected: output at t=24 changed when t>=25 was zeroed"
    )


def test_residual_skip_when_channels_differ():
    block = TCNBlock(in_ch=4, out_ch=8, kernel_size=3, dilation=1)
    # the skip/residual/downsample attribute must exist and be a Conv1d with kernel_size=1
    skip = getattr(block, "skip", None) or getattr(block, "downsample", None) or getattr(block, "residual", None)
    assert skip is not None, "Expected a skip/downsample/residual attribute when in_ch != out_ch"
    assert isinstance(skip, nn.Conv1d), f"Skip should be nn.Conv1d, got {type(skip)}"
    assert skip.kernel_size == (1,), f"Skip conv should have kernel_size=1, got {skip.kernel_size}"


def test_receptive_field():
    # Stack 4 TCNBlocks with dilations [1,2,4,8], kernel_size=3, same channels
    # Receptive field = 1 + 2*(3-1)*(1+2+4+8) = 1 + 4*15 = 61
    blocks = nn.Sequential(
        TCNBlock(in_ch=8, out_ch=8, kernel_size=3, dilation=1),
        TCNBlock(in_ch=8, out_ch=8, kernel_size=3, dilation=2),
        TCNBlock(in_ch=8, out_ch=8, kernel_size=3, dilation=4),
        TCNBlock(in_ch=8, out_ch=8, kernel_size=3, dilation=8),
    )
    for m in blocks.modules():
        if isinstance(m, nn.Module):
            m.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 8, 100, dtype=torch.float32)
    with torch.no_grad():
        y_full = blocks(x.clone())

    x_masked = x.clone()
    x_masked[:, :, 31:] = 0.0  # zero everything after t=30
    with torch.no_grad():
        y_masked = blocks(x_masked)

    assert torch.allclose(y_full[:, :, 30], y_masked[:, :, 30], atol=1e-5), (
        "Causality violated in stacked blocks: output at t=30 changed when t>30 was zeroed"
    )


def test_same_channels_residual_is_identity_path():
    block = TCNBlock(in_ch=16, out_ch=16, kernel_size=3, dilation=1)
    # residual path must not be a Conv1d when channels match
    skip = getattr(block, "skip", None) or getattr(block, "downsample", None) or getattr(block, "residual", None)
    if skip is not None:
        assert not isinstance(skip, nn.Conv1d), (
            "When in_ch == out_ch, residual skip should be identity (not Conv1d)"
        )
    # output should still differ from input (conv path is active)
    torch.manual_seed(7)
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        y = block(x)
    assert not torch.allclose(x, y), "Block output should differ from input (conv path must be active)"

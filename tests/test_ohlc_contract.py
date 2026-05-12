"""OHLC consistency contract tests for models.generator.Generator.

Implement alongside models/generator.py in Phase B using a tiny model config (z_dim=3,
hidden_channels=8, n_blocks=2, kernel_size=3, dilations=[1,2]).

Required assertions:

test_no_invalid_candles
  Sample 200 noise tensors z ~ N(0,1) of shape (B=4, T=64, z_dim=3), pass through Generator,
  reconstruct per-bar prices from the 5-channel output (see generator.py docstring for the
  exact parameterization). Assert for every (batch, time) step:
    low <= open
    low <= close
    high >= open
    high >= close
  Any violation means the OHLC contract is broken — fail immediately with the offending values.

test_softplus_margin_positive
  The high margin (high - max(open, close)) and low margin (min(open, close) - low) must both
  be strictly positive for all bars. A value of exactly 0.0 is a sign the softplus path was
  bypassed.

test_output_shape
  Generator(z).shape == (B, T, 5) for arbitrary B and T.

test_cond_ignored_in_v1
  Generator(z, cond=some_tensor) == Generator(z, cond=None) element-wise (v1 must ignore cond).

Implementation notes:
  - Use torch.no_grad() + float32 (not BF16) for deterministic assertions.
  - Seed with torch.manual_seed(0) for reproducibility.
  - Run with `uv run pytest tests/test_ohlc_contract.py -v`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from finpred.models.generator import Generator


TINY_CFG = dict(
    z_dim=3,
    hidden_channels=8,
    n_blocks=2,
    kernel_size=3,
    dilations=[1, 2],
)


def _make_gen() -> Generator:
    torch.manual_seed(0)
    return Generator(**TINY_CFG).float().eval()


def _reconstruct_prices(out: torch.Tensor):
    """Return (open, close, high, low) pseudo-prices given the 5-channel generator output.

    Reconstruction is purely for contract verification; absolute price levels are arbitrary
    since we start from cumsum(0) = 1.0 for convenience.
    """
    # out: (B, T, 5)
    dlog_close  = out[..., 0]
    dlog_volume = out[..., 1]  # noqa: F841 — not used in price reconstruction
    open_offset = out[..., 2]
    hi_raw      = out[..., 3]
    lo_raw      = out[..., 4]

    # Accumulate close prices from log-returns; start at 1.0.
    close = torch.exp(torch.cumsum(dlog_close, dim=-1))
    open_ = close * torch.exp(open_offset)

    high_margin = F.softplus(hi_raw)
    low_margin  = F.softplus(lo_raw)

    high = torch.maximum(open_, close) + high_margin
    low  = torch.minimum(open_, close) - low_margin

    return open_, close, high, low


def test_no_invalid_candles():
    gen = _make_gen()
    torch.manual_seed(0)
    with torch.no_grad():
        for _ in range(200):
            z = torch.randn(4, 64, 3)
            out = gen(z)
            open_, close, high, low = _reconstruct_prices(out)

            # low <= open
            violation = low > open_
            assert not violation.any(), (
                f"low > open: max violation {(low - open_)[violation].max().item():.6f}"
            )
            # low <= close
            violation = low > close
            assert not violation.any(), (
                f"low > close: max violation {(low - close)[violation].max().item():.6f}"
            )
            # high >= open
            violation = high < open_
            assert not violation.any(), (
                f"high < open: max violation {(open_ - high)[violation].max().item():.6f}"
            )
            # high >= close
            violation = high < close
            assert not violation.any(), (
                f"high < close: max violation {(close - high)[violation].max().item():.6f}"
            )


def test_softplus_margin_positive():
    gen = _make_gen()
    torch.manual_seed(0)
    with torch.no_grad():
        z = torch.randn(4, 64, 3)
        out = gen(z)
        hi_raw = out[..., 3]
        lo_raw = out[..., 4]
        high_margin = F.softplus(hi_raw)
        low_margin  = F.softplus(lo_raw)
        assert (high_margin > 0).all(), "high_margin contains non-positive values"
        assert (low_margin > 0).all(), "low_margin contains non-positive values"


def test_output_shape():
    gen = _make_gen()
    with torch.no_grad():
        for B, T in [(1, 1), (2, 32), (4, 64), (8, 252)]:
            z = torch.randn(B, T, 3)
            out = gen(z)
            assert out.shape == (B, T, 5), f"Expected ({B}, {T}, 5), got {out.shape}"


def test_cond_ignored_in_v1():
    gen = _make_gen()
    torch.manual_seed(42)
    z = torch.randn(4, 64, 3)
    cond = torch.randn(4, 64, 7)
    with torch.no_grad():
        out_no_cond   = gen(z, cond=None)
        out_with_cond = gen(z, cond=cond)
    assert torch.equal(out_no_cond, out_with_cond), (
        "Generator output differs when cond is provided — cond must be ignored in v1"
    )

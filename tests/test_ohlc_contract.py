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

    Format: [logret_open, logret_high, logret_low, logret_close, logdiff_volume]
    Reconstruction is purely for contract verification; absolute price levels are arbitrary
    since we start from prev_close = 1.0 at t=0.
    """
    # out: (B, T, 5)
    logret_open  = out[..., 0]  # log(Open_t / Close_{t-1})
    logret_high  = out[..., 1]  # log(high_margin + 1e-8)
    logret_low   = out[..., 2]  # log(low_margin + 1e-8)
    logret_close = out[..., 3]  # log(Close_t / Close_{t-1})

    # Cumulative log-close (prev_close at t=0 is 1.0, so log_prev_close[0] = 0).
    log_close = torch.cumsum(logret_close, dim=-1)
    log_prev_close = torch.cat(
        [torch.zeros_like(log_close[..., :1]), log_close[..., :-1]], dim=-1
    )

    close = torch.exp(log_close)
    open_ = torch.exp(log_prev_close + logret_open)

    # Margins: exp(logret_high) - 1e-8 = softplus(raw_hi) > 0 by construction.
    high_margin = torch.exp(logret_high) - 1e-8
    low_margin  = torch.exp(logret_low)  - 1e-8

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
        # ch1 = log(softplus(raw_hi) + 1e-8), ch2 = log(softplus(raw_lo) + 1e-8)
        high_margin = torch.exp(out[..., 1]) - 1e-8  # = softplus(raw_hi) > 0
        low_margin  = torch.exp(out[..., 2]) - 1e-8  # = softplus(raw_lo) > 0
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

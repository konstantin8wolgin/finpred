"""Tests for src/finpred/render/ohlcv_to_png.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finpred.render.ohlcv_to_png import ohlcv_to_png, reconstruct_ohlcv


RNG = np.random.default_rng(42)


def _make_abs_seq(T: int = 60) -> np.ndarray:
    """Absolute OHLCV array — all values positive, OHLC contract satisfied."""
    close = 100.0 + RNG.normal(0, 1, T).cumsum()
    close = np.clip(close, 10.0, None)
    open_ = close * (1 + RNG.normal(0, 0.005, T))
    open_ = np.clip(open_, 10.0, None)
    high = np.maximum(open_, close) * (1 + np.abs(RNG.normal(0, 0.003, T)))
    low = np.minimum(open_, close) * (1 - np.abs(RNG.normal(0, 0.003, T)))
    volume = np.abs(RNG.normal(1e6, 1e5, T))
    return np.stack([open_, high, low, close, volume], axis=1)


def _make_std_seq(T: int = 60) -> tuple[np.ndarray, dict]:
    """Standardized log-return sequence + matching ticker_stats."""
    seq = RNG.normal(0.0, 1.0, (T, 5))  # standard normal
    ticker_stats = {
        'mean': np.zeros(5),
        'std': np.ones(5) * 0.01,
    }
    return seq, ticker_stats


# ---------------------------------------------------------------------------
# test_ohlcv_to_png_absolute_mode
# ---------------------------------------------------------------------------

def test_ohlcv_to_png_absolute_mode(tmp_path):
    seq = _make_abs_seq(60)
    out = tmp_path / 'out.png'
    result = ohlcv_to_png(seq, out)
    assert out.exists(), "output PNG was not created"
    assert out.stat().st_size > 0, "output PNG is empty"
    assert result == out.resolve()


# ---------------------------------------------------------------------------
# test_ohlcv_to_png_standardized_mode
# ---------------------------------------------------------------------------

def test_ohlcv_to_png_standardized_mode(tmp_path):
    seq, ticker_stats = _make_std_seq(60)
    out = tmp_path / 'out.png'
    result = ohlcv_to_png(
        seq,
        out,
        ticker_stats=ticker_stats,
        start_price=150.0,
    )
    assert out.exists(), "output PNG was not created"
    assert out.stat().st_size > 0, "output PNG is empty"


# ---------------------------------------------------------------------------
# test_ohlc_validity
# ---------------------------------------------------------------------------

def test_ohlc_validity(tmp_path):
    """After reconstruction low <= open, low <= close, high >= open, high >= close."""
    seq, ticker_stats = _make_std_seq(60)
    df = reconstruct_ohlcv(seq, ticker_stats, start_price=100.0)

    open_ = df['Open'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    close = df['Close'].to_numpy()

    assert np.all(low <= open_), "low > open for some bars"
    assert np.all(low <= close), "low > close for some bars"
    assert np.all(high >= open_), "high < open for some bars"
    assert np.all(high >= close), "high < close for some bars"


# ---------------------------------------------------------------------------
# test_y_range_applied
# ---------------------------------------------------------------------------

def test_y_range_applied(tmp_path):
    seq = _make_abs_seq(60)
    out = tmp_path / 'out.png'
    # Should not raise even with an explicit y_range
    ohlcv_to_png(seq, out, y_range=(50.0, 200.0))
    assert out.exists()


# ---------------------------------------------------------------------------
# test_identical_axes
# ---------------------------------------------------------------------------

def test_identical_axes(tmp_path):
    """Two charts rendered with the same y_range and date_index should both be created."""
    T = 60
    date_index = pd.bdate_range('2013-01-02', periods=T)
    y_range = (80.0, 130.0)

    seq_a = _make_abs_seq(T)
    seq_b = _make_abs_seq(T)

    out_a = tmp_path / 'chart_a.png'
    out_b = tmp_path / 'chart_b.png'

    ohlcv_to_png(seq_a, out_a, y_range=y_range, date_index=date_index, title='Chart A')
    ohlcv_to_png(seq_b, out_b, y_range=y_range, date_index=date_index, title='Chart B')

    assert out_a.exists(), "chart_a.png was not created"
    assert out_b.exists(), "chart_b.png was not created"
    assert out_a.stat().st_size > 0
    assert out_b.stat().st_size > 0

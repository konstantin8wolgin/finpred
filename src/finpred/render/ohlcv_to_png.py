"""Render a numeric OHLCV sequence to a candlestick PNG (human inspection only).

The model never sees pixels — this exists so a human can eyeball validation samples and the
N-way quiz. See CLAUDE.md.

STUB (Phase A.1). Implement:
  - reconstruct an absolute-price OHLCV frame from a standardized (T, 5) log-return sequence:
    de-standardize with the ticker's saved stats, cumsum the close log-returns from a start
    price, derive O/H/L from the modeled spreads (respecting low<=min(o,c)<=max(o,c)<=high),
  - draw with mplfinance,
  - `y_range` / `date_index` kwargs so a set of charts can be rendered with IDENTICAL axes and
    date formatting (the quiz needs this so fakes aren't identifiable by rendering artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ohlcv_to_png(
    seq: np.ndarray,
    out_path: str | Path,
    *,
    start_price: float = 100.0,
    ticker_stats: dict | None = None,
    y_range: tuple[float, float] | None = None,
    date_index=None,
    title: str | None = None,
) -> Path:
    """Render `seq` (T, 5) standardized log-returns as a candlestick PNG. Returns the path. STUB."""
    raise NotImplementedError("render.ohlcv_to_png — implement in Phase A.1")

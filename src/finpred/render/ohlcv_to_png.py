"""Render a numeric OHLCV sequence to a candlestick PNG (human inspection only).

The model never sees pixels — this exists so a human can eyeball validation samples and the
N-way quiz. See CLAUDE.md.

OHLC consistency contract (matches generator.py):
  col 0: logret_open   — log-return of Open relative to prev Close
  col 1: high_margin   — softplus output (always > 0); high = max(open, close) + high_margin
  col 2: low_margin    — softplus output (always > 0); low  = min(open, close) - low_margin
  col 3: logret_close  — log-return of Close relative to prev Close
  col 4: logdiff_volume

When ticker_stats is provided seq is in standardized form; de-standardize first.
When ticker_stats is None seq columns are [Open, High, Low, Close, Volume] in absolute space.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import numpy as np
import pandas as pd
import mplfinance as mpf


def reconstruct_ohlcv(
    seq: np.ndarray,
    ticker_stats: dict,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """De-standardize seq and reconstruct absolute-price OHLCV DataFrame.

    Parameters
    ----------
    seq:
        Shape (T, 5) standardized log-return sequence.
    ticker_stats:
        Dict with keys 'mean' and 'std', each shape (5,).
    start_price:
        Absolute price of the first Close before any log-return is applied.

    Returns
    -------
    pd.DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'] and a
    RangeIndex (caller sets a DatetimeIndex before passing to mplfinance).
    """
    mean = np.asarray(ticker_stats['mean'], dtype=float)
    std = np.asarray(ticker_stats['std'], dtype=float)
    raw = seq * std + mean  # shape (T, 5)

    T = raw.shape[0]

    # Close prices via cumsum of log-returns
    close_logreturns = raw[:, 3]
    close = start_price * np.exp(np.cumsum(close_logreturns))  # shape (T,)

    # Open prices: open[t] = close[t-1] * exp(logret_open[t])
    prev_close = np.empty(T, dtype=float)
    prev_close[0] = start_price
    prev_close[1:] = close[:-1]
    open_prices = prev_close * np.exp(raw[:, 0])

    # High / Low: col 1/2 are log(margin + 1e-8) from ingest; invert with exp.
    high_margin = np.exp(raw[:, 1])
    low_margin = np.exp(raw[:, 2])
    high = np.maximum(open_prices, close) + high_margin
    low = np.minimum(open_prices, close) - low_margin

    # Volume: reconstruct a relative-scale series via cumsum of log-differences
    volume = np.exp(np.cumsum(raw[:, 4])) * 1e6

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    })
    return df


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
    """Render `seq` (T, 5) as a candlestick PNG. Returns the output path.

    Parameters
    ----------
    seq:
        Shape (T, 5). Either standardized log-returns (when ticker_stats is given) or
        absolute [Open, High, Low, Close, Volume] columns.
    out_path:
        Destination file path (PNG).
    start_price:
        Seed price for reconstructing absolute levels from log-returns.
    ticker_stats:
        If provided, dict with 'mean' and 'std' (shape (5,) each). seq is treated as
        standardized and will be de-standardized + reconstructed.
    y_range:
        (low, high) ylim for the price panel — use the same value across all quiz charts
        so axes are identical and fakes cannot be identified by rendering artifacts.
    date_index:
        pd.DatetimeIndex or list of dates of length T. Defaults to business-day range
        starting 2012-07-02.
    title:
        Chart title string.

    Returns
    -------
    Path
        Resolved path to the written PNG.
    """
    out_path = Path(out_path)
    seq = np.asarray(seq, dtype=float)
    T = seq.shape[0]

    if ticker_stats is not None:
        df = reconstruct_ohlcv(seq, ticker_stats, start_price)
    else:
        df = pd.DataFrame(
            seq,
            columns=['Open', 'High', 'Low', 'Close', 'Volume'],
        )

    if date_index is not None:
        idx = pd.DatetimeIndex(date_index)
    else:
        idx = pd.bdate_range('2012-07-02', periods=T)

    df.index = idx

    plot_kwargs: dict = dict(
        type='candle',
        style='charles',
        savefig=str(out_path),
        volume=True,
    )

    if y_range is not None:
        plot_kwargs['ylim'] = y_range

    if title is not None:
        plot_kwargs['title'] = title

    mpf.plot(df, **plot_kwargs)

    return out_path.resolve()

"""Overlapping rolling-window sampler over the cached OHLCV, with strict era-cutoff enforcement.

Parquet schema (one file per ticker at {cache_dir}/{ticker}.parquet):
    date           : date or str YYYY-MM-DD
    logret_open    : float  (standardized log-return of open)
    logret_high    : float
    logret_low     : float
    logret_close   : float
    logdiff_volume : float

Era boundaries (from cfg.data):
    train : date <  val_start_date               (i.e. <= train day before val)
    val   : val_start_date <= date <= train_end_date
    eval  : eval_start_date <= date <= eval_end_date

Windows are emitted in a round-robin / stratified fashion across tickers so that no single
ticker dominates a training epoch.
"""
from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path
from typing import Iterator

import polars as pl
import torch

FEATURE_COLS = [
    "logret_open",
    "logret_high",
    "logret_low",
    "logret_close",
    "logdiff_volume",
]


def _parse_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def _slice_split(df: pl.DataFrame, split: str, cfg) -> pl.DataFrame:
    """Return rows of *df* that belong to *split*, based on era boundaries in *cfg.data*."""
    val_start = _parse_date(cfg.data.val_start_date)
    train_end = _parse_date(cfg.data.train_end_date)
    eval_start = _parse_date(cfg.data.eval_start_date)
    eval_end = _parse_date(cfg.data.eval_end_date)

    # Normalise the date column to Python date objects for comparison.
    # polars may store dates as pl.Date or as strings.
    col = df["date"]
    if col.dtype == pl.Utf8 or col.dtype == pl.String:
        df = df.with_columns(pl.col("date").str.to_date().alias("date"))

    if split == "train":
        mask = pl.col("date") < val_start
    elif split == "val":
        mask = (pl.col("date") >= val_start) & (pl.col("date") <= train_end)
    elif split == "eval":
        mask = (pl.col("date") >= eval_start) & (pl.col("date") <= eval_end)
    else:
        raise ValueError(f"Unknown split: {split!r}. Must be 'train', 'val', or 'eval'.")

    return df.filter(mask).sort("date")


def _rolling_windows(
    df: pl.DataFrame,
    length: int,
    stride: int,
    return_dates: bool,
) -> list[tuple[torch.Tensor, list[date]] | torch.Tensor]:
    """Emit all rolling windows of *length* with *stride* from *df*."""
    n = len(df)
    if n < length:
        return []

    feature_array = df.select(FEATURE_COLS).to_numpy()
    date_col = df["date"].to_list()

    results = []
    for start in range(0, n - length + 1, stride):
        end = start + length
        tensor = torch.tensor(feature_array[start:end], dtype=torch.float32)
        if return_dates:
            window_dates = date_col[start:end]
            results.append((tensor, window_dates))
        else:
            results.append(tensor)
    return results


def _load_ticker_df(cache_dir: Path, ticker: str) -> pl.DataFrame | None:
    parquet_path = cache_dir / f"{ticker}.parquet"
    if not parquet_path.exists():
        return None
    return pl.read_parquet(parquet_path)


class WindowDataset(torch.utils.data.IterableDataset):
    """Iterable dataset of rolling OHLCV windows, with strict era-cutoff enforcement.

    Args:
        cfg:          Config namespace/object with cfg.data.* and cfg.windows.* attributes.
        split:        One of "train", "val", "eval".
        length:       Window length in trading days.  If None, uses cfg.windows.lengths[0].
        return_dates: If True, each iteration yields (tensor, list[date]) instead of tensor.
                      Intended for tests only.
    """

    def __init__(
        self,
        cfg,
        split: str,
        length: int | None = None,
        *,
        return_dates: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.length = length if length is not None else cfg.windows.lengths[0]
        self.stride = cfg.windows.stride
        self.cache_dir = Path(cfg.data.cache_dir)
        self.return_dates = return_dates

        # Resolve ticker list
        tickers = cfg.data.tickers
        if isinstance(tickers, str):
            # Sentinel value — real ingestion resolves it; for tests tickers is always a list.
            raise ValueError(
                f"cfg.data.tickers is a sentinel string {tickers!r}. "
                "Resolve it before constructing WindowDataset."
            )
        self.tickers: list[str] = list(tickers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _windows_for_ticker(
        self, ticker: str
    ) -> list[tuple[torch.Tensor, list[date]] | torch.Tensor]:
        df = _load_ticker_df(self.cache_dir, ticker)
        if df is None:
            return []
        sliced = _slice_split(df, self.split, self.cfg)
        return _rolling_windows(sliced, self.length, self.stride, self.return_dates)

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor | tuple[torch.Tensor, list[date]]]:
        """Yield windows round-robin across tickers (stratified by ticker)."""
        # Build per-ticker window lists
        ticker_windows: list[list] = []
        for ticker in self.tickers:
            windows = self._windows_for_ticker(ticker)
            if windows:
                ticker_windows.append(windows)

        if not ticker_windows:
            return

        # Round-robin interleaving so no single ticker floods the epoch.
        for window in itertools.chain.from_iterable(
            itertools.zip_longest(*ticker_windows)
        ):
            if window is not None:
                yield window


# ---------------------------------------------------------------------------
# Public helper for the quiz / eval harness
# ---------------------------------------------------------------------------


def fetch_real_window(cfg, ticker: str, end_date: str, length: int) -> torch.Tensor:
    """Return exactly one (length, 5) window ending on *end_date* from the eval split.

    Args:
        cfg:      Config object (same as used for WindowDataset).
        ticker:   Ticker symbol, e.g. "NVDA".
        end_date: ISO date string "YYYY-MM-DD" — the last date of the returned window.
        length:   Number of trading days in the window.

    Returns:
        torch.Tensor of shape (length, 5), dtype float32.

    Raises:
        ValueError: if the ticker has no eval data, or the requested window is not available.
    """
    cache_dir = Path(cfg.data.cache_dir)
    df = _load_ticker_df(cache_dir, ticker)
    if df is None:
        raise ValueError(f"No parquet file found for ticker {ticker!r} in {cache_dir}")

    sliced = _slice_split(df, "eval", cfg)

    # Normalise end_date
    end = _parse_date(end_date)

    # Find the position of end_date in the sorted frame.
    dates = sliced["date"].to_list()
    if end not in dates:
        raise ValueError(
            f"end_date {end_date!r} not found in eval split for ticker {ticker!r}. "
            f"Available range: {dates[0]} .. {dates[-1]}"
        )
    end_idx = dates.index(end)
    start_idx = end_idx - length + 1
    if start_idx < 0:
        raise ValueError(
            f"Not enough eval bars before {end_date!r} for a window of length {length} "
            f"(only {end_idx + 1} bars available)."
        )

    feature_array = sliced.select(FEATURE_COLS).to_numpy()[start_idx : end_idx + 1]
    return torch.tensor(feature_array, dtype=torch.float32)

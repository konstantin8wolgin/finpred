"""Download daily OHLCV and cache it to Parquet under `data/`.

Feature representation (5 columns, matching the OHLC contract in generator.py):

  col 0: logret_open   = log(Open / prev_Close)
  col 1: logret_high   = log(High - max(Open, Close) + 1e-8)   [log of non-negative margin]
  col 2: logret_low    = log(min(Open, Close) - Low  + 1e-8)   [log of non-negative margin]
  col 3: logret_close  = log(Close / prev_Close)
  col 4: logdiff_volume = log(Volume) - log(prev_Volume)

The generator's output head inverts these via:
  high = max(open_price, close_price) + softplus(raw_high)
  low  = min(open_price, close_price) - softplus(raw_low)
so that every bar satisfies low <= min(O,C) <= max(O,C) <= high by construction.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded representative ticker universe for the SP1500+NASDAQ sentinel.
# This list is intentionally comprehensive enough (>50 tickers) to cover major
# sectors and indices without a network call.  A production run may replace
# this with a live constituent pull; the sentinel is resolved at ingest time.
# ---------------------------------------------------------------------------
_SENTINEL = "__SP1500_PLUS_NASDAQ__"

_REPRESENTATIVE_TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "TSLA",
    "INTC", "AMD", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX",
    "KLAC", "MRVL", "ADI", "NXPI", "CDNS", "SNPS", "ANSS", "FTNT",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA",
    "USB", "PNC", "TFC", "COF", "AIG",
    # Healthcare
    "JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD", "BIIB",
    "REGN", "VRTX", "ISRG", "MDT", "ABT",
    # Consumer Discretionary
    "HD", "LOW", "TGT", "COST", "WMT", "NKE", "SBUX", "MCD", "CMG",
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "PSX", "VLO",
    # Industrials
    "BA", "CAT", "DE", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "RTX",
    # Communication Services
    "T", "VZ", "DIS", "CMCSA", "NFLX",
    # Utilities & Real Estate
    "NEE", "DUK", "SO", "AMT", "PLD",
    # Materials
    "LIN", "APD", "ECL", "NEM", "FCX",
    # SPY / index proxies
    "SPY", "QQQ", "IWM",
    # Notable historical delistings (supplemental; may have no data)
    "LEH", "BSC", "WM", "ENRNQ", "GM",
]


# ---------------------------------------------------------------------------
# Public API (importable by tests)
# ---------------------------------------------------------------------------

def resolve_tickers(
    tickers: list[str] | str,
    extra_tickers: list[str] | None = None,
) -> list[str]:
    """Return the resolved list of ticker symbols.

    Parameters
    ----------
    tickers:
        Either a list of ticker strings or the sentinel ``"__SP1500_PLUS_NASDAQ__"``.
        When the sentinel is provided, the representative hard-coded universe is used.
        A production variant could pull live constituents from e.g. Wikipedia or an
        ETF holdings API; for Phase A.1 the hard-coded list is sufficient.
    extra_tickers:
        Additional tickers always appended (e.g. historical delistings from config).

    Returns
    -------
    list[str]
        Deduplicated list of uppercase ticker strings.
    """
    if extra_tickers is None:
        extra_tickers = []

    if isinstance(tickers, str):
        if tickers == _SENTINEL:
            base = list(_REPRESENTATIVE_TICKERS)
        else:
            raise ValueError(f"Unknown ticker sentinel: {tickers!r}")
    else:
        base = [str(t) for t in tickers]

    seen: set[str] = set()
    result: list[str] = []
    for t in base + list(extra_tickers):
        t_upper = t.upper()
        if t_upper not in seen:
            seen.add(t_upper)
            result.append(t_upper)
    return result


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 canonical features from a raw OHLCV DataFrame.

    Parameters
    ----------
    df:
        pandas DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` and a DatetimeIndex.  Assumed to be sorted ascending by date.

    Returns
    -------
    pandas DataFrame with a DatetimeIndex and columns:
        logret_open, logret_high, logret_low, logret_close, logdiff_volume.
    Row 0 will have NaN for the log-return columns (no previous bar).
    """
    prev_close = df["Close"].shift(1)
    prev_volume = df["Volume"].shift(1)

    # OHLC-contract-compatible features
    high_margin = df["High"] - np.maximum(df["Open"], df["Close"])
    low_margin = np.minimum(df["Open"], df["Close"]) - df["Low"]

    out = pd.DataFrame(index=df.index)
    out["logret_open"] = np.log(df["Open"] / prev_close)
    out["logret_high"] = np.log(np.clip(high_margin, 0.0, None) + 1e-8)
    out["logret_low"] = np.log(np.clip(low_margin, 0.0, None) + 1e-8)
    out["logret_close"] = np.log(df["Close"] / prev_close)
    out["logdiff_volume"] = np.log(df["Volume"]) - np.log(prev_volume)

    return out


def split_eras(
    feats: pd.DataFrame,
    *,
    val_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a feature DataFrame into (train, val, eval) era slices.

    Parameters
    ----------
    feats:
        Feature DataFrame with DatetimeIndex (from :func:`compute_features`).
    val_start, train_end, eval_start, eval_end:
        ISO date strings defining the era boundaries (inclusive).

    Returns
    -------
    (train_df, val_df, eval_df)
        - train: dates < val_start  (strictly before the validation window)
        - val:   val_start <= dates <= train_end
        - eval:  eval_start <= dates <= eval_end
    """
    idx = feats.index

    val_start_ts = pd.Timestamp(val_start)
    train_end_ts = pd.Timestamp(train_end)
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts = pd.Timestamp(eval_end)

    train = feats[idx < val_start_ts]
    val = feats[(idx >= val_start_ts) & (idx <= train_end_ts)]
    eval_ = feats[(idx >= eval_start_ts) & (idx <= eval_end_ts)]

    return train, val, eval_


def compute_stats(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-feature mean and std from the training split.

    Parameters
    ----------
    train_df:
        Training-split feature DataFrame (output of :func:`split_eras`[0]).

    Returns
    -------
    dict mapping feature name -> {"mean": float, "std": float}.
    Uses ddof=1 (sample std) for consistency with sklearn's StandardScaler.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in train_df.columns:
        series = train_df[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
        }
    return stats


def should_skip(ticker: str, cache_dir: Path, *, force: bool = False) -> bool:
    """Return True if the ticker's parquet file already exists and force is False."""
    parquet_path = cache_dir / f"{ticker}.parquet"
    return parquet_path.exists() and not force


def _download_ticker(
    ticker: str,
    *,
    start: str,
    auto_adjust: bool = True,
) -> pd.DataFrame | None:
    """Download OHLCV from yfinance.  Returns None if no data found.

    Extracted as a standalone function so tests can monkeypatch it.
    """
    import yfinance as yf  # lazy import — not needed for unit tests

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            start=start,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )

    if df is None or df.empty:
        return None

    # yfinance may return multi-level columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def ingest_ticker(
    ticker: str,
    *,
    cache_dir: Path,
    stats_dir: Path,
    start_date: str,
    val_start_date: str,
    train_end_date: str,
    eval_start_date: str,
    eval_end_date: str,
    force: bool = False,
) -> bool:
    """Download, feature-engineer, split, and cache one ticker.

    Returns True on success, False if skipped or no data.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if should_skip(ticker, cache_dir, force=force):
        logger.info("Skipping %s — parquet exists (use --force to re-download)", ticker)
        return False

    raw = _download_ticker(ticker, start=start_date)
    if raw is None or raw.empty:
        logger.warning("No data for %s — skipping", ticker)
        return False

    feats = compute_features(raw)
    feats = feats.iloc[1:]  # drop row 0 (NaN log-returns)

    train, _val, _eval = split_eras(
        feats,
        val_start=val_start_date,
        train_end=train_end_date,
        eval_start=eval_start_date,
        eval_end=eval_end_date,
    )

    if train.empty:
        logger.warning("No training data for %s — skipping", ticker)
        return False

    stats = compute_stats(train)

    # Write feature parquet (full history, not standardized)
    pl_df = (
        pl.from_pandas(feats.reset_index().rename(columns={"index": "date"}))
        .with_columns(pl.col("date").cast(pl.Date))
    )
    pl_df.write_parquet(cache_dir / f"{ticker}.parquet")

    # Write stats parquet
    stats_rows = [
        {"feature": feat, "mean": vals["mean"], "std": vals["std"]}
        for feat, vals in stats.items()
    ]
    pl.DataFrame(stats_rows).write_parquet(stats_dir / f"{ticker}_stats.parquet")

    logger.info("Ingested %s: %d bars", ticker, len(feats))
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for `finpred-ingest` / `make ingest`."""
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV to the Parquet cache.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if Parquet already exists.",
    )
    args = parser.parse_args()

    import yaml  # noqa: PLC0415

    with open(args.config) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Resolve `extends:` if present (simple one-level merge for now)
    if "extends" in cfg:
        base_path = Path(args.config).parent / cfg.pop("extends")
        with open(base_path) as fh:
            base_cfg: dict[str, Any] = yaml.safe_load(fh)
        _deep_merge(base_cfg, cfg)
        cfg = base_cfg

    data_cfg = cfg["data"]
    tickers = resolve_tickers(
        data_cfg["tickers"],
        extra_tickers=data_cfg.get("extra_tickers", []),
    )

    cache_dir = Path(data_cfg["cache_dir"])
    stats_dir = cache_dir / "stats"
    start_date: str = data_cfg["start_date"]
    val_start: str = data_cfg["val_start_date"]
    train_end: str = data_cfg["train_end_date"]
    eval_start: str = data_cfg["eval_start_date"]
    eval_end: str = data_cfg["eval_end_date"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Ingesting %d tickers into %s", len(tickers), cache_dir)

    n_ok = 0
    for ticker in tickers:
        ok = ingest_ticker(
            ticker,
            cache_dir=cache_dir,
            stats_dir=stats_dir,
            start_date=start_date,
            val_start_date=val_start,
            train_end_date=train_end,
            eval_start_date=eval_start,
            eval_end_date=eval_end,
            force=args.force,
        )
        if ok:
            n_ok += 1

    logger.info("Done. %d / %d tickers ingested successfully.", n_ok, len(tickers))


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge `override` into `base` in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


if __name__ == "__main__":
    main()

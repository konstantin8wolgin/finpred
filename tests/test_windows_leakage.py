"""Era-leakage tests for src/finpred/data/windows.py.

Five test functions guard the single most important correctness property: no date from
outside a split's allowed range may enter the resulting tensors.

test_train_split_no_future_bars
    Every bar yielded by WindowDataset(cfg, split="train") must have date < val_start_date
    (2012-01-01).  No bar in [val_start_date, train_end_date] may appear in a training batch.

test_val_split_within_bounds
    Every bar yielded by WindowDataset(cfg, split="val") must satisfy
    val_start_date <= date <= train_end_date.

test_eval_split_within_bounds
    Every bar yielded by WindowDataset(cfg, split="eval") must satisfy
    eval_start_date <= date <= eval_end_date.

test_no_cross_contamination
    A single pass over all three splits produces disjoint date sets (no date appears in more
    than one split for a given ticker).

test_window_boundary_exact
    A window whose last bar is exactly train_end_date - 1 day is legal for split="train".
    A window whose first bar equals val_start_date is illegal for split="train".
    Assert both.
"""
from __future__ import annotations

import types
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from finpred.data.windows import WindowDataset, fetch_real_window

# ---------------------------------------------------------------------------
# Era boundary constants — must match configs/era_2012.yaml / default.yaml
# ---------------------------------------------------------------------------
VAL_START = date(2012, 1, 1)
TRAIN_END = date(2012, 6, 30)
EVAL_START = date(2012, 7, 1)
EVAL_END = date(2019, 12, 31)

TICKERS = ["AAAA", "BBBB", "CCCC"]
WINDOW_LENGTH = 20   # short window so tests run fast
STRIDE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path) -> types.SimpleNamespace:
    """Build a minimal config namespace that points at tmp_path."""
    data_ns = types.SimpleNamespace(
        tickers=TICKERS,
        cache_dir=str(tmp_path),
        train_end_date=str(TRAIN_END),
        val_start_date=str(VAL_START),
        eval_start_date=str(EVAL_START),
        eval_end_date=str(EVAL_END),
    )
    windows_ns = types.SimpleNamespace(
        lengths=[WINDOW_LENGTH],
        stride=STRIDE,
        stratify_by="ticker",
    )
    return types.SimpleNamespace(data=data_ns, windows=windows_ns)


def _weekdays_in_range(start: date, end: date) -> list[date]:
    """Return Mon-Fri dates in [start, end]."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _write_ticker_parquet(tmp_path, ticker: str, dates: list[date]) -> None:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    n = len(dates)
    df = pl.DataFrame(
        {
            "date": dates,
            "logret_open": rng.standard_normal(n) * 0.01,
            "logret_high": rng.standard_normal(n) * 0.005,
            "logret_low": -(np.abs(rng.standard_normal(n)) * 0.005),
            "logret_close": rng.standard_normal(n) * 0.01,
            "logdiff_volume": rng.standard_normal(n) * 0.1,
        }
    )
    df.write_parquet(tmp_path / f"{ticker}.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cache(tmp_path):
    """Write synthetic parquet files for TICKERS covering 1990-01-02..2020-12-31."""
    all_dates = _weekdays_in_range(date(1990, 1, 2), date(2020, 12, 31))
    for ticker in TICKERS:
        _write_ticker_parquet(tmp_path, ticker, all_dates)
    return tmp_path


@pytest.fixture
def cfg(synthetic_cache):
    return _make_cfg(synthetic_cache)


# ---------------------------------------------------------------------------
# Utility: collect all dates from a WindowDataset
# ---------------------------------------------------------------------------

def _collect_dates(cfg, split: str, length: int = WINDOW_LENGTH) -> list[date]:
    """Exhaust a WindowDataset with return_dates=True and collect every date seen."""
    ds = WindowDataset(cfg, split=split, length=length, return_dates=True)
    all_dates: list[date] = []
    for _tensor, window_dates in ds:
        all_dates.extend(window_dates)
    return all_dates


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_split_no_future_bars(cfg):
    """No bar in the train split may be >= val_start_date (2012-01-01)."""
    dates = _collect_dates(cfg, "train")
    assert len(dates) > 0, "train split yielded no bars at all — fixture issue?"
    bad = [d for d in dates if d >= VAL_START]
    assert bad == [], (
        f"train split contains {len(bad)} bar(s) with date >= {VAL_START}. "
        f"First bad date: {bad[0]}"
    )


def test_val_split_within_bounds(cfg):
    """Every bar in the val split must satisfy val_start_date <= date <= train_end_date."""
    dates = _collect_dates(cfg, "val")
    assert len(dates) > 0, "val split yielded no bars — fixture issue?"
    too_early = [d for d in dates if d < VAL_START]
    too_late = [d for d in dates if d > TRAIN_END]
    assert too_early == [], (
        f"val split contains {len(too_early)} bar(s) before {VAL_START}. "
        f"First: {too_early[0]}"
    )
    assert too_late == [], (
        f"val split contains {len(too_late)} bar(s) after {TRAIN_END}. "
        f"First: {too_late[0]}"
    )


def test_eval_split_within_bounds(cfg):
    """Every bar in the eval split must satisfy eval_start_date <= date <= eval_end_date."""
    dates = _collect_dates(cfg, "eval")
    assert len(dates) > 0, "eval split yielded no bars — fixture issue?"
    too_early = [d for d in dates if d < EVAL_START]
    too_late = [d for d in dates if d > EVAL_END]
    assert too_early == [], (
        f"eval split contains {len(too_early)} bar(s) before {EVAL_START}. "
        f"First: {too_early[0]}"
    )
    assert too_late == [], (
        f"eval split contains {len(too_late)} bar(s) after {EVAL_END}. "
        f"First: {too_late[0]}"
    )


def test_no_cross_contamination(cfg):
    """The three splits must have disjoint date sets (per ticker)."""
    train_set = set(_collect_dates(cfg, "train"))
    val_set = set(_collect_dates(cfg, "val"))
    eval_set = set(_collect_dates(cfg, "eval"))

    train_val_overlap = train_set & val_set
    train_eval_overlap = train_set & eval_set
    val_eval_overlap = val_set & eval_set

    assert train_val_overlap == set(), (
        f"train and val share {len(train_val_overlap)} date(s). "
        f"Sample: {sorted(train_val_overlap)[:5]}"
    )
    assert train_eval_overlap == set(), (
        f"train and eval share {len(train_eval_overlap)} date(s). "
        f"Sample: {sorted(train_eval_overlap)[:5]}"
    )
    assert val_eval_overlap == set(), (
        f"val and eval share {len(val_eval_overlap)} date(s). "
        f"Sample: {sorted(val_eval_overlap)[:5]}"
    )


def test_window_boundary_exact(tmp_path):
    """
    Boundary precision test using a single-ticker dataset with known dates.

    Scenario A (legal for train):
        Window ends on TRAIN_END - 1 business day.  This window is entirely in the
        pre-val era, so it must appear when split="train".

    Scenario B (illegal for train):
        Window whose *first* bar is VAL_START.  That date is in [val_start, train_end],
        not in the train era, so it must NOT appear when split="train".
    """
    # Build a tiny date grid around the boundary.
    # We want a contiguous block that ends just before VAL_START and another that starts at VAL_START.
    # Window length for this test: 5 bars.
    win_len = 5

    # Generate weekday dates from (VAL_START - 30 days) to (VAL_START + 30 days).
    grid_start = VAL_START - timedelta(days=30)
    grid_end = VAL_START + timedelta(days=30)
    all_dates = _weekdays_in_range(grid_start, grid_end)

    cfg = _make_cfg(tmp_path)
    _write_ticker_parquet(tmp_path, TICKERS[0], all_dates)
    # Only one ticker for this test.
    cfg.data.tickers = [TICKERS[0]]
    cfg.windows.lengths = [win_len]
    cfg.windows.stride = 1  # stride 1 so we get every possible window

    # Collect all train windows with return_dates=True, record sets of (first_date, last_date)
    ds_train = WindowDataset(cfg, split="train", length=win_len, return_dates=True)
    train_windows: list[tuple[date, date]] = []
    for _tensor, dates in ds_train:
        train_windows.append((dates[0], dates[-1]))

    train_last_dates = {last for (_first, last) in train_windows}
    train_first_dates = {first for (first, _last) in train_windows}

    # Scenario A: find the last weekday strictly before VAL_START.
    day = VAL_START - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    last_train_day = day

    # That day must exist as a window-end in the train split (provided there are enough
    # preceding bars, which our grid guarantees).
    assert last_train_day in train_last_dates, (
        f"Expected a train window ending on {last_train_day} (the last pre-val day), "
        f"but none was found.  Last dates seen: {sorted(train_last_dates)}"
    )

    # Scenario B: VAL_START must NOT appear as a window-start in the train split
    # (because that would mean the first bar of the window is already in the val era).
    assert VAL_START not in train_first_dates, (
        f"A train window starts on {VAL_START} (= val_start_date), which is illegal. "
        f"This is a leakage bug."
    )
    # More strongly: VAL_START must not appear anywhere in the train split.
    all_train_dates: set[date] = set()
    ds_train2 = WindowDataset(cfg, split="train", length=win_len, return_dates=True)
    for _tensor, window_dates in ds_train2:
        all_train_dates.update(window_dates)
    assert VAL_START not in all_train_dates, (
        f"{VAL_START} (val_start_date) appears in a training window — era leakage detected."
    )

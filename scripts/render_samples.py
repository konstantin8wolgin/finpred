"""Render real-vs-generated candlestick PNGs across training checkpoints.

For each checkpoint in checkpoints/<run_id>/ (sorted by step), generates N fake windows
from G and pulls N real eval windows, rendering both sets side-by-side as PNGs.

Output layout:
    reports/<run_id>/samples/
        step_000500/
            fake_0.png  fake_1.png  ...
            real_0.png  real_1.png  ...
        step_001000/
            ...

Usage:
    uv run python scripts/render_samples.py --config configs/default.yaml
    uv run python scripts/render_samples.py --config configs/era_2012.yaml --run_id 20260512_155119_0 --n_samples 4
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)

# Preference order for real-sample ticker (must be in data/ directory).
_PREFERRED_TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]


def _latest_run_id(ckpt_root: Path) -> str | None:
    """Return the run_id whose checkpoint dir has the most .pt files."""
    best_id, best_count = None, -1
    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        count = len(list(d.glob("*.pt")))
        if count > best_count:
            best_count = count
            best_id = d.name
    return best_id


def _load_ticker_stats(stats_dir: Path, ticker: str) -> dict:
    """Load per-ticker mean/std as {'mean': np.ndarray(5,), 'std': np.ndarray(5,)}."""
    path = stats_dir / f"{ticker}_stats.parquet"
    df = pl.read_parquet(path)
    feature_order = [
        "logret_open", "logret_high", "logret_low", "logret_close", "logdiff_volume"
    ]
    row_by_feat = {row["feature"]: row for row in df.iter_rows(named=True)}
    mean = np.array([row_by_feat[f]["mean"] for f in feature_order], dtype=float)
    std  = np.array([row_by_feat[f]["std"]  for f in feature_order], dtype=float)
    return {"mean": mean, "std": std}


def _pick_ticker(cache_dir: Path) -> str | None:
    """Pick the best available ticker for real eval samples."""
    for t in _PREFERRED_TICKERS:
        if (cache_dir / f"{t}.parquet").exists():
            return t
    parquets = [p.stem for p in cache_dir.glob("*.parquet")]
    return parquets[0] if parquets else None


def _real_eval_windows(
    cfg,
    ticker: str,
    length: int,
    n_samples: int,
) -> list[np.ndarray]:
    """Pull up to n_samples eval windows for ticker."""
    from finpred.data.windows import WindowDataset

    ds = WindowDataset(cfg, split="eval", length=length)
    windows = []
    for i, item in enumerate(ds):
        if len(windows) >= n_samples:
            break
        t = item if not isinstance(item, tuple) else item[0]
        windows.append(t.numpy())
    return windows


def _render_checkpoint(
    ckpt_path: Path,
    cfg,
    ticker: str,
    ticker_stats: dict,
    out_dir: Path,
    n_samples: int,
) -> None:
    """Load one checkpoint, generate fakes + pull reals, render all."""
    from finpred.models.generator import Generator
    from finpred.render.ohlcv_to_png import ohlcv_to_png

    T = cfg.windows.lengths[0]
    device = cfg.device

    G = Generator(
        z_dim=cfg.model.z_dim,
        hidden_channels=cfg.model.hidden_channels,
        n_blocks=cfg.model.n_blocks,
        kernel_size=cfg.model.kernel_size,
        dilations=cfg.model.dilations,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Unwrap compiled-model prefix if present.
    state = ckpt["G"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    G.load_state_dict(state)
    G.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fakes ---
    with torch.no_grad():
        z = torch.randn(n_samples, T, cfg.model.z_dim, device=device)
        fakes = G(z).cpu().numpy()  # (n_samples, T, 5)

    for i, seq in enumerate(fakes):
        ohlcv_to_png(
            seq,
            out_dir / f"fake_{i}.png",
            ticker_stats=ticker_stats,
            title=f"Fake {i} — {ckpt_path.stem}",
        )

    # --- Reals ---
    real_windows = _real_eval_windows(cfg, ticker, T, n_samples)
    if not real_windows:
        logger.warning("No eval windows found for ticker %s — skipping real PNGs", ticker)
        return

    for i, seq in enumerate(real_windows):
        ohlcv_to_png(
            seq,
            out_dir / f"real_{i}.png",
            ticker_stats=ticker_stats,
            title=f"Real {ticker} {i} — {ckpt_path.stem}",
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Render real-vs-generated PNGs across checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    from finpred.config.schema import load_config
    from finpred.data.ingest import resolve_tickers

    cfg = load_config(args.config)
    resolved = resolve_tickers(cfg.data.tickers, extra_tickers=list(cfg.data.extra_tickers))
    cfg.data.tickers = resolved

    ckpt_root = Path("checkpoints")
    run_id = args.run_id or _latest_run_id(ckpt_root)
    if run_id is None:
        logger.error("No checkpoint directories found under %s", ckpt_root)
        raise SystemExit(1)

    ckpt_dir = ckpt_root / run_id
    ckpt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not ckpt_files:
        logger.error("No .pt files in %s", ckpt_dir)
        raise SystemExit(1)

    cache_dir = Path(cfg.data.cache_dir)
    stats_dir = cache_dir / "stats"
    ticker = _pick_ticker(cache_dir)
    if ticker is None:
        logger.error("No parquet files found in %s", cache_dir)
        raise SystemExit(1)

    ticker_stats = _load_ticker_stats(stats_dir, ticker)
    logger.info("Using ticker %s for real eval samples", ticker)

    samples_root = Path(f"reports/{run_id}/samples")

    for ckpt_path in ckpt_files:
        step_label = ckpt_path.stem  # e.g. "step_000500"
        out_dir = samples_root / step_label
        logger.info("Rendering checkpoint %s -> %s", ckpt_path.name, out_dir)
        _render_checkpoint(ckpt_path, cfg, ticker, ticker_stats, out_dir, args.n_samples)

    logger.info("Done. PNGs written to %s", samples_root)


if __name__ == "__main__":
    main()

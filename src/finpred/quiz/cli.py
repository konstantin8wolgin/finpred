"""`finpred-quiz` — pose the N-way pick-the-real quiz to a human and/or the trained model.

Phase A.1: CLI skeleton with config loading, --dry-run, and --n-way option.
Phase B will wire this to a real checkpoint and the discriminator.

Behaviour (when fully implemented):
  - load config (+ optional --checkpoint),
  - build one episode: pick a real window from the held-out era (default ticker NVDA),
    generate N-1 fakes from G (length-matched, start-price-aligned),
  - render all N to `quiz.out_dir` with identical axes/date formatting, shuffled, labelled 1..N,
  - prompt the human for a guess; if a checkpoint is given, also report the model's pick and
    each candidate's discriminator score; reveal the answer.

Uses typer; `app` is the entry point referenced in pyproject.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
import yaml

app = typer.Typer(add_completion=False, help="N-way pick-the-real stock-chart quiz.")


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config, resolving a one-level `extends:` parent via deep merge."""
    path = Path(config_path)
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if "extends" in cfg:
        base_path = path.parent / cfg.pop("extends")
        with open(base_path) as fh:
            base_cfg: dict[str, Any] = yaml.safe_load(fh)
        _deep_merge(base_cfg, cfg)
        cfg = base_cfg

    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge `override` into `base` in-place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


@app.command()
def main(
    config: str = typer.Option(..., "--config", help="Path to YAML config file."),
    checkpoint: str | None = typer.Option(None, "--checkpoint", help="Path to model checkpoint."),
    ticker: str = typer.Option("NVDA", "--ticker", help="Real-chart ticker for the quiz."),
    n_way: int = typer.Option(0, "--n-way", help="N for N-way quiz (0 = use config value)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print resolved config and exit."),
) -> None:
    """Run one N-way quiz episode.

    Phase A.1: config loading and CLI interface are fully wired.
    The core quiz logic (rendering + discriminator) will be wired in Phase B.
    """
    # Try to use the proper config loader; fall back to raw YAML if not yet implemented.
    cfg: dict[str, Any]
    try:
        from finpred.config.schema import load_config  # noqa: PLC0415

        cfg_obj = load_config(config)
        # If load_config returns a Pydantic model, convert to dict for uniform access.
        if hasattr(cfg_obj, "model_dump"):
            cfg = cfg_obj.model_dump()
        elif hasattr(cfg_obj, "dict"):
            cfg = cfg_obj.dict()
        else:
            cfg = dict(cfg_obj)  # type: ignore[arg-type]
    except NotImplementedError:
        # config.schema not yet implemented — fall back to raw YAML
        cfg = _load_yaml_config(config)

    # Resolve effective n_way: CLI flag overrides config, config overrides default.
    effective_n_way = n_way if n_way > 0 else cfg.get("quiz", {}).get("n_way", 5)

    if dry_run:
        typer.echo("Config loaded successfully (dry-run mode).")
        typer.echo(f"  config path : {config}")
        typer.echo(f"  n_way       : {effective_n_way}")
        typer.echo(f"  ticker      : {ticker}")
        typer.echo(f"  checkpoint  : {checkpoint}")
        typer.echo(f"  seed        : {cfg.get('seed', 'not set')}")
        typer.echo(f"  quiz.out_dir: {cfg.get('quiz', {}).get('out_dir', 'not set')}")
        raise typer.Exit(0)

    import random as _random
    import re
    import sys
    import numpy as np
    import polars as pl
    import torch
    from pathlib import Path
    from finpred.config.schema import load_config
    from finpred.data.ingest import resolve_tickers
    from finpred.data.windows import WindowDataset, _slice_split, _load_ticker_df
    from finpred.models.generator import Generator
    from finpred.models.discriminator import Discriminator
    from finpred.render.ohlcv_to_png import ohlcv_to_png, reconstruct_ohlcv

    cfg_obj = load_config(config)
    resolved = resolve_tickers(cfg_obj.data.tickers, extra_tickers=list(cfg_obj.data.extra_tickers))
    cfg_obj.data.tickers = resolved

    # Resolve checkpoint: CLI arg or latest.
    if checkpoint is None:
        ckpt_root = Path("checkpoints")
        best_dir = max(
            (d for d in ckpt_root.iterdir() if d.is_dir()),
            key=lambda d: len(list(d.glob("*.pt"))),
            default=None,
        )
        if best_dir is None:
            typer.echo("No checkpoint found. Run `make train` first.", err=True)
            raise typer.Exit(1)
        pts = sorted(best_dir.glob("*.pt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
        ckpt_path = pts[-1]
    else:
        ckpt_path = Path(checkpoint)
    typer.echo(f"Checkpoint: {ckpt_path}")

    device = cfg_obj.device
    G = Generator(
        z_dim=cfg_obj.model.z_dim,
        hidden_channels=cfg_obj.model.hidden_channels,
        n_blocks=cfg_obj.model.n_blocks,
        kernel_size=cfg_obj.model.kernel_size,
        dilations=cfg_obj.model.dilations,
    ).to(device)
    D = Discriminator(
        hidden_channels=cfg_obj.model.hidden_channels,
        n_blocks=cfg_obj.model.n_blocks,
        kernel_size=cfg_obj.model.kernel_size,
        dilations=cfg_obj.model.dilations,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for model, key in [(G, "G"), (D, "D")]:
        state = ckpt[key]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
    G.eval()
    D.eval()

    T = cfg_obj.windows.lengths[0]
    effective_n_way = n_way if n_way > 0 else cfg_obj.quiz.n_way

    # Pick a real eval window for the ticker.
    cache_dir = Path(cfg_obj.data.cache_dir)
    df = _load_ticker_df(cache_dir, ticker)
    if df is None:
        typer.echo(f"No data for ticker {ticker}", err=True)
        raise typer.Exit(1)
    sliced = _slice_split(df, "eval", cfg_obj)
    if len(sliced) < T:
        typer.echo(f"Not enough eval bars for {ticker} (need {T}, have {len(sliced)})", err=True)
        raise typer.Exit(1)
    # Pick a random start position.
    max_start = len(sliced) - T
    start = _random.randint(0, max_start)
    from finpred.data.windows import FEATURE_COLS
    real_array = sliced.select(FEATURE_COLS).to_numpy()[start:start + T]
    real_tensor = torch.tensor(real_array, dtype=torch.float32).to(device)

    # Load ticker stats for rendering.
    stats_path = cache_dir / "stats" / f"{ticker}_stats.parquet"
    stats_df = pl.read_parquet(stats_path)
    feat_order = ["logret_open", "logret_high", "logret_low", "logret_close", "logdiff_volume"]
    row_by_feat = {row["feature"]: row for row in stats_df.iter_rows(named=True)}
    ticker_stats = {
        "mean": np.array([row_by_feat[f]["mean"] for f in feat_order]),
        "std": np.array([row_by_feat[f]["std"] for f in feat_order]),
    }

    # Generate fakes and align starting price to real.
    with torch.no_grad():
        z = torch.randn(effective_n_way - 1, T, cfg_obj.model.z_dim, device=device)
        fakes = G(z)  # (n_way-1, T, 5)
        # D scores for all candidates.
        candidates_tensor = torch.cat([real_tensor.unsqueeze(0), fakes], dim=0)
        scores = D(candidates_tensor).cpu().numpy()  # (n_way,)

    # Reconstruct real to get its starting close price.
    real_df = reconstruct_ohlcv(real_array, ticker_stats, start_price=100.0)
    real_start_price = float(real_df["Close"].iloc[0])

    # Build candidate list: (array, label_type, d_score).
    candidates = [(real_array, "real", float(scores[0]))]
    for i in range(effective_n_way - 1):
        candidates.append((fakes[i].cpu().numpy(), "fake", float(scores[i + 1])))

    # Shuffle so human doesn't know real=index 0.
    _random.shuffle(candidates)
    real_position = next(i for i, (_, t, _) in enumerate(candidates) if t == "real") + 1

    # Compute shared y-axis range over all reconstructed charts.
    all_closes = []
    for arr, _, _ in candidates:
        tmp_df = reconstruct_ohlcv(arr, ticker_stats, start_price=real_start_price)
        all_closes.extend(tmp_df["High"].tolist())
        all_closes.extend(tmp_df["Low"].tolist())
    y_lo = min(all_closes) * 0.95
    y_hi = max(all_closes) * 1.05

    # Render all N charts.
    out_dir = Path(cfg_obj.quiz.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths = []
    for i, (arr, _, _) in enumerate(candidates, start=1):
        out_path = out_dir / f"chart_{i}.png"
        ohlcv_to_png(
            arr, out_path,
            start_price=real_start_price,
            ticker_stats=ticker_stats,
            y_range=(y_lo, y_hi),
            title=f"Chart {i}",
        )
        png_paths.append(out_path)

    typer.echo(f"\n{effective_n_way} charts saved to {out_dir}")
    for p in png_paths:
        typer.echo(f"  {p}")
    typer.echo(f"\nOne of these is a real {ticker} chart from 2012-2019.")
    typer.echo("Which chart is real? Enter a number (1–{n}):".format(n=effective_n_way))

    try:
        guess = int(input("> ").strip())
    except (ValueError, EOFError):
        guess = -1

    typer.echo(f"\nThe real chart was: Chart {real_position}")
    if guess == real_position:
        typer.echo("Correct!")
    else:
        typer.echo(f"Wrong. You picked {guess}.")

    typer.echo("\nDiscriminator scores (higher = D thinks it's more real):")
    for i, (_, label, score) in enumerate(candidates, start=1):
        marker = "<-- real" if label == "real" else ""
        typer.echo(f"  Chart {i}: {score:+.2f}  {marker}")


if __name__ == "__main__":
    app()

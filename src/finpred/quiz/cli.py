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

    # Phase B wires the real quiz logic here.
    raise NotImplementedError(
        "quiz.cli.main — core quiz logic will be implemented in Phase B. "
        "Run with --dry-run to verify the CLI works without a checkpoint."
    )


if __name__ == "__main__":
    app()

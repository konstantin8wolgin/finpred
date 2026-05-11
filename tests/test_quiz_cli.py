"""Tests for src/finpred/quiz/cli.py (Phase A.1 CLI skeleton)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from finpred.quiz.cli import app

runner = CliRunner()


def _write_minimal_config(path: Path) -> Path:
    """Write a minimal valid config YAML and return its path."""
    cfg = {
        "seed": 0,
        "deterministic": False,
        "device": "cpu",
        "data": {
            "source": "yfinance",
            "tickers": ["NVDA"],
            "start_date": "1990-01-01",
            "train_end_date": "2012-06-30",
            "val_start_date": "2012-01-01",
            "eval_start_date": "2012-07-01",
            "eval_end_date": "2019-12-31",
            "cache_dir": str(path / "data"),
            "features": [
                "logret_open",
                "logret_high",
                "logret_low",
                "logret_close",
                "logdiff_volume",
            ],
        },
        "windows": {"lengths": [252], "stride": 21, "stratify_by": "ticker"},
        "model": {
            "z_dim": 3,
            "hidden_channels": 32,
            "n_blocks": 4,
            "kernel_size": 5,
            "dilations": [1, 2, 4, 8],
        },
        "train": {
            "steps": 100,
            "batch_size": 4,
            "grad_accum": 1,
            "n_critic": 5,
            "lr": 1e-4,
            "betas": [0.5, 0.9],
            "gp_lambda": 10.0,
            "bf16": False,
            "compile": False,
            "ckpt_every": 50,
            "eval_every": 50,
        },
        "eval": {"n_way": 5, "n_episodes": 10, "stylized_facts": {"acf_lags": [1, 5, 22]}},
        "quiz": {"n_way": 5, "out_dir": str(path / "reports" / "quiz")},
        "promotion": {"min_n_way_acc": 0.45, "min_ks_pvalue": 0.05},
    }
    cfg_path = path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg_path


# ---------------------------------------------------------------------------
# test_dry_run
# ---------------------------------------------------------------------------

def test_dry_run(tmp_path: Path):
    """--dry-run prints config info and exits with code 0 without needing a checkpoint."""
    cfg_path = _write_minimal_config(tmp_path)
    result = runner.invoke(app, ["--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0, f"Non-zero exit: {result.output}\n{result.exception}"
    # Some config-related text should appear in the output
    output = result.output
    assert any(
        token in output
        for token in ("config", "quiz", "dry", "n_way", "seed", "Config")
    ), f"Expected config info in output, got:\n{output}"


# ---------------------------------------------------------------------------
# test_missing_config_fails
# ---------------------------------------------------------------------------

def test_missing_config_fails():
    """Invoking without --config should fail with a non-zero exit code."""
    result = runner.invoke(app, [])
    assert result.exit_code != 0, (
        f"Expected non-zero exit when --config is missing, got 0\nOutput: {result.output}"
    )


# ---------------------------------------------------------------------------
# test_normal_run_without_checkpoint_raises_clear_error
# ---------------------------------------------------------------------------

def test_normal_run_without_checkpoint_raises_clear_error(tmp_path: Path):
    """Running without --checkpoint and without --dry-run should raise NotImplementedError
    (Phase B wires the real logic) — the CLI should surface this, not silently succeed."""
    cfg_path = _write_minimal_config(tmp_path)
    result = runner.invoke(app, ["--config", str(cfg_path)])
    # Either a non-zero exit code or the output contains a 'not implemented' message.
    output_lower = result.output.lower()
    has_not_implemented = (
        "not implement" in output_lower
        or "phase b" in output_lower
        or result.exit_code != 0
    )
    assert has_not_implemented, (
        f"Expected NotImplementedError or non-zero exit.\nExit: {result.exit_code}\n"
        f"Output: {result.output}"
    )


# ---------------------------------------------------------------------------
# test_n_way_option
# ---------------------------------------------------------------------------

def test_n_way_option(tmp_path: Path):
    """--n-way option is accepted without errors (even in dry-run mode)."""
    cfg_path = _write_minimal_config(tmp_path)
    result = runner.invoke(app, ["--config", str(cfg_path), "--n-way", "3", "--dry-run"])
    assert result.exit_code == 0, f"Non-zero exit with --n-way: {result.output}"

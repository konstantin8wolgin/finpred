"""Tests for train.wgan_gp — gradient penalty and training loop smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from finpred.models.discriminator import Discriminator
from finpred.models.generator import Generator

_DILATIONS_SMALL = [1, 2]
_T_SMALL = 16
_BATCH = 2


def _make_d_small() -> Discriminator:
    return Discriminator(
        hidden_channels=8,
        n_blocks=2,
        kernel_size=3,
        dilations=_DILATIONS_SMALL,
    )


def _make_real_fake() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    real = torch.randn(_BATCH, _T_SMALL, 5)
    fake = torch.randn(_BATCH, _T_SMALL, 5)
    return real, fake


# ---------------------------------------------------------------------------
# gradient_penalty tests
# ---------------------------------------------------------------------------


def test_gradient_penalty_shape() -> None:
    from finpred.train.wgan_gp import gradient_penalty

    d = _make_d_small()
    real, fake = _make_real_fake()
    gp = gradient_penalty(d, real, fake, device="cpu")
    assert gp.shape == torch.Size([]), f"Expected scalar, got shape {gp.shape}"


def test_gradient_penalty_positive() -> None:
    from finpred.train.wgan_gp import gradient_penalty

    torch.manual_seed(42)
    d = _make_d_small()
    for _ in range(5):
        real = torch.randn(_BATCH, _T_SMALL, 5)
        fake = torch.randn(_BATCH, _T_SMALL, 5)
        gp = gradient_penalty(d, real, fake, device="cpu")
        assert gp.item() >= 0.0, f"GP must be non-negative, got {gp.item()}"


# ---------------------------------------------------------------------------
# Synthetic data helpers for smoke tests
# ---------------------------------------------------------------------------


def _write_synthetic_parquets(data_dir: Path, n_tickers: int = 4, n_rows: int = 320) -> list[str]:
    """Write tiny synthetic parquets with pre-2012-01-01 dates into *data_dir*."""
    import numpy as np

    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=7)
    tickers = [f"T{i}" for i in range(n_tickers)]

    # Use dates well before val_start so they land in the 'train' split
    # (train split: date < val_start_date = "2011-01-01" in our test config)
    base_date = pl.date_range(
        pl.date(2007, 1, 1), pl.date(2010, 12, 31), interval="1d", eager=True
    ).filter(pl.Series([d.weekday() < 5 for d in pl.date_range(pl.date(2007, 1, 1), pl.date(2010, 12, 31), interval="1d", eager=True).to_list()]))

    for ticker in tickers:
        n = min(n_rows, len(base_date))
        dates = base_date[:n]
        df = pl.DataFrame(
            {
                "date": dates,
                "logret_open": rng.normal(0, 0.01, n).tolist(),
                "logret_high": rng.normal(0, 0.005, n).tolist(),
                "logret_low": rng.normal(0, 0.005, n).tolist(),
                "logret_close": rng.normal(0, 0.01, n).tolist(),
                "logdiff_volume": rng.normal(0, 0.1, n).tolist(),
            }
        )
        df.write_parquet(data_dir / f"{ticker}.parquet")

    return tickers


def _make_smoke_config(tmp_path: Path, tickers: list[str], extra: dict | None = None) -> object:
    """Build a minimal Config object for smoke tests pointing at tmp_path data."""
    from finpred.config.schema import Config

    cfg_dict: dict = {
        "seed": 0,
        "deterministic": False,
        "device": "cpu",
        "data": {
            "source": "yfinance",
            "tickers": tickers,
            "start_date": "2007-01-01",
            "train_end_date": "2011-06-30",
            "val_start_date": "2011-01-01",
            "eval_start_date": "2011-07-01",
            "eval_end_date": "2012-12-31",
            "cache_dir": str(tmp_path / "data"),
        },
        "windows": {
            "lengths": [32],
            "stride": 8,
        },
        "model": {
            "z_dim": 2,
            "hidden_channels": 8,
            "n_blocks": 2,
            "kernel_size": 3,
            "dilations": [1, 2],
        },
        "train": {
            "steps": 3,
            "batch_size": 2,
            "grad_accum": 1,
            "n_critic": 2,
            "lr": 1e-4,
            "betas": [0.5, 0.9],
            "gp_lambda": 10.0,
            "bf16": False,
            "compile": False,
            # Large enough that ckpt/eval are NOT triggered during the 3-step run.
            "ckpt_every": 10,
            "eval_every": 10,
        },
        "eval": {
            "n_way": 3,
            "n_episodes": 5,
            "stylized_facts": {"acf_lags": [1]},
        },
        "quiz": {"n_way": 3, "out_dir": str(tmp_path / "reports" / "quiz")},
        "promotion": {"min_n_way_acc": 0.45, "min_ks_pvalue": 0.05},
    }
    if extra:
        cfg_dict.update(extra)
    return Config.model_validate(cfg_dict)


# ---------------------------------------------------------------------------
# Smoke test: train() completes without error
# ---------------------------------------------------------------------------


def test_train_smoke(tmp_path: Path, monkeypatch) -> None:
    from finpred.train.wgan_gp import train

    # Run from tmp_path so reports/ and checkpoints/ land there, not in the repo root.
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    tickers = _write_synthetic_parquets(data_dir)
    cfg = _make_smoke_config(tmp_path, tickers)
    train(cfg)  # must not raise


# ---------------------------------------------------------------------------
# run.json is created with required keys
# ---------------------------------------------------------------------------


def test_run_json_created(tmp_path: Path, monkeypatch) -> None:
    from finpred.train.wgan_gp import train

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    tickers = _write_synthetic_parquets(data_dir)
    cfg = _make_smoke_config(tmp_path, tickers)
    train(cfg)

    # Locate the run.json under reports/<run_id>/run.json
    # Filter out tb/ and quiz/ sub-dirs — we want the run_id dir with run.json
    run_jsons = list((tmp_path / "reports").glob("*/run.json"))
    assert len(run_jsons) >= 1, "No run.json found under reports/"
    data = json.loads(run_jsons[0].read_text())
    for key in ("seed", "git_sha", "effective_grad_accum"):
        assert key in data, f"Missing key {key!r} in run.json"


# ---------------------------------------------------------------------------
# Checkpoint is created when ckpt_every is small enough
# ---------------------------------------------------------------------------


def test_checkpoint_created(tmp_path: Path, monkeypatch) -> None:
    from finpred.train.wgan_gp import train

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    tickers = _write_synthetic_parquets(data_dir)
    # ckpt_every=2 and steps=3 => checkpoint at step 2
    cfg = _make_smoke_config(tmp_path, tickers, extra={"train": {
        "steps": 3,
        "batch_size": 2,
        "grad_accum": 1,
        "n_critic": 2,
        "lr": 1e-4,
        "betas": [0.5, 0.9],
        "gp_lambda": 10.0,
        "bf16": False,
        "compile": False,
        "ckpt_every": 2,
        "eval_every": 100,
    }})
    train(cfg)

    ckpt_files = list((tmp_path / "checkpoints").glob("**/*.pt"))
    assert len(ckpt_files) >= 1, "No checkpoint .pt file found under checkpoints/"

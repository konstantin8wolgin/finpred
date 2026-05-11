"""Tests for src/finpred/config/schema.py."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from finpred.config.schema import Config, DataConfig, load_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_load_default_yaml():
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.seed == 0
    assert cfg.data.tickers == ["NVDA", "AAPL", "MSFT", "SPY"]
    assert cfg.model.n_blocks == 4
    assert cfg.train.steps == 2000


def test_load_era_2012_yaml():
    cfg = load_config(CONFIGS / "era_2012.yaml")
    assert cfg.seed == 1234
    assert cfg.model.n_blocks == 10
    assert cfg.data.tickers == "__SP1500_PLUS_NASDAQ__"
    assert cfg.train.steps == 80000


def test_extends_merge_keeps_parent_keys():
    """Keys only present in default.yaml must survive after merging era_2012.yaml."""
    cfg = load_config(CONFIGS / "era_2012.yaml")
    # windows.stride is defined in default.yaml and also in era_2012, but the
    # value is the same (21); check the field is present and correct.
    assert cfg.windows.stride == 21
    # data.features is only set explicitly in default.yaml
    assert "logret_close" in cfg.data.features


def test_device_auto_resolves():
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.device in ("cuda", "cpu"), f"Unexpected device: {cfg.device!r}"


def test_invalid_config_raises():
    """Constructing Config with a missing required field must raise ValidationError."""
    with pytest.raises(ValidationError):
        # DataConfig.start_date is required; omitting it (and tickers) triggers validation error
        Config(
            seed=0,
            data=DataConfig(tickers=["NVDA"]),  # missing required date fields
            device="cpu",
        )

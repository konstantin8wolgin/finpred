"""Pydantic config models + YAML loader with `extends:` merge.

Config hierarchy:
- Config (root)
  - DataConfig
  - WindowsConfig
  - ModelConfig
  - TrainConfig
  - EvalConfig
    - StylizedFactsConfig
  - QuizConfig
  - PromotionConfig

load_config(path) resolves an optional `extends:` key by deep-merging the parent YAML
(resolved relative to the child file's directory) then validates the merged dict.

After validation, device="auto" is resolved to "cuda" or "cpu" depending on
torch.cuda.is_available().
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    source: str = "yfinance"
    tickers: list[str] | str  # list or sentinel "__SP1500_PLUS_NASDAQ__"
    extra_tickers: list[str] = []
    start_date: str
    train_end_date: str
    val_start_date: str
    eval_start_date: str
    eval_end_date: str
    cache_dir: str = "data/"
    features: list[str] = [
        "logret_open",
        "logret_high",
        "logret_low",
        "logret_close",
        "logdiff_volume",
    ]


class WindowsConfig(BaseModel):
    lengths: list[int] = [252]
    stride: int = 21
    stratify_by: str = "ticker"


class ModelConfig(BaseModel):
    z_dim: int = 3
    hidden_channels: int = 32
    n_blocks: int = 4
    kernel_size: int = 5
    dilations: list[int] = [1, 2, 4, 8]


class TrainConfig(BaseModel):
    steps: int = 2000
    batch_size: int = 16
    grad_accum: int = 1
    n_critic: int = 5
    lr: float = 1.0e-4
    betas: list[float] = [0.5, 0.9]
    gp_lambda: float = 10.0
    bf16: bool = True
    compile: bool = False
    ckpt_every: int = 500
    eval_every: int = 500


class StylizedFactsConfig(BaseModel):
    acf_lags: list[int] = [1, 5, 22]


class EvalConfig(BaseModel):
    n_way: int = 5
    n_episodes: int = 1000
    stylized_facts: StylizedFactsConfig = StylizedFactsConfig()


class QuizConfig(BaseModel):
    n_way: int = 5
    out_dir: str = "reports/quiz/"


class PromotionConfig(BaseModel):
    min_n_way_acc: float = 0.45
    min_ks_pvalue: float = 0.05


class Config(BaseModel):
    seed: int = 0
    deterministic: bool = False
    device: str = "auto"

    data: DataConfig
    windows: WindowsConfig = WindowsConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    quiz: QuizConfig = QuizConfig()
    promotion: PromotionConfig = PromotionConfig()

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        # Resolve "auto" -> actual device. Deferred to load_config so tests
        # can also trigger it; the validator handles it if called directly.
        if v == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v


# ---------------------------------------------------------------------------
# Deep-merge helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict.

    - Dicts are merged recursively (override wins on scalar conflicts).
    - Lists and scalars in override replace those in base entirely.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a possible `extends:` parent via deep merge.

    If the YAML contains `extends: some_file.yaml`, the parent is loaded first
    (relative to the child file's directory), then the child is deep-merged on top.
    The `extends` key itself is stripped before validation.

    After building the Config, device="auto" is resolved (the field_validator
    handles this automatically during model construction).
    """
    path = Path(path).resolve()
    raw = _load_yaml(path)

    if "extends" in raw:
        parent_name = raw.pop("extends")
        parent_path = path.parent / parent_name
        parent_raw = _load_yaml(parent_path)
        # parent may itself have extends — recurse
        if "extends" in parent_raw:
            parent_cfg = load_config(parent_path)
            parent_raw = parent_cfg.model_dump()
            # device was already resolved; keep it as resolved string
        raw = _deep_merge(parent_raw, raw)

    return Config.model_validate(raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    return dict(data)

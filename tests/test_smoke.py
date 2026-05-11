"""Smoke tests — confirm the package imports and configs are well-formed.

Real coverage (windows leakage, OHLC consistency, n-way harness, etc.) is added with the
implementation in Phases A.1 / B — see CLAUDE.md.
"""

from pathlib import Path

import yaml

import finpred

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert finpred.__version__


def test_configs_are_valid_yaml():
    for name in ("default.yaml", "era_2012.yaml"):
        cfg = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
        assert isinstance(cfg, dict)
        # era boundaries we rely on everywhere downstream
        if name == "era_2012.yaml":
            assert cfg["data"]["train_end_date"] == "2012-06-30"
            assert cfg["data"]["eval_start_date"] == "2012-07-01"
        # default config carries the actual block structure
        if name == "default.yaml":
            for block in ("data", "windows", "model", "train", "eval", "quiz", "promotion"):
                assert block in cfg


def test_submodules_importable():
    import finpred.config.schema  # noqa: F401
    import finpred.data.ingest  # noqa: F401
    import finpred.data.windows  # noqa: F401
    import finpred.eval.n_way  # noqa: F401
    import finpred.eval.stylized_facts  # noqa: F401
    import finpred.models.discriminator  # noqa: F401
    import finpred.models.generator  # noqa: F401
    import finpred.models.tcn_blocks  # noqa: F401
    import finpred.quiz.cli  # noqa: F401
    import finpred.render.ohlcv_to_png  # noqa: F401
    import finpred.train.wgan_gp  # noqa: F401

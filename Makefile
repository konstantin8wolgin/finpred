# finpred — convenience targets. Assumes `uv` is installed (https://docs.astral.sh/uv/).
# If you don't use uv, replace `uv run` with `python -m` after `pip install -e .[dev]`.

CONFIG ?= configs/era_2012.yaml

.PHONY: help sync ingest train eval quiz render test lint typecheck

help:
	@echo "Targets:"
	@echo "  sync       - create venv + install deps (uv sync --extra dev)"
	@echo "  ingest     - download + cache OHLCV to data/  (CONFIG=$(CONFIG))"
	@echo "  train      - train the TCN-GAN              (CONFIG=$(CONFIG))"
	@echo "  eval       - N-way accuracy + stylized facts (CONFIG=$(CONFIG))"
	@echo "  quiz       - render N candlesticks + take the quiz"
	@echo "  render     - render real-vs-generated sample PNGs across checkpoints"
	@echo "  test       - pytest"
	@echo "  lint       - ruff check"
	@echo "  typecheck  - mypy"

sync:
	uv sync --extra dev

ingest:
	uv run python -m finpred.data.ingest --config $(CONFIG)

train:
	uv run python -m finpred.train.wgan_gp --config $(CONFIG)

eval:
	uv run python -m finpred.eval.n_way --config $(CONFIG)

quiz:
	uv run python -m finpred.quiz.cli --config $(CONFIG)

render:
	uv run bash scripts/render_samples.sh

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

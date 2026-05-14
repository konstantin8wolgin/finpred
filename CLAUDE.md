# CLAUDE.md — finpred

> Read this first. It is the entry point for any fresh session. After this, look at
> `configs/era_2012.yaml`, then `src/finpred/train/wgan_gp.py`.

## What this project is

A GAN-style system over historical US daily stock data. Two coupled goals:

1. **Generate** realistic synthetic daily OHLCV price series (configurable window length,
   252–2520 trading days ≈ 1–10 years).
2. **Discriminate**: given **N** charts where exactly one is a real chart from a **held-out
   future era**, pick the real one. Top-1 accuracy on this N-way task is the headline metric.

The headline framing: *"Trained only on data ≤ 2012-06-30, can the model spot a real
2012-2019 NVIDIA chart hidden among 4 convincing fakes — and are its fakes actually
convincing?"*

Phase 2 of the broader project (NOT this repo's scope yet) repeats the exercise for the 2020s
/ COVID era. Keep eras configurable so that's a config change, not a rewrite.

## Era split (current target)

- **Train** on all data with date ≤ **2012-06-30**.
- **Training-time validation slice**: dates **2012-01-01 .. 2012-06-30** — used for hyperparam
  selection / early stopping ONLY. Never report final numbers on it.
- **Held-out evaluation era**: dates **2012-07-01 .. 2019-12-31** — touched only by `make eval`
  / `make quiz`. The cutoff is placed well after the GFC (2007–2009) and QE-1/QE-2 so the
  held-out era is closer to a "normal regime" than a crisis recovery.
- `windows.py` MUST enforce these boundaries; there are tests for leakage.

## Architecture (PINNED — change deliberately, and update this section if you do)

QuantGAN-style. Reference: **Wiese, Knobloch, Korn, Kretschmer (2020), "Quant GANs: Deep
Generation of Financial Time Series"**, *Quantitative Finance*. Loss: **WGAN-GP** (Gulrajani et
al. 2017, "Improved Training of Wasserstein GANs").

- **Data representation**: each sample is a `(T, 5)` sequence — per-day log-returns of Open,
  High, Low, Close and the log-difference of Volume. Standardized per ticker using statistics
  computed on that ticker's pre-2012-07-01 window. The renderer reconstructs absolute prices
  (`cumsum` on close log-returns, then derive O/H/L from the modeled spreads) only for
  human-facing PNGs — the model only ever sees numbers.
- **Generator G** — TCN with dilated causal convolutions.
  - Input: noise `(B, T, z_dim=3)`. Optional `cond` arg accepted but **ignored in v1**
    (reserved for v2 conditional model: sector / ticker-class / starting volatility).
  - **10 residual TCN blocks**, kernel size 5, hidden channels 80,
    dilations `{1,2,4,8,16,32,64,128,256,512}`.
  - Receptive field: `1 + 2*(5-1)*(1+2+…+512) = 1 + 8*1023 = 8185` steps ≥ T_max=2520.
  - ~3–4 M parameters.
  - **OHLC consistency contract** (document at top of `generator.py`): the output head
    parameterizes `high = max(open, close) + softplus(...)`, `low = min(open, close) -
    softplus(...)`, so every emitted bar satisfies `low ≤ min(open,close) ≤ max(open,close) ≤
    high`. Invalid candles are a classic financial-GAN pitfall; this makes them impossible by
    construction. See `generator.py` docstring for the precise 5-channel parameterization.
- **Discriminator D** — same TCN depth (10 blocks, same dilations), global average pool →
  linear → scalar critic score. ~3–4 M params.
- **Optimizer**: Adam, `betas=(0.5, 0.9)`, `lr=1e-4`. `n_critic=5` critic steps per generator
  step. Gradient penalty coefficient `λ=10`.
- **Precision**: BF16 autocast (RTX 4070 / Ada supports it natively). `torch.compile` on the
  generator + discriminator.
- **Batch size and gradient accumulation**: nominal `batch_size=64` at `T=252`. For longer
  windows the training loop auto-scales: `effective_grad_accum = max(cfg.train.grad_accum, T//252)`
  so that effective tokens per update step stays constant and VRAM stays under ~10 GB. Example:
  T=2520 → grad_accum=10, per-step batch≈6. The resolved value is logged to `run.json`.
  See `configs/era_2012.yaml` for the per-length breakdown.
- **Schedule**: 50k–100k generator steps; checkpoint + run eval every 5k steps.

## Hardware target

NVIDIA **RTX 4070**, 12 GB VRAM, Ada Lovelace — BF16 + FP8 capable. Code should:
- use BF16 autocast (not FP16 — Ada handles BF16 well and it's more stable for GANs);
- keep peak VRAM under ~10 GB (leave headroom);
- fall back gracefully to CPU for tests (tiny configs in `tests/`).

## Stack

Python only. PyTorch + CUDA (the "fast" path is already C++/CUDA via PyTorch — there is no
Julia/C++ host code, by design). `polars`+`pyarrow` for the data pipeline, `yfinance` for
ingestion, `mplfinance` for rendering, `pydantic`+`pyyaml` for config, `tensorboard` for
tracking (no wandb — keep it offline-friendly), `typer` for the quiz CLI, `pytest` for tests.
Env managed with `uv`.

## Commands

```bash
uv sync --extra dev          # or: pip install -e .[dev]
make ingest                  # yfinance (+ Stooq supplement) -> data/*.parquet  (~10-20 GB)
make train                   # writes checkpoints/<run_id>/ and reports/tb/<run_id>/
make eval                    # writes reports/<run_id>/eval.json (N-way acc + stylized facts)
make quiz                    # renders reports/quiz/*.png — take the quiz yourself
make render                  # real-vs-generated sample PNGs across checkpoints
make test lint typecheck

# Run a single test file or test function:
uv run pytest tests/test_wgan_gp.py -v
uv run pytest tests/test_windows_leakage.py::test_no_train_leakage -v
```

## Evaluation — what "good" means

1. **Primary**: **N-way pick-the-real accuracy** on the held-out era. Default `N=5` ⇒ chance
   = 0.20. We want clearly above chance but **not** 1.0 — a perfect score means the generator
   is too weak. Healthy target band roughly 0.30–0.80.
2. **Secondary — stylized-facts battery** on generated samples (reported every checkpoint):
   - kurtosis of returns (heavy tails),
   - autocorrelation of `|returns|` at lags 1/5/22 (volatility clustering),
   - correlation of returns with future realized vol (leverage effect),
   - near-zero autocorrelation of raw returns,
   - KS test of the return distribution vs the real held-out distribution (report p-value).
3. **Human quiz** (`make quiz`): render `1 real + (N-1) fake` candlestick PNGs and pick. The
   fakes are length-matched to the real chart, drawn from G unconditionally, and rescaled so
   their starting price equals the real chart's; all PNGs use identical y-axis ranges and date
   formatting so nobody can pick on rendering artifacts.

**Promotion criterion** (gate for any autonomous loop in Phase C): a checkpoint is "promoted"
iff `5-way accuracy > 0.45` AND `KS p-value on returns > 0.05` on the training-time validation
slice. (Tune these once real numbers exist.)

## Known gotchas

- **Overlapping windows are correlated.** We sample overlapping rolling windows for data
  volume; don't treat them as i.i.d. when computing eval confidence intervals — block-bootstrap
  by ticker.
- **Survivorship bias.** yfinance over-represents currently-listed tickers. v1 mitigation:
  explicitly add known historical delistings (Lehman, Bear Stearns, WaMu, Enron, ...) via a
  Stooq supplemental pull for the *training* set. Fuller delisting coverage is deferred.
- **Mode collapse signs**: generated stylized facts collapse to a narrow band, discriminator
  loss → 0, N-way accuracy → 1.0. If you see this, lower `n_critic`, add noise to D inputs, or
  reduce G capacity.
- **Invalid candles**: must be impossible by construction (see OHLC contract above). If a test
  ever sees `high < close`, the generator head is wrong.
- **Era leakage**: the single most important correctness property. `windows.py` tests assert no
  date > 2012-06-30 ever enters a training batch.
- **`torch.use_deterministic_algorithms(True)`** may error on some ops — wrap in a config flag
  and disable for those ops rather than removing seeding entirely.

## Reproducibility

Every run takes a `seed`. Set `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and
(optionally) `torch.use_deterministic_algorithms(True)`. Log the resolved config + seed + git
SHA to `reports/<run_id>/run.json`.

## Repo layout

```
finpred/
  CLAUDE.md                # you are here
  README.md
  pyproject.toml           # deps + entry points; uv-managed
  Makefile                 # convenience targets
  configs/
    default.yaml           # base config (small/fast — good for smoke runs)
    era_2012.yaml          # the real run: train <=2012-06-30, eval 2012-07..2019
  src/finpred/
    config/    schema.py   # pydantic config models + YAML loader; `extends:` deep-merge
    data/      ingest.py   # yfinance (+Stooq) -> parquet cache; standardizes per ticker
               windows.py  # overlapping rolling-window sampler, era-cutoff enforced
    models/    tcn_blocks.py  # dilated causal conv residual block (shared by G & D)
               generator.py   # TCN generator, OHLC-consistent head
               discriminator.py  # TCN critic: global avg pool -> scalar
    train/     wgan_gp.py  # training loop: WGAN-GP, BF16, checkpoints, eval hook
    eval/      n_way.py    # N-way pick-the-real harness + CLI
               stylized_facts.py  # kurtosis / ACF|r| / leverage / KS
    render/    ohlcv_to_png.py  # mplfinance candlestick renderer
    quiz/      cli.py      # typer CLI: pose the N-way quiz to model and/or human
  scripts/
    render_samples.sh      # real-vs-generated PNGs across checkpoints
  tests/                   # pytest, TDD — paired with each module above
  data/  checkpoints/  reports/   # gitignored, regenerated
```

## Implementation roadmap & multi-agent orchestration

Full design + rationale: `/home/konstantin/.claude/plans/i-want-to-build-luminous-rabbit.md`.

The plan is itself an experiment in orchestration style. Evidence we're leaning on: Anthropic's
"Building a multi-agent research system" (parallel subagents win for independent, read-heavy
work) vs. Cognition's "Don't Build Multi-Agents" (single long-context agent wins for tightly
coupled software); ChatDev/MetaGPT role-pipelines look good in demos but underperform on
research code; fresh-context reviewers cut anchoring bias; a separate "tester" agent adds
little vs. TDD inside the dev agent.

- **Phase A.0 — DONE**: `git init`, this `CLAUDE.md`, `pyproject.toml`, configs, package
  skeleton with stubs, first commit.
- **Phase A.1 — DONE**: implemented the independent leaves with TDD — `data/` (ingest +
  windows), `render/`, `quiz/` CLI, `config/`.
- **Phase B — DONE**: built `models/` + `train/wgan_gp.py` + `eval/n_way.py` +
  `eval/stylized_facts.py` together, TDD.
- **Phase-boundary review**: after A.1 and after B, dispatch a *fresh-context* code-reviewer
  subagent over the diff (no implementation context → no anchoring). Human signs off.
- **Phase C — CURRENT**: human-in-loop iteration. Look at loss curves + sample PNGs + N-way
  accuracy each run. Only wrap it in an autonomous loop (`ralph-loop`) once the promotion
  criterion above is wired and trustworthy.
- **No separate tester agent.** Whoever writes code writes its tests.

## Definition of done (v1)

- `make ingest` → `data/` has >1e6 daily bars across ≥1500 tickers (≤2012-06-30) + a separate
  2012-07..2019 eval slice.
- `make test` green; `windows.py` leakage tests pass.
- `make train` runs ≥5k generator steps on the 4070 in BF16 without OOM; TensorBoard logs
  appear under `reports/tb/`.
- `make eval` writes `reports/<run_id>/eval.json` with 5-way accuracy > 0.20 and the full
  stylized-facts table.
- `make quiz` renders 5 candlestick PNGs (1 real post-2012 NVDA + 4 synthetic, length-matched,
  start-price-aligned) to `reports/quiz/`.
- `make render` shows a clear visual progression real-vs-generated across checkpoints 0/25k/50k.

## Out of scope (v1)

2020s / pre-2012 eras (phase 2); intraday data; alternative generators (TimeGAN, diffusion) —
leave the `generator.py` interface clean for a swap; live trading or any forecasting beyond the
quiz; a web UI for the quiz (CLI only).

## Rules

- No emojis anywhere (code, docs, commit messages).
- Prefer `uv run` / `uv` over bare `python3`.

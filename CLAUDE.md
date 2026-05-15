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

## Current state (as of last session)

**All commands are implemented and working.** The project has completed one full era_2012 run.

| Item | Status |
|---|---|
| `make ingest` | Works. **98 tickers** currently in `data/` (target: ≥1500). Run again to expand. |
| `make train` | Works. Full 80k-step run completed in ~10.5 hours on RTX 4070. |
| `make eval` | Works. Measures held-out 5-way accuracy on 2012-2019 era. |
| `make quiz` | Works. Renders 5 PNGs + prompts for human guess + reveals answer. |
| `make render` | Works. Renders real-vs-fake PNGs per checkpoint for visual inspection. |
| `make test` | 58 tests, all passing. |

**Best run so far: `20260514_173956_1234`** (era_2012.yaml, 80k steps, seed=1234)
- Held-out 5-way accuracy: **0.420** (95% CI: 0.407–0.434) — inside the healthy 0.30–0.80 band
- KS p-value on returns: 1.0 — generated return distribution statistically matches real
- Gradient penalty converged to ~0.007 — Lipschitz constraint well satisfied
- Visual quality: generator produces diverse, plausible candlestick charts by step 5k–10k

**Remaining known issues:**
- Volume drift: `cumsum(logdiff_volume)` has no mean-reversion so volume drifts to 0 or ∞
  over long windows. Price dynamics are solid; volume is the main visual tell.
- Only 98 tickers: `make ingest` with era_2012.yaml targets ~3000. More data = better diversity.

**Immediate next step:** run `make ingest` to expand the ticker universe, then re-train.

---

## Era split (current target)

- **Train** on all data with date ≤ **2012-06-30**.
- **Training-time validation slice**: dates **2012-01-01 .. 2012-06-30** — used for hyperparam
  selection / early stopping ONLY. Never report final numbers on it.
- **Held-out evaluation era**: dates **2012-07-01 .. 2019-12-31** — touched only by `make eval`
  / `make quiz`. The cutoff is placed well after the GFC (2007–2009) and QE-1/QE-2 so the
  held-out era is closer to a "normal regime" than a crisis recovery.
- `windows.py` MUST enforce these boundaries; there are tests for leakage.

---

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
  generator only — the discriminator is used with `create_graph=True` (double-backward for the
  gradient penalty), which torch.compile/aot_autograd does not support.
- **Batch size and gradient accumulation**: nominal `batch_size=64` at `T=252`. For longer
  windows the training loop auto-scales: `effective_grad_accum = max(cfg.train.grad_accum, T//252)`
  so that effective tokens per update step stays constant and VRAM stays under ~8 GB. Example:
  T=2520 → grad_accum=10, per-step batch≈6. The resolved value is logged to `run.json`.
  See `configs/era_2012.yaml` for the per-length breakdown.
- **DataLoader**: `num_workers=4`, `pin_memory=True`, `persistent_workers=True` — required for
  good GPU utilization. Hardcoded to 0 workers would leave the GPU idle between batches.
- **Schedule**: 80k generator steps (era_2012); checkpoint + run eval every 5k steps.

---

## Hardware target

NVIDIA **RTX 4070 (laptop, 8 GB VRAM)**, Ada Lovelace — BF16 + FP8 capable. Code should:
- use BF16 autocast (not FP16 — Ada handles BF16 well and it's more stable for GANs);
- keep peak VRAM under ~7 GB (leave headroom for the OS and display);
- fall back gracefully to CPU for tests (tiny configs in `tests/`).

---

## Stack

Python only. PyTorch + CUDA (the "fast" path is already C++/CUDA via PyTorch — there is no
Julia/C++ host code, by design). `polars`+`pyarrow` for the data pipeline, `yfinance` for
ingestion, `mplfinance` for rendering, `pydantic`+`pyyaml` for config, `tensorboard` for
tracking (no wandb — keep it offline-friendly), `typer` for the quiz CLI, `pytest` for tests.
Env managed with `uv`.

---

## Commands

```bash
uv sync --extra dev          # or: pip install -e .[dev]
make ingest                  # yfinance (+ Stooq supplement) -> data/*.parquet
make train                   # writes checkpoints/<run_id>/ and reports/tb/<run_id>/
make eval                    # N-way accuracy on held-out 2012-2019 era -> reports/<run_id>/eval.json
make quiz                    # renders 5 PNGs to reports/quiz/ and prompts for human guess
make render                  # real-vs-generated PNGs per checkpoint -> reports/<run_id>/samples/
make test lint typecheck

# Specific checkpoint / config overrides:
make eval CONFIG=configs/era_2012.yaml
make eval CONFIG=configs/era_2012.yaml ARGS="--checkpoint checkpoints/<run_id>/step_050000.pt"
make quiz CONFIG=configs/era_2012.yaml

# Background training in tmux (survives sleep/disconnect):
tmux new -s train
make train >> logs/train.log 2>&1
# Ctrl+B, D to detach; tmux attach -t train to reattach

# Monitor a running training:
tail -f logs/train.log
uv run tensorboard --logdir reports/tb/

# Run a single test:
uv run pytest tests/test_wgan_gp.py -v
uv run pytest tests/test_windows_leakage.py::test_no_train_leakage -v
```

---

## Evaluation — what "good" means

### Understanding the N-way accuracy metric

The metric is **inverted** relative to a normal classifier. It is a lineup test:
- Present D with 1 real chart + 4 fakes; D scores each
- Accuracy = fraction of episodes where D correctly picks the real one
- Chance = 1/5 = **0.20**

| Accuracy | Meaning |
|---|---|
| 1.0 | D always spots the real → fakes are obviously fake → **bad generator** |
| 0.30–0.80 | D has a slight edge → fakes are competitive → **healthy range** |
| 0.20 (chance) | D can't tell → fakes indistinguishable → **generator winning** |

Training-time accuracy (measured on training data after WGAN-GP convergence) will be near
chance — this is expected and correct, not a failure. The honest metric is `make eval` on the
held-out 2012-2019 era.

### Metrics

1. **Primary**: **N-way pick-the-real accuracy** on held-out era. Default `N=5` ⇒ chance = 0.20.
   Target band: **0.30–0.80**. First run achieved **0.420**.
2. **Secondary — stylized-facts battery** on generated samples (reported every checkpoint):
   - kurtosis of returns (heavy tails),
   - autocorrelation of `|returns|` at lags 1/5/22 (volatility clustering),
   - correlation of returns with future realized vol (leverage effect),
   - near-zero autocorrelation of raw returns,
   - KS test of the return distribution vs the real held-out distribution (report p-value).
3. **Human quiz** (`make quiz`): render `1 real + (N-1) fake` candlestick PNGs and pick. All
   PNGs use identical y-axis ranges and start-price-aligned so nobody can pick on rendering
   artifacts.

**Promotion criterion** (gate for autonomous loop): a checkpoint is "promoted"
iff `5-way held-out accuracy > 0.45` AND `KS p-value on returns > 0.05`.
First run: acc=0.420 (just below), ks_pvalue=1.0. Close.

---

## Known gotchas

- **N-way accuracy is inverted**: higher = worse generator, lower = better. Near-chance on
  training data after convergence is expected and healthy.
- **Training-time eval uses train split**: the 6-month val window (2012-01-01..2012-06-30 ≈ 125
  trading days) is shorter than T=252, so zero valid windows exist. The training loop falls back
  to `train_dataset` and logs a warning. The honest number comes from `make eval`.
- **Volume drift**: `cumsum(logdiff_volume)` has no mean-reversion; volume drifts toward 0 or ∞
  over a 252-day window. Price dynamics are solid; volume is the main visual tell. Future fix:
  model volume log-level with mean-reversion, or use a separate volume head.
- **Overlapping windows are correlated.** We sample overlapping rolling windows for data
  volume; don't treat them as i.i.d. when computing eval confidence intervals — block-bootstrap
  by ticker (already done in `n_way_accuracy()`).
- **Survivorship bias.** yfinance over-represents currently-listed tickers. v1 mitigation:
  explicitly add known historical delistings (Lehman, Bear Stearns, WaMu, Enron, ...) via a
  Stooq supplemental pull for the *training* set. Fuller delisting coverage is deferred.
- **Discriminator dominates early**: N-way accuracy near 1.0 in the first few hundred steps is
  normal — the untrained generator produces trivially distinguishable noise. It should trend down
  into 0.50–0.85 by step 2k–5k (smoke config). A plateau at exactly 1.0 beyond step 5k is the
  real mode-collapse warning.
- **Mode collapse signs**: generated stylized facts collapse to a narrow band, discriminator
  loss → 0, N-way accuracy → 1.0 (sustained). If you see this, lower `n_critic`, add noise to D
  inputs, or reduce G capacity.
- **Invalid candles**: must be impossible by construction (see OHLC contract above). If a test
  ever sees `high < close`, the generator head is wrong.
- **Era leakage**: the single most important correctness property. `windows.py` tests assert no
  date > 2012-06-30 ever enters a training batch.
- **`torch.use_deterministic_algorithms(True)`** may error on some ops — wrap in a config flag
  and disable for those ops rather than removing seeding entirely.
- **`torch.compile` on G only**: D uses `create_graph=True` in the gradient penalty
  (double-backward), which torch.compile/aot_autograd does not support. Compiling D will crash.

---

## Reproducibility

Every run takes a `seed`. Set `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and
(optionally) `torch.use_deterministic_algorithms(True)`. Log the resolved config + seed + git
SHA to `reports/<run_id>/run.json`.

---

## Repo layout

```
finpred/
  CLAUDE.md                # you are here
  README.md
  pyproject.toml           # deps + entry points; uv-managed
  Makefile                 # convenience targets
  configs/
    default.yaml           # base config (small/fast — good for smoke runs, 2k steps)
    era_2012.yaml          # the real run: train <=2012-06-30, eval 2012-07..2019, 80k steps
  src/finpred/
    config/    schema.py   # pydantic config models + YAML loader; `extends:` deep-merge
    data/      ingest.py   # yfinance (+Stooq) -> parquet cache; standardizes per ticker
               windows.py  # map-style Dataset, rolling windows, era-cutoff enforced
    models/    tcn_blocks.py     # dilated causal conv residual block (shared by G & D)
               generator.py     # TCN generator, OHLC-consistent output head
               discriminator.py # TCN critic: global avg pool -> scalar
    train/     wgan_gp.py  # training loop: WGAN-GP, BF16, checkpoints, eval hook
    eval/      n_way.py    # n_way_accuracy() function + make eval CLI entry point
               stylized_facts.py  # kurtosis / ACF|r| / leverage / KS
    render/    ohlcv_to_png.py  # mplfinance candlestick renderer; reconstruct_ohlcv()
    quiz/      cli.py      # typer CLI: render quiz PNGs + human guess + D scores
  scripts/
    render_samples.py      # real-vs-generated PNGs across all checkpoints in a run
  tests/                   # pytest, TDD — 58 tests, all passing
  data/                    # gitignored; *.parquet per ticker + stats/ subdirectory
  checkpoints/             # gitignored; <run_id>/step_XXXXXX.pt
  reports/                 # gitignored; <run_id>/run.json, eval.json, samples/, tb/
  logs/                    # gitignored; train.log
```

---

## Implementation roadmap

- **Phase A.0 — DONE**: skeleton, CLAUDE.md, pyproject.toml, configs, stubs.
- **Phase A.1 — DONE**: `data/` (ingest + windows), `render/`, `quiz/` CLI, `config/`.
- **Phase B — DONE**: `models/` + `train/wgan_gp.py` + `eval/n_way.py` + `eval/stylized_facts.py`.
- **Phase C — IN PROGRESS**: human-in-loop iteration.
  - All make targets implemented and working.
  - First full run (80k steps, 98 tickers) achieved **0.420 held-out 5-way accuracy**.
  - Immediate next step: `make ingest` to expand to ≥1500 tickers, then re-train.
  - Volume drift is the main remaining quality issue (architectural fix deferred).
  - Wrap in autonomous loop (`ralph-loop`) once promotion criterion is reliably met.

### Phase C iteration workflow

```bash
# 1. Expand data (one-time, takes hours):
make ingest

# 2. Train (era_2012, ~10 hours on 4070):
tmux new -s train
make train >> logs/train.log 2>&1

# 3. While training, watch loss curves:
uv run tensorboard --logdir reports/tb/

# 4. After training, render visual samples:
make render

# 5. Get the honest held-out accuracy:
make eval

# 6. Take the human quiz:
make quiz
```

---

## Definition of done (v1)

- [x] `make test` green (58 tests)
- [x] `make train` runs ≥5k steps on 4070 in BF16 without OOM
- [x] `make eval` writes held-out accuracy (first result: 0.420)
- [x] `make quiz` renders 5 PNGs and prompts for human guess
- [x] `make render` shows real-vs-generated progression across checkpoints
- [ ] `make ingest` → ≥1500 tickers in `data/` (currently 98)
- [ ] Held-out 5-way accuracy > 0.45 on a ≥1500-ticker run

---

## Out of scope (v1)

2020s / pre-2012 eras (phase 2); intraday data; alternative generators (TimeGAN, diffusion) —
leave the `generator.py` interface clean for a swap; live trading or any forecasting beyond the
quiz; a web UI for the quiz (CLI only); fixing volume drift (deferred to v2).

---

## Rules

- No emojis anywhere (code, docs, commit messages).
- Prefer `uv run` / `uv` over bare `python3`.

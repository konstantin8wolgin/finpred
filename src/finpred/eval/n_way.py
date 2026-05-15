"""N-way pick-the-real evaluation — the headline metric.

Algorithm per episode:
  1. Sample 1 real window from dataset (eval split) -> shape (T, 5).
  2. Generate n_way-1 fakes from generator; noise z ~ N(0,1) shape (n_way-1, T, z_dim).
  3. Concatenate real (at index 0) + fakes -> (n_way, T, 5).
  4. Score all n_way with discriminator -> (n_way,) scores.
  5. Episode is correct iff argmax == 0 (the real one).
  6. Average over n_episodes -> accuracy; block-bootstrap 95% CI.

`main()` is the `finpred-eval` / `make eval` entry point (evaluates a checkpoint on the
held-out era).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def _get_z_dim(generator: nn.Module) -> int:
    # Generator stores input_proj = Conv1d(z_dim, hidden_channels, 1).
    try:
        return generator.input_proj.in_channels
    except AttributeError:
        return 3


def n_way_accuracy(
    generator: nn.Module,
    discriminator: nn.Module,
    dataset: Dataset,
    n_way: int,
    n_episodes: int,
) -> dict:
    """Return {'accuracy': float, 'ci_low': float, 'ci_high': float, 'n_way': int}."""
    device = next(discriminator.parameters()).device
    z_dim = _get_z_dim(generator)

    n_samples = len(dataset)  # type: ignore[arg-type]
    # Per-episode correctness flags (1 = correct, 0 = wrong).
    correct = np.zeros(n_episodes, dtype=np.float32)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            # Sample one real window from the dataset.
            idx = random.randrange(n_samples)
            real = dataset[idx]  # (T, 5)
            if not isinstance(real, torch.Tensor):
                # Some datasets return (tensor, dates) tuples.
                real = real[0]
            T = real.shape[0]
            real = real.to(device)

            # Generate n_way-1 fakes of the same length.
            z = torch.randn(n_way - 1, T, z_dim, device=device)
            fakes = generator(z)  # (n_way-1, T, 5)

            # Candidates: real at index 0, then fakes.
            candidates = torch.cat(
                [real.unsqueeze(0), fakes], dim=0
            )  # (n_way, T, 5)

            scores = discriminator(candidates)  # (n_way,)
            picked = int(scores.argmax().item())
            correct[ep] = float(picked == 0)

    accuracy = float(correct.mean())

    # Block-bootstrap 95% CI: resample episode results with replacement 1000 times.
    rng = np.random.default_rng(seed=0)
    bootstrap_accs = np.empty(1000, dtype=np.float32)
    for b in range(1000):
        sample = rng.choice(correct, size=n_episodes, replace=True)
        bootstrap_accs[b] = sample.mean()
    ci_low = float(np.percentile(bootstrap_accs, 2.5))
    ci_high = float(np.percentile(bootstrap_accs, 97.5))

    return {
        "accuracy": accuracy,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_way": n_way,
    }


def _latest_checkpoint(ckpt_root: "Path") -> "Path":
    from pathlib import Path
    import re
    best_dir = max(
        (d for d in Path(ckpt_root).iterdir() if d.is_dir()),
        key=lambda d: len(list(d.glob("*.pt"))),
        default=None,
    )
    if best_dir is None:
        raise FileNotFoundError(f"No checkpoint directories found under {ckpt_root}")
    pts = sorted(best_dir.glob("*.pt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {best_dir}")
    return pts[-1]


def _load_models(ckpt_path: "Path", cfg):
    import re
    import torch
    from finpred.models.generator import Generator
    from finpred.models.discriminator import Discriminator

    device = cfg.device
    G = Generator(
        z_dim=cfg.model.z_dim,
        hidden_channels=cfg.model.hidden_channels,
        n_blocks=cfg.model.n_blocks,
        kernel_size=cfg.model.kernel_size,
        dilations=cfg.model.dilations,
    ).to(device)
    D = Discriminator(
        hidden_channels=cfg.model.hidden_channels,
        n_blocks=cfg.model.n_blocks,
        kernel_size=cfg.model.kernel_size,
        dilations=cfg.model.dilations,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for model, key in [(G, "G"), (D, "D")]:
        state = ckpt[key]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
    return G, D


def main() -> None:
    import json
    import logging
    from pathlib import Path
    from finpred.config.schema import load_config
    from finpred.data.ingest import resolve_tickers
    from finpred.data.windows import WindowDataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="N-way pick-the-real evaluation on held-out era.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint; default: latest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolved = resolve_tickers(cfg.data.tickers, extra_tickers=list(cfg.data.extra_tickers))
    cfg.data.tickers = resolved

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path("checkpoints"))
    run_id = ckpt_path.parent.name
    logger.info("Evaluating checkpoint: %s", ckpt_path)

    G, D = _load_models(ckpt_path, cfg)

    T = cfg.windows.lengths[0]
    eval_dataset = WindowDataset(cfg, split="eval", length=T)
    if len(eval_dataset) == 0:
        logger.error("eval_dataset is empty — no held-out data found for split='eval'")
        raise SystemExit(1)
    logger.info("Eval dataset: %d windows (T=%d)", len(eval_dataset), T)

    result = n_way_accuracy(
        G, D, eval_dataset,
        n_way=cfg.eval.n_way,
        n_episodes=cfg.eval.n_episodes,
    )

    logger.info(
        "%d-way accuracy: %.3f  (95%% CI: %.3f – %.3f)",
        result["n_way"], result["accuracy"], result["ci_low"], result["ci_high"],
    )

    report_dir = Path(f"reports/{run_id}")
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_path = report_dir / "eval.json"
    existing = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    existing[f"held_out_{ckpt_path.stem}"] = {
        "checkpoint": str(ckpt_path),
        "split": "eval",
        **result,
    }
    eval_path.write_text(json.dumps(existing, indent=2))
    logger.info("Results written to %s", eval_path)


if __name__ == "__main__":
    main()

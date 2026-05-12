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


def main() -> None:
    parser = argparse.ArgumentParser(description="N-way pick-the-real evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    parser.parse_args()
    raise NotImplementedError("eval.n_way.main — implement in Phase B")


if __name__ == "__main__":
    main()

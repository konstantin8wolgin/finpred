"""N-way pick-the-real evaluation — the headline metric.

STUB (Phase B). Implement:
  - episode = sample 1 real window from a split (default: held-out eval era) + (N-1) fakes from
    G; fakes are length-matched and rescaled so their start price equals the real one's,
  - score all N with the discriminator; the model "picks" argmax; episode is correct if it
    picked the real one,
  - average over `eval.n_episodes`; report top-1 accuracy (chance = 1/N) with a block-bootstrap
    CI by ticker (overlapping windows are correlated — see CLAUDE.md gotchas),
  - write reports/<run_id>/eval.json combining this with stylized_facts output.

`main()` is the `finpred-eval` / `make eval` entry point (evaluates a checkpoint on the
held-out era).
"""

from __future__ import annotations

import argparse

from torch import nn
from torch.utils.data import Dataset


def n_way_accuracy(
    generator: nn.Module,
    discriminator: nn.Module,
    dataset: Dataset,
    n_way: int,
    n_episodes: int,
) -> dict:
    """Return {'accuracy': float, 'ci_low': float, 'ci_high': float, 'n_way': int}. STUB."""
    raise NotImplementedError("eval.n_way.n_way_accuracy — implement in Phase B")


def main() -> None:
    parser = argparse.ArgumentParser(description="N-way pick-the-real evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    parser.parse_args()
    raise NotImplementedError("eval.n_way.main — implement in Phase B")


if __name__ == "__main__":
    main()

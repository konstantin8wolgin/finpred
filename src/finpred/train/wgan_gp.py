"""WGAN-GP training loop for the TCN generator + discriminator.

STUB (Phase B). Implement:
  - build G, D from config; Adam(betas=(0.5,0.9), lr); optional torch.compile,
  - per generator step: `n_critic` critic updates (real batch from data.windows + fake batch
    from G) with gradient penalty (lambda from config), then one generator update,
  - BF16 autocast on CUDA; grad accumulation scaled by window length to keep VRAM < ~10 GB (4070):
      effective_grad_accum = max(cfg.train.grad_accum, T // 252)
    so a T=252 batch runs at grad_accum=1 (batch_size=64) and T=2520 runs at grad_accum=10
    (effective batch_size=64, per-step size=6 or 7). Log the resolved value to run.json.
  - seeding + determinism from config; log resolved config + seed + git SHA to
    reports/<run_id>/run.json,
  - TensorBoard scalars (losses, GP, grad norms) to reports/tb/<run_id>/,
  - every `ckpt_every`: save checkpoints/<run_id>/step_<n>.pt,
  - every `eval_every`: call eval.n_way + eval.stylized_facts on the *val* split, log results,
    and mark "promoted" if it clears configs' promotion thresholds.

`main()` is the `finpred-train` / `make train` entry point.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TCN-GAN (WGAN-GP).")
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError("train.wgan_gp.main — implement in Phase B")


if __name__ == "__main__":
    main()

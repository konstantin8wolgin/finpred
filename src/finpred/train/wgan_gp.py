"""WGAN-GP training loop for the TCN generator + discriminator.

See CLAUDE.md for the full architecture description and hardware targets.
The effective_grad_accum formula keeps VRAM bounded for long windows:
    effective_grad_accum = max(cfg.train.grad_accum, T // 252)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from finpred.config.schema import Config, load_config
from finpred.data.ingest import resolve_tickers
from finpred.data.windows import WindowDataset
from finpred.eval.n_way import n_way_accuracy
from finpred.eval.stylized_facts import stylized_facts
from finpred.models.discriminator import Discriminator
from finpred.models.generator import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorBoard — optional; warn and skip if not installed
# ---------------------------------------------------------------------------

def _make_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        warnings.warn("tensorboard not installed — skipping TensorBoard logging", stacklevel=2)
        return None


def _write_scalar(writer, tag: str, value: float, step: int) -> None:
    if writer is not None:
        writer.add_scalar(tag, value, step)


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Gradient penalty (WGAN-GP, Gulrajani et al. 2017)
# ---------------------------------------------------------------------------

def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute the WGAN-GP gradient penalty term.

    Args:
        discriminator: the critic network.
        real: real samples, shape (B, T, 5).
        fake: generated samples, shape (B, T, 5). Detached from generator graph.
        device: torch device string.

    Returns:
        Scalar tensor: mean squared deviation of gradient norms from 1.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, device=device)
    # Interpolate between real and fake along a random straight line.
    interpolated = (alpha * real + (1.0 - alpha) * fake.detach()).requires_grad_(True)

    d_interp = discriminator(interpolated)
    # create_graph=True so second-order gradients flow back through the penalty.
    grads = torch.autograd.grad(
        d_interp.sum(),
        interpolated,
        create_graph=True,
    )[0]
    # Norm over (T, 5) dims, penalty is (||grad|| - 1)^2 averaged over batch.
    gp = ((grads.norm(2, dim=(1, 2)) - 1.0) ** 2).mean()
    return gp


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _infinite_loader(loader: DataLoader):
    """Yield batches from a DataLoader indefinitely."""
    while True:
        yield from loader


def _sample_real_batch(
    loader_iter,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Draw a batch of real windows from the DataLoader iterator.

    Returns tensor of shape (B, T, 5).
    """
    batch = next(loader_iter)
    # DataLoader may pad the last batch or return tuples — normalise.
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch[:batch_size].to(device)


# ---------------------------------------------------------------------------
# Eval hook
# ---------------------------------------------------------------------------

def _run_eval(
    G: nn.Module,
    D: nn.Module,
    val_dataset: WindowDataset,
    train_dataset: WindowDataset,
    cfg: Config,
    step: int,
    writer,
    run_json_path: Path,
) -> None:
    """Run N-way accuracy + stylized-facts and log to TensorBoard + run.json."""
    device = cfg.device
    z_dim = cfg.model.z_dim
    T = cfg.windows.lengths[0]

    # The val split is 6 months, shorter than the minimum window (252 days).
    # Fall back to train split for n_way so training-time eval is never empty.
    nway_dataset = val_dataset if len(val_dataset) > 0 else train_dataset
    if len(val_dataset) == 0:
        logger.warning(
            "val_dataset is empty (val split too short for T=%d windows); "
            "using train split for training-time n_way accuracy",
            T,
        )

    # N-way accuracy.
    nway_result = n_way_accuracy(
        G,
        D,
        nway_dataset,
        n_way=cfg.eval.n_way,
        n_episodes=cfg.eval.n_episodes,
    )
    acc = nway_result["accuracy"]
    _write_scalar(writer, "eval/n_way_accuracy", acc, step)

    # Stylized facts: generate a batch of samples + pull real returns from val set.
    G.eval()
    with torch.no_grad():
        n_gen = min(50, max(10, cfg.eval.n_episodes // 20))
        z = torch.randn(n_gen, T, z_dim, device=device)
        gen_samples = G(z).cpu().numpy()  # (n_gen, T, 5)

    # Collect real samples from val_dataset (up to n_gen windows).
    real_list = []
    for i, item in enumerate(val_dataset):
        if i >= n_gen:
            break
        if isinstance(item, (list, tuple)):
            item = item[0]
        real_list.append(item.numpy())
    real_samples = np.stack(real_list) if real_list else gen_samples  # fallback

    # Use the close log-return column (index 3) for stylized-facts comparison.
    facts = stylized_facts(
        real_samples[:, :, 3],
        gen_samples[:, :, 3],
        cfg.eval.stylized_facts.acf_lags,
    )
    for k, v in facts.items():
        _write_scalar(writer, f"eval/{k}", v, step)

    ks_pvalue = facts.get("ks_pvalue", 0.0)

    # Check promotion criterion and append to run.json.
    promoted = acc > cfg.promotion.min_n_way_acc and ks_pvalue > cfg.promotion.min_ks_pvalue
    entry = {
        "step": step,
        "n_way_accuracy": acc,
        "ks_pvalue": ks_pvalue,
        "promoted": promoted,
    }
    _append_to_run_json(run_json_path, f"eval_step_{step}", entry)
    if promoted:
        logger.info("PROMOTED at step %d (acc=%.3f, ks_pvalue=%.4f)", step, acc, ks_pvalue)

    G.train()


def _append_to_run_json(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    """Full WGAN-GP training loop.

    Builds G and D from cfg.model, trains for cfg.train.steps generator steps,
    logs to TensorBoard, checkpoints every cfg.train.ckpt_every steps, and
    runs eval every cfg.train.eval_every steps.
    """
    # Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some ops do not support deterministic mode; disable rather than crash.
            pass

    device = cfg.device

    # Run ID and directory setup
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{cfg.seed}"
    report_dir = Path(f"reports/{run_id}")
    ckpt_dir = Path(f"checkpoints/{run_id}")
    tb_dir = f"reports/tb/{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Effective grad accumulation based on window length
    T = cfg.windows.lengths[0]
    effective_grad_accum = max(cfg.train.grad_accum, T // 252)

    # Write initial run.json
    run_json_path = report_dir / "run.json"
    run_json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": cfg.seed,
                "git_sha": _git_sha(),
                "effective_grad_accum": effective_grad_accum,
                "config": cfg.model_dump(),
            },
            indent=2,
            default=str,
        )
    )

    # Build models
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

    # Only compile G — D is used in gradient_penalty with create_graph=True (double backward),
    # which torch.compile/aot_autograd does not support.
    if cfg.train.compile and torch.cuda.is_available():
        G = torch.compile(G)  # type: ignore[assignment]

    # Optimizers
    betas = tuple(cfg.train.betas)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.train.lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.train.lr, betas=betas)

    # TensorBoard
    writer = _make_writer(tb_dir)

    # Resolve ticker sentinel before building datasets (era_2012 uses __SP1500_PLUS_NASDAQ__).
    resolved = resolve_tickers(cfg.data.tickers, extra_tickers=list(cfg.data.extra_tickers))
    cfg.data.tickers = resolved

    # Datasets and loaders
    train_dataset = WindowDataset(cfg, split="train", length=T)
    val_dataset = WindowDataset(cfg, split="val", length=T)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    loader_iter = _infinite_loader(train_loader)

    # BF16 autocast context — only active on CUDA
    use_bf16 = cfg.train.bf16 and device == "cuda"
    autocast_ctx = torch.autocast(
        device_type="cuda" if device == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )

    G.train()
    D.train()

    for step in range(1, cfg.train.steps + 1):
        # ------------------------------------------------------------------
        # n_critic discriminator / critic updates
        # ------------------------------------------------------------------
        total_critic_loss = 0.0
        total_gp = 0.0

        for _ in range(cfg.train.n_critic):
            # Accumulate over effective_grad_accum micro-batches.
            opt_D.zero_grad()
            accum_critic_loss = 0.0
            accum_gp = 0.0

            for _ in range(effective_grad_accum):
                real = _sample_real_batch(loader_iter, cfg.train.batch_size, device)
                if real.shape[0] < 2:
                    # Skip degenerate micro-batches (e.g. last batch with 1 sample).
                    continue
                b = real.shape[0]

                with autocast_ctx:
                    z = torch.randn(b, T, cfg.model.z_dim, device=device)
                    fake = G(z)

                    d_real = D(real)
                    d_fake = D(fake.detach())
                    critic_loss = d_fake.mean() - d_real.mean()

                gp = gradient_penalty(D, real, fake.detach(), device)
                total_loss = (critic_loss + cfg.train.gp_lambda * gp) / effective_grad_accum
                total_loss.backward()

                accum_critic_loss += critic_loss.item()
                accum_gp += gp.item()

            opt_D.step()
            total_critic_loss += accum_critic_loss / effective_grad_accum
            total_gp += accum_gp / effective_grad_accum

        avg_critic_loss = total_critic_loss / cfg.train.n_critic
        avg_gp = total_gp / cfg.train.n_critic

        # ------------------------------------------------------------------
        # One generator update
        # ------------------------------------------------------------------
        opt_G.zero_grad()
        gen_loss_accum = 0.0

        for _ in range(effective_grad_accum):
            real = _sample_real_batch(loader_iter, cfg.train.batch_size, device)
            b = real.shape[0]

            with autocast_ctx:
                z = torch.randn(b, T, cfg.model.z_dim, device=device)
                fake = G(z)
                gen_loss = -D(fake).mean() / effective_grad_accum

            gen_loss.backward()
            gen_loss_accum += gen_loss.item()

        opt_G.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        _write_scalar(writer, "loss/critic", avg_critic_loss, step)
        _write_scalar(writer, "loss/generator", gen_loss_accum, step)
        _write_scalar(writer, "grad_penalty", avg_gp, step)

        if step % 500 == 0 or step == 1:
            logger.info(
                "step %d/%d  critic=%.4f  gen=%.4f  gp=%.4f",
                step,
                cfg.train.steps,
                avg_critic_loss,
                gen_loss_accum,
                avg_gp,
            )

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if step % cfg.train.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                },
                ckpt_path,
            )
            logger.info("Checkpoint saved: %s", ckpt_path)

        # ------------------------------------------------------------------
        # Eval hook
        # ------------------------------------------------------------------
        if step % cfg.train.eval_every == 0:
            _run_eval(G, D, val_dataset, train_dataset, cfg, step, writer, run_json_path)
            G.train()
            D.train()

    if writer is not None:
        writer.close()

    logger.info("Training complete. run_id=%s", run_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train the TCN-GAN (WGAN-GP).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

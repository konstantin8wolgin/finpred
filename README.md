# finpred

Train a GAN-style model on historical US stock data **up to 2012-06-30**, then have it:

1. **Generate** realistic synthetic daily OHLCV price series (1–10 year windows), and
2. **Discriminate** — given N charts (1 real from the held-out 2012-07-01..2019-12-31 era +
   N-1 synthetic), pick the real one. Top-1 accuracy is the headline metric.

Architecture: QuantGAN-style Temporal Convolutional Network generator + discriminator, trained
with WGAN-GP. Target hardware: NVIDIA RTX 4070 (12 GB, BF16). Python/PyTorch only.

**This repo is currently a scaffold (Phase A.0 complete).** See `CLAUDE.md` for the full design,
the implementation roadmap (Phases A.1 / B / C), and how to continue from a fresh session.

## Quickstart (once implemented)

```bash
uv sync --extra dev      # create env, install deps
make ingest              # download + cache OHLCV to data/ (~10-20 GB)
make train               # train the TCN-GAN (writes checkpoints/ + reports/tb/)
make eval                # N-way pick-the-real accuracy + stylized-facts battery
make quiz                # render 1 real + N-1 fake candlestick PNGs, take the quiz yourself
make test                # pytest
```

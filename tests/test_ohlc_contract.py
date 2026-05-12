"""OHLC consistency contract tests for models.generator.Generator.

Implement alongside models/generator.py in Phase B using a tiny model config (z_dim=3,
hidden_channels=8, n_blocks=2, kernel_size=3, dilations=[1,2]).

Required assertions:

test_no_invalid_candles
  Sample 200 noise tensors z ~ N(0,1) of shape (B=4, T=64, z_dim=3), pass through Generator,
  reconstruct per-bar prices from the 5-channel output (see generator.py docstring for the
  exact parameterization). Assert for every (batch, time) step:
    low <= open
    low <= close
    high >= open
    high >= close
  Any violation means the OHLC contract is broken — fail immediately with the offending values.

test_softplus_margin_positive
  The high margin (high - max(open, close)) and low margin (min(open, close) - low) must both
  be strictly positive for all bars. A value of exactly 0.0 is a sign the softplus path was
  bypassed.

test_output_shape
  Generator(z).shape == (B, T, 5) for arbitrary B and T.

test_cond_ignored_in_v1
  Generator(z, cond=some_tensor) == Generator(z, cond=None) element-wise (v1 must ignore cond).

Implementation notes:
  - Use torch.no_grad() + float32 (not BF16) for deterministic assertions.
  - Seed with torch.manual_seed(0) for reproducibility.
  - Run with `uv run pytest tests/test_ohlc_contract.py -v`.
"""

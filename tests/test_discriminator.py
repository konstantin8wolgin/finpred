"""Tests for Discriminator (TCN WGAN critic)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from finpred.models.discriminator import Discriminator
from finpred.models.generator import Generator


@pytest.fixture
def seed():
    """Seed RNG for reproducibility."""
    torch.manual_seed(0)


def test_output_shape(seed):
    """Discriminator(hidden_channels=16, n_blocks=2, kernel_size=3, dilations=[1,2])
    on input (B=3, T=64, 5) produces output shape (3,).
    """
    d = Discriminator(hidden_channels=16, n_blocks=2, kernel_size=3, dilations=[1, 2])
    x = torch.randn(3, 64, 5)
    with torch.no_grad():
        out = d(x)
    assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"


def test_output_unbounded(seed):
    """Critic scores can be negative (not bounded to [0,1]).
    Generate random input, run forward, assert that not all scores are in [0,1].
    """
    d = Discriminator(hidden_channels=16, n_blocks=2, kernel_size=3, dilations=[1, 2])
    x = torch.randn(10, 64, 5)
    with torch.no_grad():
        out = d(x)
    # With high probability for random weights, at least one score should be outside [0,1]
    in_unit_interval = ((out >= 0) & (out <= 1)).all().item()
    assert not in_unit_interval, "All scores are in [0,1]; expected unbounded WGAN critic"


def test_generator_discriminator_pipeline(seed):
    """Create a small Generator and Discriminator, sample noise, G produces (B,T,5),
    D scores it as (B,). Check types and shapes.
    """
    g = Generator(
        z_dim=3, hidden_channels=8, n_blocks=2, kernel_size=3, dilations=[1, 2]
    )
    d = Discriminator(hidden_channels=8, n_blocks=2, kernel_size=3, dilations=[1, 2])

    noise = torch.randn(4, 64, 3)
    with torch.no_grad():
        fake = g(noise)
        scores = d(fake)

    assert fake.shape == (4, 64, 5), f"Generator output shape: {fake.shape}"
    assert scores.shape == (4,), f"Discriminator output shape: {scores.shape}"
    assert fake.dtype == torch.float32
    assert scores.dtype == torch.float32


def test_different_sequence_lengths(seed):
    """D must handle both T=64 and T=128 (global average pool makes T-invariant).
    Assert correct output shape for both.
    """
    d = Discriminator(hidden_channels=16, n_blocks=2, kernel_size=3, dilations=[1, 2])

    x_64 = torch.randn(2, 64, 5)
    x_128 = torch.randn(2, 128, 5)

    with torch.no_grad():
        out_64 = d(x_64)
        out_128 = d(x_128)

    assert out_64.shape == (2,), f"Expected (2,), got {out_64.shape}"
    assert out_128.shape == (2,), f"Expected (2,), got {out_128.shape}"


def test_no_sigmoid_in_architecture(seed):
    """Inspect the module list: no nn.Sigmoid anywhere in the discriminator's modules."""
    d = Discriminator(hidden_channels=16, n_blocks=2, kernel_size=3, dilations=[1, 2])

    has_sigmoid = False
    for module in d.modules():
        if isinstance(module, nn.Sigmoid):
            has_sigmoid = True
            break

    assert not has_sigmoid, "Discriminator should not contain nn.Sigmoid (WGAN critic)"

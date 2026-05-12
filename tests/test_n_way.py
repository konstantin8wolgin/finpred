"""Tests for eval.n_way — N-way pick-the-real harness."""
from __future__ import annotations

import torch
import torch.utils.data

from finpred.eval.n_way import n_way_accuracy
from finpred.models.discriminator import Discriminator
from finpred.models.generator import Generator

_DILATIONS = [1, 2]
_T = 64
_Z_DIM = 3


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i]


def _make_g() -> Generator:
    return Generator(
        z_dim=_Z_DIM,
        hidden_channels=8,
        n_blocks=2,
        kernel_size=3,
        dilations=_DILATIONS,
    )


def _make_d() -> Discriminator:
    return Discriminator(
        hidden_channels=8,
        n_blocks=2,
        kernel_size=3,
        dilations=_DILATIONS,
    )


def _make_dataset() -> _ListDataset:
    torch.manual_seed(42)
    tensors = [torch.randn(_T, 5) for _ in range(20)]
    return _ListDataset(tensors)


def test_accuracy_range() -> None:
    torch.manual_seed(42)
    g, d = _make_g(), _make_d()
    ds = _make_dataset()
    result = n_way_accuracy(g, d, ds, n_way=5, n_episodes=50)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_ci_bounds_valid() -> None:
    torch.manual_seed(42)
    g, d = _make_g(), _make_d()
    ds = _make_dataset()
    result = n_way_accuracy(g, d, ds, n_way=5, n_episodes=50)
    assert result["ci_low"] <= result["accuracy"] <= result["ci_high"]
    assert result["ci_low"] >= 0.0
    assert result["ci_high"] <= 1.0


def test_return_dict_keys() -> None:
    torch.manual_seed(42)
    g, d = _make_g(), _make_d()
    ds = _make_dataset()
    result = n_way_accuracy(g, d, ds, n_way=5, n_episodes=10)
    assert set(result.keys()) == {"accuracy", "ci_low", "ci_high", "n_way"}


def test_n_way_stored_in_result() -> None:
    torch.manual_seed(42)
    g, d = _make_g(), _make_d()
    ds = _make_dataset()
    result = n_way_accuracy(g, d, ds, n_way=5, n_episodes=10)
    assert result["n_way"] == 5


def test_random_discriminator_near_chance() -> None:
    # With identically distributed real and fake windows, a random discriminator should
    # not systematically prefer index 0.  We seed G outputs as the "dataset" so real and
    # fake tensors come from the same distribution (both are outputs of the same random G).
    torch.manual_seed(42)
    g, d = _make_g(), _make_d()

    # Build dataset from generator outputs so real/fake are identically distributed.
    g.eval()
    with torch.no_grad():
        dataset_tensors = [
            g(torch.randn(1, _T, _Z_DIM)).squeeze(0) for _ in range(50)
        ]
    ds = _ListDataset(dataset_tensors)

    result = n_way_accuracy(g, d, ds, n_way=5, n_episodes=200)
    # 1/5 = 0.2 chance; anything above 0.6 is statistically implausible for random weights
    # when real and fake are drawn from the same distribution.
    assert result["accuracy"] <= 0.6

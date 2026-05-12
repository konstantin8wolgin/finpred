"""Tests for src/finpred/eval/stylized_facts.py."""

from __future__ import annotations

import numpy as np
import pytest

from finpred.eval.stylized_facts import stylized_facts


RNG = np.random.default_rng(42)


def test_return_dict_keys():
    """All expected keys present in returned dict."""
    real_returns = RNG.normal(0, 0.01, (50, 252))
    gen_returns = RNG.normal(0, 0.01, (50, 252))
    acf_lags = [1, 5, 22]

    result = stylized_facts(real_returns, gen_returns, acf_lags)

    expected_keys = {
        'gen_kurtosis',
        'real_kurtosis',
        'gen_acf_abs_lag1',
        'gen_acf_abs_lag5',
        'gen_acf_abs_lag22',
        'real_acf_abs_lag1',
        'real_acf_abs_lag5',
        'real_acf_abs_lag22',
        'gen_acf_raw_lag1',
        'real_acf_raw_lag1',
        'leverage_corr',
        'ks_stat',
        'ks_pvalue',
    }
    assert set(result.keys()) == expected_keys


def test_kurtosis_normal_distribution():
    """Normal distribution should have excess kurtosis ~0."""
    real_returns = RNG.normal(0, 1, (100, 100))
    gen_returns = RNG.normal(0, 1, (100, 100))

    result = stylized_facts(real_returns, gen_returns, [1])

    assert abs(result['gen_kurtosis']) < 0.5, f"Expected ~0, got {result['gen_kurtosis']}"
    assert abs(result['real_kurtosis']) < 0.5, f"Expected ~0, got {result['real_kurtosis']}"


def test_kurtosis_heavy_tails():
    """t-distribution with df=4 has excess kurtosis = 6.0."""
    gen_returns = RNG.standard_t(df=4, size=(100, 500))
    real_returns = RNG.normal(0, 1, (100, 500))

    result = stylized_facts(real_returns, gen_returns, [1])

    assert result['gen_kurtosis'] > 2.0, f"Expected heavy tails, got {result['gen_kurtosis']}"


def test_ks_same_distribution():
    """KS test should not reject same distribution."""
    data = RNG.normal(0, 1, (50, 100))
    real_returns = data
    gen_returns = data + RNG.normal(0, 0.001, (50, 100))

    result = stylized_facts(real_returns, gen_returns, [1])

    assert result['ks_pvalue'] > 0.05, f"Expected p > 0.05, got {result['ks_pvalue']}"


def test_ks_different_distribution():
    """KS test should reject clearly different distributions."""
    real_returns = RNG.normal(0, 1, (50, 100))
    gen_returns = RNG.normal(5, 1, (50, 100))

    result = stylized_facts(real_returns, gen_returns, [1])

    assert result['ks_pvalue'] < 0.01, f"Expected p < 0.01, got {result['ks_pvalue']}"


def test_all_values_are_floats():
    """All returned values should be Python floats."""
    real_returns = RNG.normal(0, 1, (50, 100))
    gen_returns = RNG.normal(0, 1, (50, 100))

    result = stylized_facts(real_returns, gen_returns, [1, 5, 22])

    for key, value in result.items():
        assert isinstance(value, float), f"Key '{key}': expected float, got {type(value)}"

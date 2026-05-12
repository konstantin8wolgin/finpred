"""Stylized-facts battery for generated vs real returns (secondary metric).

STUB (Phase B). Implement, on a batch of generated return series and a matched real sample:
  - kurtosis of returns (heavy tails),
  - autocorrelation of |returns| at `eval.stylized_facts.acf_lags` (volatility clustering),
  - correlation of returns_t with future realized vol (leverage effect),
  - autocorrelation of raw returns (should be ~0),
  - two-sample KS test of the return distributions (report statistic + p-value).
Return a dict of named scalars; the trainer logs these and they go into eval.json.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, ks_2samp


def stylized_facts(real_returns: np.ndarray, gen_returns: np.ndarray, acf_lags: list[int]) -> dict:
    """Compute stylized facts battery for generated vs real return series.

    Args:
        real_returns: shape (N_real, T), real return series
        gen_returns: shape (N_gen, T), generated return series
        acf_lags: list of lag values for ACF computation (e.g. [1, 5, 22])

    Returns:
        dict with keys:
            - gen/real_kurtosis: excess kurtosis (fisher=True)
            - gen/real_acf_abs_lag{k} for each k in acf_lags: ACF of |returns| at lag k
            - gen/real_acf_raw_lag1: ACF of raw returns at lag 1
            - leverage_corr: corr(gen_returns[:-1], gen_returns[1:]**2)
            - ks_stat, ks_pvalue: two-sample KS test
    """
    real_flat = real_returns.flatten()
    gen_flat = gen_returns.flatten()

    result = {}

    # Kurtosis (excess)
    result['gen_kurtosis'] = float(kurtosis(gen_flat, fisher=True))
    result['real_kurtosis'] = float(kurtosis(real_flat, fisher=True))

    # ACF of absolute returns for each lag
    for lag in acf_lags:
        gen_abs = np.abs(gen_flat)
        real_abs = np.abs(real_flat)
        gen_acf = float(np.corrcoef(gen_abs[:-lag], gen_abs[lag:])[0, 1])
        real_acf = float(np.corrcoef(real_abs[:-lag], real_abs[lag:])[0, 1])
        result[f'gen_acf_abs_lag{lag}'] = gen_acf
        result[f'real_acf_abs_lag{lag}'] = real_acf

    # ACF of raw returns at lag 1
    gen_raw_acf1 = float(np.corrcoef(gen_flat[:-1], gen_flat[1:])[0, 1])
    real_raw_acf1 = float(np.corrcoef(real_flat[:-1], real_flat[1:])[0, 1])
    result['gen_acf_raw_lag1'] = gen_raw_acf1
    result['real_acf_raw_lag1'] = real_raw_acf1

    # Leverage correlation (returns[t] vs realized_vol[t+1])
    gen_realized_vol = gen_flat[1:] ** 2
    gen_leverage = float(np.corrcoef(gen_flat[:-1], gen_realized_vol)[0, 1])
    result['leverage_corr'] = gen_leverage

    # KS test
    ks_stat, ks_pvalue = ks_2samp(real_flat, gen_flat)
    result['ks_stat'] = float(ks_stat)
    result['ks_pvalue'] = float(ks_pvalue)

    return result

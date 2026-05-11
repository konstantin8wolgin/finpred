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


def stylized_facts(real_returns: np.ndarray, gen_returns: np.ndarray, acf_lags: list[int]) -> dict:
    """Return {'gen_kurtosis': ..., 'acf_abs_lag1': ..., ..., 'ks_stat': ..., 'ks_pvalue': ...}. STUB."""
    raise NotImplementedError("eval.stylized_facts.stylized_facts — implement in Phase B")

"""Stationary block bootstrap with Politis-White automatic block length.

Daily strategy P&L is autocorrelated (vol clustering, held positions), so
iid resampling understates Sharpe uncertainty. The stationary bootstrap
(Politis-Romano 1994) resamples geometric-length blocks; the expected block
length comes from the Politis-White (2004) automatic selector with Patton's
(2009) correction.

Implementation follows the published estimator: flat-top (trapezoidal)
lag window, correlogram-based bandwidth rule (Politis 2003), and
b_opt = (2 g^2 / D_SB)^(1/3) N^(1/3) for the stationary bootstrap.
Unit tests pin behavior on iid vs AR(1) series rather than paper tables.
"""

from __future__ import annotations

import numpy as np


def _flat_top(t: np.ndarray) -> np.ndarray:
    """Trapezoidal lag window: 1 on |t|<=1/2, linear to 0 at |t|=1."""
    at = np.abs(t)
    return np.clip(2.0 * (1.0 - at), 0.0, 1.0) * (at <= 1.0) + 0.0 * at


def politis_white_block_length(x: np.ndarray, b_max_frac: float = 0.25) -> float:
    """Automatic expected block length for the stationary bootstrap."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20:
        return 1.0
    xc = x - x.mean()

    # Bandwidth m_hat: smallest lag after which K_n consecutive sample
    # autocorrelations are all inside the +/- c*sqrt(log10(n)/n) band.
    kn = max(5, int(np.sqrt(np.log10(n))))
    c = 2.0
    max_lag = min(n - 2, int(np.ceil(np.sqrt(n))) + kn)
    denom = float(np.dot(xc, xc))
    if denom == 0.0:
        return 1.0
    rho = np.array([np.dot(xc[: n - k], xc[k:]) / denom for k in range(1, max_lag + 1)])
    band = c * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(0, max_lag - kn + 1):
        if np.all(np.abs(rho[m : m + kn]) < band):
            m_hat = m
            break
    if m_hat is None:
        m_hat = max_lag - kn
    big_m = min(max(2 * m_hat, 1), max_lag)

    lags = np.arange(-big_m, big_m + 1)
    gamma = np.array(
        [np.dot(xc[: n - abs(k)], xc[abs(k):]) / n for k in lags]
    )
    w = _flat_top(lags / big_m)
    g_hat = float(np.sum(w * gamma))
    d_sb = 2.0 * g_hat**2  # stationary-bootstrap D (Patton correction uses
    # the long-run variance g_hat = sum of weighted autocovariances)
    g_deriv = float(np.sum(w * np.abs(lags) * gamma))
    if d_sb <= 0 or g_deriv == 0:
        return 1.0
    b = ((2.0 * g_deriv**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, b_max_frac * n))


def stationary_bootstrap_indices(
    n: int, expected_block: float, n_boot: int, seed: int = 7
) -> np.ndarray:
    """(n_boot, n) index matrix; geometric block lengths, circular wrap."""
    rng = np.random.default_rng(seed)
    p = 1.0 / max(expected_block, 1.0)
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random(size=(n_boot, n)) < p
    new_block[:, 0] = True
    idx = np.zeros((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        cur = 0
        for t in range(n):
            if new_block[b, t]:
                cur = starts[b, t]
            else:
                cur = (cur + 1) % n
            idx[b, t] = cur
    return idx


def sharpe_ci(
    returns: np.ndarray,
    ann_factor: float = 365.0,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict:
    """Annualized Sharpe with a stationary-block-bootstrap percentile CI."""
    x = np.asarray(returns, dtype=float)
    n = len(x)
    block = politis_white_block_length(x)
    idx = stationary_bootstrap_indices(n, block, n_boot, seed)
    samples = x[idx]
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    valid = sd > 0
    sh = np.sqrt(ann_factor) * mu[valid] / sd[valid]
    point = float(np.sqrt(ann_factor) * x.mean() / x.std(ddof=1))
    lo, hi = np.percentile(sh, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "block_length": float(block),
        "n_boot": int(valid.sum()),
        "alpha": alpha,
    }

"""Peaks-over-threshold extreme value tail model used by the GARCH.EVT
forecaster of Kratz, Lok & McNeil, Sec. 3.3.1 ("uses EVT tail model on
residuals"). The paper does not specify the exact EVT variant; this module
follows the standard POT approach of McNeil, Frey & Embrechts (2015), Ch. 5,
which the paper's bibliography cites for VaR/ES estimation background.

`scipy.stats.genpareto` provides the GPD fit and quantile function used
below (`fit`, `ppf`); the POT tail-quantile formula itself is textbook
risk-management methodology, implemented here following McNeil, Frey &
Embrechts (2015), Eq. (5.15)-(5.16).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fit_gpd_tail(sample: np.ndarray, tail_fraction: float) -> tuple[float, float, float]:
    """Fit a Generalized Pareto Distribution to the exceedances of `sample`
    over its empirical (1 - tail_fraction) quantile (the POT threshold).

    Returns (threshold, xi, beta): the threshold u, GPD shape xi and scale
    beta estimated by maximum likelihood (McNeil et al., 2015, Sec. 5.2.2).
    """
    sample = np.asarray(sample, dtype=float)
    threshold = np.quantile(sample, 1.0 - tail_fraction)
    exceedances = sample[sample > threshold] - threshold
    assert exceedances.size >= 10, "Too few tail exceedances to fit a GPD reliably"
    xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0.0)
    return float(threshold), float(xi), float(beta)


def evt_quantile(alpha: np.ndarray, sample: np.ndarray, tail_fraction: float) -> np.ndarray:
    """POT quantile estimator (McNeil et al., 2015, Eq. 5.16):

        VaR_alpha = u + (beta/xi) * [((1-alpha) / tail_fraction)^(-xi) - 1]

    for alpha above the (1 - tail_fraction) threshold level. Levels below
    the threshold fall back to the plain empirical quantile of `sample`,
    since the GPD approximation is only valid in the tail.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    threshold, xi, beta = fit_gpd_tail(sample, tail_fraction)
    out = np.empty_like(alpha)
    in_tail = alpha > (1.0 - tail_fraction)
    exceed_prob = (1.0 - alpha[in_tail]) / tail_fraction
    if abs(xi) < 1e-8:
        out[in_tail] = threshold - beta * np.log(exceed_prob)
    else:
        out[in_tail] = threshold + (beta / xi) * (exceed_prob ** (-xi) - 1.0)
    out[~in_tail] = np.quantile(sample, alpha[~in_tail])
    return out

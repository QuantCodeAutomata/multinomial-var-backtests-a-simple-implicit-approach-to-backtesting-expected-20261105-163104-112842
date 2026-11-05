"""Maximum-likelihood fits of the static (iid) loss distributions used by
the "good" and "poor" modellers of Sec. 3.2.1: the good modeller estimates
the correct distribution family by MLE on the rolling window, the poor
modeller always fits a normal distribution regardless of the true family.

Library search (Context7): `scipy.stats.norm.fit` and `scipy.stats.t.fit`
cover the normal and (unstandardized) Student t MLE fits directly. No
library fits the Fernandez & Steel (1998) skewed-t location-scale family,
so its MLE is implemented from scratch below (custom implementation,
Fernandez & Steel (1998); paper Sec. 3.1.1 for the family definition).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def fit_normal(window: np.ndarray) -> dict:
    loc, scale = stats.norm.fit(window)
    return {"family": "normal", "loc": loc, "scale": scale}


def fit_student_t(window: np.ndarray) -> dict:
    """Unconstrained MLE of (dof, loc, scale) for a Student t family."""
    dof, loc, scale = stats.t.fit(window)
    return {"family": "t", "dof": dof, "loc": loc, "scale": scale}


def _fs_skew_t_logpdf(x: np.ndarray, dof: float, gamma: float, loc: float, scale: float) -> np.ndarray:
    """Log-density of the Fernandez & Steel (1998) skewed-t location-scale
    family, base density f0 = standard (unstandardized) Student t with
    `dof` degrees of freedom."""
    z = (x - loc) / scale
    sign_adj = np.where(z < 0, gamma, 1.0 / gamma)
    base_logpdf = stats.t.logpdf(z * sign_adj, dof)
    norm_const = np.log(2.0 / (gamma + 1.0 / gamma)) - np.log(scale)
    return norm_const + base_logpdf


def fit_skewed_t(window: np.ndarray) -> dict:
    """MLE of (dof, gamma, loc, scale) for the Fernandez & Steel skewed-t
    family via `scipy.optimize.minimize` on the negative log-likelihood."""
    window = np.asarray(window, dtype=float)

    def neg_loglik(params: np.ndarray) -> float:
        log_dof_m2, log_gamma, loc, log_scale = params
        dof = 2.0 + np.exp(log_dof_m2)
        gamma = np.exp(log_gamma)
        scale = np.exp(log_scale)
        return -np.sum(_fs_skew_t_logpdf(window, dof, gamma, loc, scale))

    x0 = np.array([np.log(3.0), 0.0, float(np.mean(window)), np.log(float(np.std(window)))])
    res = minimize(neg_loglik, x0, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 3000})
    log_dof_m2, log_gamma, loc, log_scale = res.x
    return {
        "family": "skewed_t",
        "dof": 2.0 + np.exp(log_dof_m2),
        "gamma": np.exp(log_gamma),
        "loc": loc,
        "scale": np.exp(log_scale),
    }


FIT_FUNCTIONS = {"normal": fit_normal, "t": fit_student_t, "skewed_t": fit_skewed_t}


def _fs_skew_t_raw_ppf(q: np.ndarray, dof: float, gamma: float) -> np.ndarray:
    """Quantile function of the *standard-scale* (loc=0, scale=1) Fernandez
    & Steel skewed-t, base density f0 = standard Student t with `dof` dof."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    p_negative = 1.0 / (1.0 + gamma ** 2)
    out = np.empty_like(q)
    neg = q < p_negative
    out[neg] = stats.t.ppf(q[neg] * (gamma ** 2 + 1.0) / 2.0, dof) / gamma
    pos = ~neg
    out[pos] = gamma * stats.t.ppf(0.5 + (q[pos] - p_negative) / (2.0 * (1.0 - p_negative)), dof)
    return out


def quantiles_from_fit(fit: dict, alpha: np.ndarray) -> np.ndarray:
    """VaR forecasts F^{-1}(alpha) implied by a fitted distribution dict
    returned by one of the `fit_*` functions above."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    if fit["family"] == "normal":
        return stats.norm.ppf(alpha, loc=fit["loc"], scale=fit["scale"])
    if fit["family"] == "t":
        return fit["loc"] + fit["scale"] * stats.t.ppf(alpha, fit["dof"])
    if fit["family"] == "skewed_t":
        return fit["loc"] + fit["scale"] * _fs_skew_t_raw_ppf(alpha, fit["dof"], fit["gamma"])
    raise ValueError(f"Unknown family: {fit['family']}")

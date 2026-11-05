"""GARCH(1,1)-t data-generating process used in the dynamic backtests of
Sec. 4 ("Dynamic Backtesting"), and the GARCH forecasters that model it.

`arch.univariate` (ConstantMean/ZeroMean with GARCH or ARCH volatility and
a Student's t or Normal error distribution) implements exactly the model
class the paper fits and simulates, so simulation and model fitting both
go through `arch_model` rather than a hand-rolled GARCH recursion.
"""

from __future__ import annotations

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.distributions import StandardizedT
from src.evt import evt_quantile


def simulate_garch_t_losses(
    n: int,
    omega: float,
    alpha1: float,
    beta1: float,
    dof: float,
    rng: np.random.Generator,
    burn_in: int = 500,
) -> np.ndarray:
    """Simulate a GARCH(1,1) process with Student-t(dof) innovations
    (paper Sec. 4, Eq. for alpha_0, alpha_1, beta_1) and return the
    corresponding loss series L_t = -r_t. Because the Student-t innovation
    density is symmetric, L_t follows the identical GARCH(1,1) recursion
    with the sign of every return flipped, so the loss series' conditional
    variance process is unchanged.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(seed)  # arch_model.simulate draws from the global numpy RNG
    am = arch_model(None, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    sim = am.simulate(params=[omega, alpha1, beta1, dof], nobs=n, burn=burn_in)
    returns = sim["data"].to_numpy()
    return -returns


class GarchFilterState:
    """Running one-step-ahead GARCH(1,1) variance filter, seeded from a
    fitted (or known) model and advanced day by day as realized losses
    arrive, so that parameters are updated only every `step` days
    (Sec. 4: "Models re-estimated every 10 steps") while the conditional
    variance itself evolves with every new observation in between.
    """

    def __init__(self, omega: float, alpha1: float, beta1: float, sigma2_prev: float, loss_prev: float):
        self.omega, self.alpha1, self.beta1 = omega, alpha1, beta1
        self.sigma2_prev, self.loss_prev = sigma2_prev, loss_prev

    def next_sigma2(self) -> float:
        """GARCH(1,1) recursion: sigma2_t = omega + alpha1*r_{t-1}^2 + beta1*sigma2_{t-1}."""
        return self.omega + self.alpha1 * self.loss_prev ** 2 + self.beta1 * self.sigma2_prev

    def advance(self, sigma2_t: float, loss_t: float) -> None:
        self.sigma2_prev, self.loss_prev = sigma2_t, loss_t


def _fit_arch(window: np.ndarray, vol: str, p_lags: int, q_lags: int, dist: str) -> ARCHModelResult:
    am = arch_model(window, mean="Zero", vol=vol, p=p_lags, q=q_lags, dist=dist, rescale=False)
    return am.fit(disp="off", show_warning=False)


def fit_garch_t(window: np.ndarray) -> dict:
    res = _fit_arch(window, "GARCH", 1, 1, "t")
    p = res.params
    return {
        "omega": p["omega"], "alpha1": p["alpha[1]"], "beta1": p["beta[1]"], "dof": p["nu"],
        "sigma2_last": float(res.conditional_volatility[-1] ** 2), "loss_last": float(window[-1]),
        "std_resid": res.std_resid,
    }


def fit_garch_norm(window: np.ndarray) -> dict:
    res = _fit_arch(window, "GARCH", 1, 1, "normal")
    p = res.params
    return {
        "omega": p["omega"], "alpha1": p["alpha[1]"], "beta1": p["beta[1]"],
        "sigma2_last": float(res.conditional_volatility[-1] ** 2), "loss_last": float(window[-1]),
        "std_resid": res.std_resid,
    }


def fit_arch1(window: np.ndarray, dist: str) -> dict:
    res = _fit_arch(window, "ARCH", 1, 0, dist)
    p = res.params
    out = {
        "omega": p["omega"], "alpha1": p["alpha[1]"], "beta1": 0.0,
        "sigma2_last": float(res.conditional_volatility[-1] ** 2), "loss_last": float(window[-1]),
    }
    if dist == "t":
        out["dof"] = p["nu"]
        out["std_resid"] = res.std_resid
    return out


def garch_block_quantiles(
    fit: dict, future_losses: np.ndarray, alpha_grid: np.ndarray, quantile_mode: str, evt_tail_fraction: float = 0.1,
) -> np.ndarray:
    """One-step-ahead VaR forecasts for every day of a re-estimation block,
    filtering the GARCH(1,1) variance recursion forward with the realized
    losses of the block (no look-ahead: day t's forecast uses only
    information up to t-1).

    `quantile_mode` selects the innovation-quantile model held fixed for
    the whole block: 't' (Student-t, GARCH.t/ARCH.t), 'normal'
    (GARCH.norm/ARCH.norm), 'hs' (empirical quantile of standardized
    residuals, GARCH.HS) or 'evt' (POT tail model, GARCH.EVT).
    """
    state = GarchFilterState(fit["omega"], fit["alpha1"], fit["beta1"], fit["sigma2_last"], fit["loss_last"])
    out = np.empty((len(future_losses), len(alpha_grid)))
    if quantile_mode == "t":
        innov = StandardizedT(fit["dof"])
        base_q = innov.ppf(alpha_grid)
    elif quantile_mode == "normal":
        from scipy.stats import norm
        base_q = norm.ppf(alpha_grid)
    elif quantile_mode == "hs":
        base_q = np.quantile(fit["std_resid"], alpha_grid)
    elif quantile_mode == "evt":
        base_q = evt_quantile(alpha_grid, fit["std_resid"], evt_tail_fraction)
    else:
        raise ValueError(f"Unknown quantile_mode: {quantile_mode}")
    for t, loss_t in enumerate(future_losses):
        sigma2_t = state.next_sigma2()
        out[t] = np.sqrt(sigma2_t) * base_q
        state.advance(sigma2_t, loss_t)
    return out


def oracle_block_quantiles(
    omega: float, alpha1: float, beta1: float, dof: float,
    window: np.ndarray, future_losses: np.ndarray, alpha_grid: np.ndarray,
) -> np.ndarray:
    """Oracle forecaster: true GARCH(1,1)-t parameters are known exactly;
    the variance filter is seeded by running the true recursion through
    the whole estimation window (unconditional variance start) rather
    than by an estimated fit.
    """
    sigma2 = omega / max(1e-12, 1.0 - alpha1 - beta1)
    loss_prev = float(window[0])
    for loss_t in window[1:]:
        sigma2 = omega + alpha1 * loss_prev ** 2 + beta1 * sigma2
        loss_prev = float(loss_t)
    fit = {"omega": omega, "alpha1": alpha1, "beta1": beta1, "sigma2_last": sigma2, "loss_last": loss_prev, "dof": dof}
    return garch_block_quantiles(fit, future_losses, alpha_grid, quantile_mode="t")

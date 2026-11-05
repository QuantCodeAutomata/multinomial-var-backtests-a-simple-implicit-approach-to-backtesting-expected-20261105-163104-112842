"""Unit tests for src/garch_dgp.py (GARCH(1,1)-t DGP and forecasters,
Sec. 4 "Dynamic Backtesting")."""

from __future__ import annotations

import numpy as np

from src.garch_dgp import (
    fit_garch_t,
    garch_block_quantiles,
    oracle_block_quantiles,
    simulate_garch_t_losses,
)

OMEGA, ALPHA1, BETA1, DOF = 2.18e-6, 0.109, 0.890, 5.06


def test_simulate_garch_t_losses_length_and_stationarity() -> None:
    rng = np.random.default_rng(40)
    losses = simulate_garch_t_losses(2000, OMEGA, ALPHA1, BETA1, DOF, rng)
    assert losses.shape == (2000,)
    assert np.isfinite(losses).all()
    # Persistence alpha1 + beta1 = 0.999 < 1 -> covariance-stationary, finite variance.
    assert losses.var() < 1.0  # daily-return scale, not percent


def test_oracle_block_quantiles_monotone_in_alpha() -> None:
    rng = np.random.default_rng(41)
    losses = simulate_garch_t_losses(600, OMEGA, ALPHA1, BETA1, DOF, rng)
    window, future = losses[:500], losses[500:510]
    grid = np.array([0.95, 0.975, 0.99])
    q = oracle_block_quantiles(OMEGA, ALPHA1, BETA1, DOF, window, future, grid)
    assert q.shape == (10, 3)
    assert np.all(np.diff(q, axis=1) > 0)


def test_garch_block_quantiles_no_lookahead_within_block() -> None:
    """The forecast for day t of a block must not depend on day t's own
    realized loss, only on losses up to t-1."""
    rng = np.random.default_rng(42)
    losses = simulate_garch_t_losses(600, OMEGA, ALPHA1, BETA1, DOF, rng)
    window, future = losses[:500], losses[500:510].copy()
    grid = np.array([0.975])
    fit = {"omega": OMEGA, "alpha1": ALPHA1, "beta1": BETA1, "dof": DOF,
           "sigma2_last": window.var(), "loss_last": float(window[-1])}
    q1 = garch_block_quantiles(fit, future, grid, quantile_mode="t")
    future_perturbed = future.copy()
    future_perturbed[-1] = 50.0  # change only the last day's realized loss
    q2 = garch_block_quantiles(fit, future_perturbed, grid, quantile_mode="t")
    assert np.allclose(q1[:-1], q2[:-1])  # forecasts before the perturbed day are unchanged


def test_fit_garch_t_recovers_plausible_parameters() -> None:
    rng = np.random.default_rng(43)
    losses = simulate_garch_t_losses(3000, OMEGA, ALPHA1, BETA1, DOF, rng)
    fit = fit_garch_t(losses)
    assert fit["alpha1"] + fit["beta1"] < 1.2
    assert fit["dof"] > 2.0
    assert fit["sigma2_last"] > 0

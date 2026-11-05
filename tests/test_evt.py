"""Unit tests for src/evt.py (GARCH.EVT forecaster tail model, Sec. 3.3.1)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.evt import evt_quantile, fit_gpd_tail


def test_fit_gpd_tail_on_known_pareto_tail() -> None:
    rng = np.random.default_rng(20)
    sample = stats.t.rvs(4, size=5000, random_state=rng)
    threshold, xi, beta = fit_gpd_tail(sample, tail_fraction=0.1)
    assert threshold > np.median(sample)
    assert xi > -0.5  # Student-t tails are regularly varying, xi > 0 expected
    assert beta > 0


def test_evt_quantile_monotone_and_above_threshold() -> None:
    rng = np.random.default_rng(21)
    sample = stats.t.rvs(4, size=5000, random_state=rng)
    alphas = np.array([0.95, 0.975, 0.99, 0.999])
    q = evt_quantile(alphas, sample, tail_fraction=0.1)
    assert np.all(np.diff(q) > 0)


def test_evt_quantile_below_threshold_matches_empirical() -> None:
    rng = np.random.default_rng(22)
    sample = rng.standard_normal(5000)
    alpha = np.array([0.5])
    q = evt_quantile(alpha, sample, tail_fraction=0.1)
    assert np.isclose(q[0], np.quantile(sample, 0.5))

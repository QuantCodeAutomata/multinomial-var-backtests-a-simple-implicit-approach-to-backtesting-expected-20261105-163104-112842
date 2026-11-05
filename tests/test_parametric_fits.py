"""Unit tests for src/parametric_fits.py (good/poor modeller MLE fits,
Sec. 3.2.1)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.parametric_fits import fit_normal, fit_skewed_t, fit_student_t, quantiles_from_fit


def test_fit_normal_recovers_parameters() -> None:
    rng = np.random.default_rng(10)
    sample = rng.normal(loc=0.3, scale=1.7, size=20_000)
    fit = fit_normal(sample)
    assert abs(fit["loc"] - 0.3) < 0.05
    assert abs(fit["scale"] - 1.7) < 0.05


def test_fit_student_t_recovers_dof() -> None:
    rng = np.random.default_rng(11)
    sample = stats.t.rvs(5, size=30_000, random_state=rng)
    fit = fit_student_t(sample)
    assert 3.5 < fit["dof"] < 8.0


def test_quantiles_from_fit_normal_matches_scipy() -> None:
    fit = {"family": "normal", "loc": 0.0, "scale": 2.0}
    q = quantiles_from_fit(fit, np.array([0.1, 0.5, 0.975]))
    assert np.allclose(q, stats.norm.ppf([0.1, 0.5, 0.975], loc=0.0, scale=2.0))


def test_fit_skewed_t_runs_and_is_skewed() -> None:
    rng = np.random.default_rng(12)
    from src.distributions import get_distribution

    dist = get_distribution("skt3")
    sample = dist.rvs(5000, rng)
    fit = fit_skewed_t(sample)
    assert fit["family"] == "skewed_t"
    assert fit["dof"] > 2.0
    assert fit["gamma"] > 0.0


def test_quantiles_monotone_in_alpha() -> None:
    fit = {"family": "t", "loc": 0.0, "scale": 1.0, "dof": 5.0}
    q = quantiles_from_fit(fit, np.array([0.1, 0.5, 0.9, 0.99]))
    assert np.all(np.diff(q) > 0)

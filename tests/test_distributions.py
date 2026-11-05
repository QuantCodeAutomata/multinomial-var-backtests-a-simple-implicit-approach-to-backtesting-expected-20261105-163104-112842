"""Unit tests for src/distributions.py (Kratz, Lok & McNeil, Sec. 3.1.1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.distributions import DISTRIBUTIONS, expected_shortfall, get_distribution, value_at_risk


@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_mean_zero_variance_one(name: str) -> None:
    """Every distribution is calibrated to mean 0, variance 1 (Sec. 3.1.1)."""
    rng = np.random.default_rng(0)
    dist = get_distribution(name)
    sample = dist.rvs(400_000, rng)
    assert abs(sample.mean()) < 0.02
    assert abs(sample.var() - 1.0) < 0.05


@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_ppf_cdf_roundtrip(name: str) -> None:
    dist = get_distribution(name)
    q = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    x = dist.ppf(q)
    back = np.atleast_1d(dist.cdf(x))
    assert np.allclose(back, q, atol=1e-6)


def test_skewed_t_is_skewed() -> None:
    """gamma = 1.2 != 1 must break the symmetry of the base Student t."""
    dist = get_distribution("skt3")
    median = dist.ppf(np.array([0.5]))[0]
    assert abs(median) > 1e-6


def test_var_monotone_in_alpha() -> None:
    dist = get_distribution("t3")
    v1 = value_at_risk(dist, 0.975)
    v2 = value_at_risk(dist, 0.99)
    assert v2 > v1


def test_expected_shortfall_exceeds_var() -> None:
    """ES_alpha >= VaR_alpha for any alpha (Sec. 1, Eq. 1.1)."""
    for name in DISTRIBUTIONS:
        dist = get_distribution(name)
        alpha = 0.975
        assert expected_shortfall(dist, alpha) > value_at_risk(dist, alpha)


def test_normal_matches_scipy() -> None:
    from scipy.stats import norm

    dist = get_distribution("normal")
    q = np.array([0.1, 0.5, 0.9])
    assert np.allclose(dist.ppf(q), norm.ppf(q))

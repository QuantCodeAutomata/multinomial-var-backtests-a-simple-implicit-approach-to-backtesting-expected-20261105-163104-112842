"""Unit tests for src/multinomial_tests.py (Kratz, Lok & McNeil, Sec. 2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.multinomial_tests import (
    alpha_levels,
    binomial_score_test_one_sided,
    binomial_tests,
    bin_counts_from_probabilities,
    full_alpha_grid,
    lrt_test,
    nass_adjustment,
    nass_test,
    pearson_test,
    traffic_light,
)


def test_alpha_levels_eq_2_1() -> None:
    """alpha_j = alpha + (j-1)/N (1-alpha), j=1..N (Eq. 2.1)."""
    levels = alpha_levels(0.975, 4)
    expected = 0.975 + (np.arange(4)) / 4 * 0.025
    assert np.allclose(levels, expected)
    assert levels[0] == 0.975
    assert levels[-1] < 1.0


def test_full_alpha_grid_endpoints() -> None:
    grid = full_alpha_grid(0.975, 8)
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert len(grid) == 10
    assert np.all(np.diff(grid) > 0)


def test_bin_counts_sum_to_n() -> None:
    rng = np.random.default_rng(1)
    grid = full_alpha_grid(0.975, 4)
    probs = np.diff(grid)
    counts = bin_counts_from_probabilities(1000, probs, 50, rng)
    assert counts.shape == (50, 5)
    assert np.all(counts.sum(axis=-1) == 1000)


def test_pearson_test_size_near_nominal() -> None:
    """Under H0 (true model = bin model), rejection rate ~ kappa (Sec. 3.1.1)."""
    rng = np.random.default_rng(2)
    n_levels = 4
    grid = full_alpha_grid(0.975, n_levels)
    probs = np.diff(grid)
    counts = bin_counts_from_probabilities(1000, probs, 20_000, rng)
    _, _, reject = pearson_test(counts, grid, kappa=0.05)
    assert 0.02 < reject.mean() < 0.09


def test_nass_adjustment_matches_formula() -> None:
    grid = full_alpha_grid(0.975, 8)
    n = 1000
    c, nu = nass_adjustment(grid, n)
    n_levels = 8
    p = np.diff(grid)
    e_sn = n_levels
    var_sn = 2 * n_levels - (n_levels ** 2 + 4 * n_levels + 1) / n + (1.0 / n) * np.sum(1.0 / p)
    assert np.isclose(c, 2 * e_sn / var_sn)
    assert np.isclose(nu, c * e_sn)
    assert nu > 0


def test_nass_more_stable_than_pearson_for_large_N() -> None:
    """Expected outcome (Sec. 3.1.1): Pearson size deteriorates for N>=8,
    Nass stays close to the nominal level."""
    rng = np.random.default_rng(3)
    n_levels = 32
    grid = full_alpha_grid(0.975, n_levels)
    probs = np.diff(grid)
    counts = bin_counts_from_probabilities(500, probs, 5_000, rng)
    _, _, reject_pearson = pearson_test(counts, grid, kappa=0.05)
    _, _, reject_nass = nass_test(counts, grid, kappa=0.05)
    assert abs(reject_nass.mean() - 0.05) < abs(reject_pearson.mean() - 0.05)


def test_lrt_statistic_vectorized_matches_scalar_reference() -> None:
    from src.multinomial_tests import lrt_statistic_scalar, lrt_statistic_vectorized

    rng = np.random.default_rng(4)
    n_levels = 4
    grid = full_alpha_grid(0.975, n_levels)
    probs = np.diff(grid)
    counts = bin_counts_from_probabilities(1000, probs, 5, rng)
    vec_stat = lrt_statistic_vectorized(counts.astype(float), grid, n_iter=500, lr=0.2)
    for i in range(counts.shape[0]):
        scalar_stat, _, _ = lrt_statistic_scalar(counts[i], grid)
        assert abs(vec_stat[i] - scalar_stat) < 0.5


def test_lrt_statistic_nonnegative() -> None:
    rng = np.random.default_rng(5)
    grid = full_alpha_grid(0.975, 8)
    probs = np.diff(grid)
    counts = bin_counts_from_probabilities(1000, probs, 50, rng)
    stat, p_value, _ = lrt_test(counts.astype(float), grid)
    assert np.all(stat > -1e-6)
    assert np.all((p_value >= 0) & (p_value <= 1))


def test_binomial_tests_reject_under_strong_alternative() -> None:
    """A much larger-than-expected exception count must be rejected by
    every binomial test variant."""
    n, alpha = 1000, 0.975
    exceptions = np.array([80])  # true p0 = 0.025 -> 25 expected
    results = binomial_tests(exceptions, n, alpha)
    for name, res in results.items():
        assert res.reject[0], f"{name} failed to reject a strong alternative"


def test_binomial_tests_size_near_nominal() -> None:
    rng = np.random.default_rng(6)
    n, alpha = 1000, 0.975
    exceptions = rng.binomial(n, 1 - alpha, size=20_000)
    results = binomial_tests(exceptions, n, alpha, kappa=0.05)
    for name, res in results.items():
        assert 0.0 <= res.reject.mean() <= 0.12, f"{name} rejection rate out of range"


def test_binomial_score_one_sided_matches_formula() -> None:
    n, alpha = 1000, 0.99
    exceptions = np.array([20])
    z, p = binomial_score_test_one_sided(exceptions, n, alpha)
    z_expected = (20 - n * 0.01) / np.sqrt(n * 0.01 * 0.99)
    assert np.isclose(z, z_expected)
    assert np.isclose(p, stats.norm.sf(z_expected))


def test_traffic_light_thresholds() -> None:
    colours = traffic_light(np.array([0.5, 0.01, 0.00001]))
    assert list(colours) == ["green", "yellow", "red"]

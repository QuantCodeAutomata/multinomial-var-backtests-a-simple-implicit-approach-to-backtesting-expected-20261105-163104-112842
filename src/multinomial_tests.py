"""Binomial and multinomial VaR backtests of Kratz, Lok & McNeil,
"Multinomial VaR Backtests: A simple implicit approach to backtesting
expected shortfall", Section 2.

statsmodels and scipy.stats provide generic chi-squared / binomial tests
but not the Nass (1959) small-sample correction or the constrained
normal-family LRT used in this paper, so both are implemented from
scratch, exactly as specified in Sec. 2.2 and
Sec. 3.1.1. The binomial Wald/score/LRT tests are likewise implemented
directly from their textbook formulas (no library exposes the specific
one- and two-sided variants compared in Table 2 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Quantile levels (Sec. 2.1, Eq. 2.1)
# ---------------------------------------------------------------------------
def alpha_levels(alpha: float, n_levels: int) -> np.ndarray:
    """alpha_1, ..., alpha_N as defined by Eq. (2.1):
    alpha_j = alpha + (j-1)/N * (1 - alpha), j = 1, ..., N."""
    j = np.arange(1, n_levels + 1)
    return alpha + (j - 1) / n_levels * (1 - alpha)


def full_alpha_grid(alpha: float, n_levels: int) -> np.ndarray:
    """alpha_0 = 0, alpha_1, ..., alpha_N, alpha_{N+1} = 1 (the N+1 bin
    edges used to count observations O_0, ..., O_N)."""
    mid = alpha_levels(alpha, n_levels)
    return np.concatenate([[0.0], mid, [1.0]])


def bin_counts_from_probabilities(
    n: int, cell_probabilities: np.ndarray, n_reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw (O_0, ..., O_N) ~ Multinomial(n, cell_probabilities) directly.

    This reproduces the paper's Monte Carlo design exactly: under either the
    null (size) or a specific alternative (power), the vector of bin counts
    is by construction multinomially distributed with these cell
    probabilities (Sec. 3.1.1, Eq. 2.3), so sampling it directly instead of
    simulating a full continuous sample and binning it gives identical
    finite-sample behaviour of every test statistic (which depends on the
    data only through the counts) at a fraction of the computational cost.
    Shape of the returned array: (n_reps, len(cell_probabilities)).
    """
    assert abs(cell_probabilities.sum() - 1.0) < 1e-8
    return rng.multinomial(n, cell_probabilities, size=n_reps)


def cell_probabilities_from_quantile_levels(theta: np.ndarray) -> np.ndarray:
    """theta is the full grid (theta_0=0, theta_1, ..., theta_N, theta_{N+1}=1)
    of true probabilities associated with the quantile estimates; returns
    the N+1 cell probabilities theta_{j+1} - theta_j."""
    return np.diff(theta)


# ---------------------------------------------------------------------------
# Pearson chi-squared test (Sec. 2.2.1, Eq. 2.4)
# ---------------------------------------------------------------------------
def pearson_statistic(counts: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    """S_N = sum_{j=0}^{N} (O_{j+1} - n * p_j)^2 / (n * p_j), p_j = alpha_{j+1}-alpha_j.

    `counts` has shape (..., N+1); the test statistic is computed along the
    last axis and broadcasts over any leading (replication) dimensions.
    """
    n = counts.sum(axis=-1, keepdims=True)
    p = np.diff(alpha_grid)
    expected = n * p
    return np.sum((counts - expected) ** 2 / expected, axis=-1)


def pearson_test(counts: np.ndarray, alpha_grid: np.ndarray, kappa: float = 0.05):
    """Reject H0 when S_N > chi2_N(1-kappa). Returns (statistic, p_value, reject)."""
    n_levels = len(alpha_grid) - 2
    stat = pearson_statistic(counts, alpha_grid)
    p_value = stats.chi2.sf(stat, df=n_levels)
    return stat, p_value, p_value < kappa


# ---------------------------------------------------------------------------
# Nass test (Nass, 1959; Sec. 2.2.2)
# ---------------------------------------------------------------------------
def nass_adjustment(alpha_grid: np.ndarray, n: int) -> tuple[float, float]:
    """Return (c, nu) with c = 2 E(S_N) / var(S_N), nu = c * E(S_N),
    E(S_N) = N, var(S_N) = 2N - (N^2+4N+1)/n + (1/n) sum_{j=0}^N 1/p_j."""
    n_levels = len(alpha_grid) - 2
    p = np.diff(alpha_grid)
    e_sn = n_levels
    var_sn = 2 * n_levels - (n_levels ** 2 + 4 * n_levels + 1) / n + (1.0 / n) * np.sum(1.0 / p)
    c = 2 * e_sn / var_sn
    nu = c * e_sn
    return c, nu


def nass_test(counts: np.ndarray, alpha_grid: np.ndarray, kappa: float = 0.05):
    """Reject H0 when c * S_N > chi2_nu(1-kappa)."""
    n = int(counts.sum(axis=-1).reshape(-1)[0])  # n is constant across replications
    c, nu = nass_adjustment(alpha_grid, n)
    stat = pearson_statistic(counts, alpha_grid)
    adjusted = c * stat
    p_value = stats.chi2.sf(adjusted, df=nu)
    return adjusted, p_value, p_value < kappa


# ---------------------------------------------------------------------------
# Likelihood-ratio test (Sec. 2.2.3)
# ---------------------------------------------------------------------------
def _neg_loglik_scalar(params: np.ndarray, counts: np.ndarray, z_alpha: np.ndarray) -> float:
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    theta = stats.norm.cdf((z_alpha - mu) / sigma)
    theta_full = np.concatenate([[0.0], theta, [1.0]])
    p = np.diff(theta_full)
    p = np.clip(p, 1e-300, None)
    return -np.sum(counts * np.log(p))


def lrt_statistic_scalar(counts: np.ndarray, alpha_grid: np.ndarray) -> tuple[float, float, float]:
    """Reference (non-vectorized) computation of the LRT statistic via
    `scipy.optimize.minimize`, used to validate the vectorized Newton
    solver below. Returns (statistic, mu_hat, sigma_hat)."""
    z_alpha = stats.norm.ppf(alpha_grid[1:-1])
    res = minimize(
        _neg_loglik_scalar, x0=np.array([0.0, 0.0]), args=(counts, z_alpha), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000},
    )
    mu_hat, sigma_hat = res.x[0], np.exp(res.x[1])
    alpha_diff = np.diff(alpha_grid)
    theta_full = np.concatenate([[0.0], stats.norm.cdf((z_alpha - mu_hat) / sigma_hat), [1.0]])
    theta_diff = np.diff(theta_full)
    stat = 2 * np.sum(counts * np.log(np.clip(theta_diff, 1e-300, None) / alpha_diff))
    return stat, mu_hat, sigma_hat


def fit_normal_mle_vectorized(
    counts: np.ndarray, z_alpha: np.ndarray, n_iter: int = 300, lr: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood estimates (mu_hat, sigma_hat) of the N(mu, sigma)
    model underlying theta_j = Phi((z_alpha_j - mu)/sigma), for a whole
    batch of replications at once (vectorized Adam gradient ascent on the
    multinomial log-likelihood, Sec. 2.2.3, equivalent to the
    `scipy.optimize.minimize` MLE described in the paper but applied to all
    replications simultaneously for tractability at Monte Carlo scale).

    `counts` has shape (R, N+1); `z_alpha` has shape (N,). Gradients of the
    log-likelihood w.r.t. mu and log_sigma are computed analytically.
    """
    counts = np.asarray(counts, dtype=float)
    r = counts.shape[0]
    mu = np.zeros(r)
    log_sigma = np.zeros(r)
    m_mu = np.zeros(r); v_mu = np.zeros(r)
    m_ls = np.zeros(r); v_ls = np.zeros(r)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, n_iter + 1):
        sigma = np.exp(log_sigma)
        z = (z_alpha[None, :] - mu[:, None]) / sigma[:, None]      # (R, N)
        phi = stats.norm.pdf(z)
        theta = stats.norm.cdf(z)                                   # (R, N)
        theta_full = np.concatenate(
            [np.zeros((r, 1)), theta, np.ones((r, 1))], axis=1
        )                                                            # (R, N+2)
        p = np.diff(theta_full, axis=1)                               # (R, N+1)
        p = np.clip(p, 1e-12, None)
        inv_p = counts / p                                            # O_j / p_j, shape (R, N+1)

        # d theta_k / d mu = -phi_k / sigma ; d theta_k / d log_sigma = -phi_k * z_k
        dtheta_dmu = -phi / sigma[:, None]                            # (R, N)
        dtheta_dls = -phi * z                                         # (R, N)

        # dp_j/d. = dtheta_{j+1}/d. - dtheta_j/d. with dtheta_0 = dtheta_{N+1} = 0
        dtheta_dmu_full = np.concatenate(
            [np.zeros((r, 1)), dtheta_dmu, np.zeros((r, 1))], axis=1
        )
        dtheta_dls_full = np.concatenate(
            [np.zeros((r, 1)), dtheta_dls, np.zeros((r, 1))], axis=1
        )
        dp_dmu = np.diff(dtheta_dmu_full, axis=1)                      # (R, N+1)
        dp_dls = np.diff(dtheta_dls_full, axis=1)                      # (R, N+1)

        grad_mu = np.sum(inv_p * dp_dmu, axis=1)
        grad_ls = np.sum(inv_p * dp_dls, axis=1)

        m_mu[:] = beta1 * m_mu + (1 - beta1) * grad_mu
        v_mu[:] = beta2 * v_mu + (1 - beta2) * grad_mu ** 2
        m_ls[:] = beta1 * m_ls + (1 - beta1) * grad_ls
        v_ls[:] = beta2 * v_ls + (1 - beta2) * grad_ls ** 2
        mhat_mu = m_mu / (1 - beta1 ** t); vhat_mu = v_mu / (1 - beta2 ** t)
        mhat_ls = m_ls / (1 - beta1 ** t); vhat_ls = v_ls / (1 - beta2 ** t)

        mu = mu + lr * mhat_mu / (np.sqrt(vhat_mu) + eps)
        log_sigma = log_sigma + lr * mhat_ls / (np.sqrt(vhat_ls) + eps)
        log_sigma = np.clip(log_sigma, -3.0, 3.0)
        mu = np.clip(mu, -10.0, 10.0)

    return mu, np.exp(log_sigma)


def lrt_statistic_vectorized(
    counts: np.ndarray, alpha_grid: np.ndarray, n_iter: int = 300, lr: float = 0.3
) -> np.ndarray:
    """Vectorized computation of the LRT statistic (Sec. 2.2.3):

        S~_N = 2 sum_{j=0}^{N} O_j log((theta_hat_{j+1}-theta_hat_j) / (alpha_{j+1}-alpha_j))

    where theta_hat is obtained by maximizing the multinomial log-likelihood
    over (mu, sigma) of a normal model, for every replication in `counts`
    (shape (R, N+1)) simultaneously.
    """
    z_alpha = stats.norm.ppf(alpha_grid[1:-1])
    mu_hat, sigma_hat = fit_normal_mle_vectorized(counts, z_alpha, n_iter=n_iter, lr=lr)
    r = counts.shape[0]
    z = (z_alpha[None, :] - mu_hat[:, None]) / sigma_hat[:, None]
    theta = stats.norm.cdf(z)
    theta_full = np.concatenate([np.zeros((r, 1)), theta, np.ones((r, 1))], axis=1)
    theta_diff = np.clip(np.diff(theta_full, axis=1), 1e-300, None)
    alpha_diff = np.diff(alpha_grid)
    stat = 2 * np.sum(counts * np.log(theta_diff / alpha_diff[None, :]), axis=1)
    return stat


def lrt_test(counts: np.ndarray, alpha_grid: np.ndarray, kappa: float = 0.05, n_iter: int = 300):
    """Multinomial LRT: reject H0 when S~_N > chi2_2(1-kappa) (two estimated
    parameters mu, sigma; Sec. 2.2.3 and implementation note on p.10)."""
    stat = lrt_statistic_vectorized(counts, alpha_grid, n_iter=n_iter)
    p_value = stats.chi2.sf(stat, df=2)
    return stat, p_value, p_value < kappa


# ---------------------------------------------------------------------------
# Binomial tests of VaR exceptions (Sec. 3.1.2 / Table 2): Wald, score, LRT,
# both one-sided and two-sided.
# ---------------------------------------------------------------------------
@dataclass
class BinomialTestResult:
    statistic: np.ndarray
    p_value: np.ndarray
    reject: np.ndarray


def binomial_tests(exceptions: np.ndarray, n: int, alpha: float, kappa: float = 0.05):
    """exceptions: array of observed exception counts B (number of losses
    exceeding VaR_alpha), shape (R,). Returns a dict of BinomialTestResult
    for {"wald","score","lrt"} x {"one_sided","two_sided"}."""
    p0 = 1 - alpha  # nominal exception probability
    phat = exceptions / n
    results = {}

    # Wald test: (phat - p0) / sqrt(phat(1-phat)/n) ~ N(0,1)
    se_wald = np.sqrt(np.clip(phat * (1 - phat), 1e-12, None) / n)
    z_wald = (phat - p0) / se_wald

    # Score test: (phat - p0) / sqrt(p0(1-p0)/n) ~ N(0,1)
    se_score = np.sqrt(p0 * (1 - p0) / n)
    z_score = (phat - p0) / se_score

    # Binomial LRT (two-sided): 2 n [phat log(phat/p0) + (1-phat) log((1-phat)/(1-p0))] ~ chi2_1
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(exceptions > 0, exceptions * np.log(phat / p0), 0.0)
        term2 = np.where(exceptions < n, (n - exceptions) * np.log((1 - phat) / (1 - p0)), 0.0)
    lrt_stat_two_sided = 2 * (term1 + term2)

    for name, z in (("wald", z_wald), ("score", z_score)):
        chi2_stat = z ** 2
        results[f"{name}_two_sided"] = BinomialTestResult(
            chi2_stat, stats.chi2.sf(chi2_stat, df=1), stats.chi2.sf(chi2_stat, df=1) < kappa
        )
        # one-sided: reject only for an excess of exceptions (upper tail)
        p_one = stats.norm.sf(z)
        results[f"{name}_one_sided"] = BinomialTestResult(z, p_one, p_one < kappa)

    results["lrt_two_sided"] = BinomialTestResult(
        lrt_stat_two_sided, stats.chi2.sf(lrt_stat_two_sided, df=1),
        stats.chi2.sf(lrt_stat_two_sided, df=1) < kappa,
    )
    sign = np.sign(phat - p0)
    one_sided_lrt_stat = np.where(sign >= 0, lrt_stat_two_sided, 0.0)
    p_lrt_one = stats.chi2.sf(one_sided_lrt_stat, df=1) / 2.0
    p_lrt_one = np.where(sign >= 0, p_lrt_one, 1.0)
    results["lrt_one_sided"] = BinomialTestResult(one_sided_lrt_stat, p_lrt_one, p_lrt_one < kappa)
    return results


def binomial_score_test_one_sided(exceptions: np.ndarray, n: int, alpha: float):
    """One-sided score test used in Sec. 5 real-data application:
    Z = (B - n(1-alpha)) / sqrt(n alpha (1-alpha)); reject for large Z."""
    p0 = 1 - alpha
    z = (exceptions - n * p0) / np.sqrt(n * alpha * p0)
    p_value = stats.norm.sf(z)
    return z, p_value


# ---------------------------------------------------------------------------
# Traffic-light system (Sec. 4)
# ---------------------------------------------------------------------------
def traffic_light(p_value, yellow_threshold: float = 0.05, red_threshold: float = 0.0001):
    """Green if p >= yellow_threshold, yellow if red_threshold <= p < yellow_threshold,
    red if p < red_threshold (Sec. 4 and Sec. 5)."""
    p_value = np.asarray(p_value)
    colour = np.where(
        p_value < red_threshold, "red", np.where(p_value < yellow_threshold, "yellow", "green")
    )
    return colour

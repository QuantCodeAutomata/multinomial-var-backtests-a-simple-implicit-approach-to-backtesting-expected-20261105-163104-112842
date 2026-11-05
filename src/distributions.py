"""Loss distributions used in the simulation study of Kratz, Lok & McNeil,
Section 3.1.1: Standard Normal, Student t5, Student t3, and the skewed
Student t3 of Fernandez & Steel (1998) with skewness parameter gamma.

All four distributions are calibrated to have mean zero and variance one,
as stated in the paper ("These distributions have all been calibrated to
have mean zero and variance one.", Sec. 3.1.1).

Library search (Context7): scipy.stats provides `norm` and `t`, used
directly below. No maintained Python library implements the Fernandez &
Steel (1998) skew transform with the exact standardisation used in this
paper, so it is implemented from scratch (custom implementation, paper
Sec. 3.1.1 and Fernandez & Steel (1998), eq. for first two moments of the
skewed family).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import stats
from scipy.special import gammaln


@dataclass(frozen=True)
class StandardizedT:
    """Student t distribution scaled to have unit variance (mean 0)."""

    dof: float
    name: str = field(default="t", compare=False)

    @property
    def scale(self) -> float:
        assert self.dof > 2, "Variance of a Student t distribution requires dof > 2"
        return np.sqrt((self.dof - 2) / self.dof)

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return stats.t.rvs(self.dof, size=size, random_state=rng) * self.scale

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return stats.t.ppf(q, self.dof) * self.scale

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.t.cdf(np.asarray(x) / self.scale, self.dof)


def _t_abs_moment(dof: float) -> float:
    """E|Z| for a *standardized* (unit-variance) Student t with `dof` degrees
    of freedom. Closed form used by Fernandez & Steel (1998)-type skew
    transforms, e.g. as implemented in the fGarch/rugarch skewed-t
    parameterisation (custom implementation, no standard library ships it)."""
    raw_t_abs_mean = (
        2 * np.sqrt(dof) * np.exp(gammaln((dof + 1) / 2) - gammaln(dof / 2))
        / ((dof - 1) * np.sqrt(np.pi))
    )
    scale = np.sqrt((dof - 2) / dof)
    return raw_t_abs_mean * scale


@dataclass(frozen=True)
class SkewedStandardizedT:
    """Fernandez & Steel (1998) skewed Student t, standardized to mean 0,
    variance 1 (paper Sec. 3.1.1: "skewed Student distribution of Fernandez
    & Steel (1998) with 3 degrees of freedom and a skewness parameter
    gamma = 1.2").

    Construction (Fernandez & Steel, 1998, Sec. 2-3): let f0 be the
    symmetric, unit-variance base density (here Student t with `dof`
    degrees of freedom). The skewed density is

        g(x; gamma) = 2/(gamma + 1/gamma) * f0(gamma x)      for x < 0
                    = 2/(gamma + 1/gamma) * f0(x / gamma)     for x >= 0

    with raw moments E[Y^r] = M_r (gamma^{r+1} + (-1)^r gamma^{-(r+1)}) /
    (gamma + 1/gamma), where M_r = E[|X|^r] under f0. The distribution is
    then re-centred and re-scaled to have mean 0 and variance 1, matching
    the paper's calibration of all four distributions.
    """

    dof: float
    gamma: float
    name: str = field(default="skt3", compare=False)

    def __post_init__(self) -> None:
        assert self.gamma > 0, "Skewness parameter must be positive"

    @property
    def _base(self) -> StandardizedT:
        return StandardizedT(self.dof)

    @property
    def _p_negative(self) -> float:
        """P(X < 0) under the (un-standardized) skewed density, = 1/(1+gamma^2)."""
        return 1.0 / (1.0 + self.gamma ** 2)

    @property
    def _raw_mean_and_std(self) -> tuple[float, float]:
        m1 = _t_abs_moment(self.dof)  # M_1 = E|X| under the unit-variance base density
        m2 = 1.0  # M_2 = E[X^2] under the unit-variance base density (already standardized)
        g = self.gamma
        mean = m1 * (g - 1.0 / g)
        second_moment = m2 * (g ** 3 + 1.0 / g ** 3) / (g + 1.0 / g)
        variance = second_moment - mean ** 2
        assert variance > 0
        return mean, np.sqrt(variance)

    def _raw_ppf(self, u: np.ndarray) -> np.ndarray:
        """Quantile function of the *un-standardized* skewed density at
        probability level u, by inverting the piecewise cdf (Fernandez &
        Steel, 1998)."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        p = self._p_negative
        gamma = self.gamma
        base = self._base
        x = np.empty_like(u)
        neg = u < p
        # Negative branch: F0(gamma * x) = u * (gamma^2 + 1) / 2
        x[neg] = base.ppf(u[neg] * (gamma ** 2 + 1.0) / 2.0) / gamma
        # Positive branch: F0(x / gamma) = 0.5 + (u - p) / (2 (1 - p))
        pos = ~neg
        x[pos] = gamma * base.ppf(0.5 + (u[pos] - p) / (2.0 * (1.0 - p)))
        return x

    def _raw_cdf(self, x: np.ndarray) -> np.ndarray:
        """cdf of the *un-standardized* skewed density."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        p = self._p_negative
        gamma = self.gamma
        base = self._base
        out = np.empty_like(x)
        neg = x < 0
        out[neg] = (2.0 / (gamma + 1.0 / gamma)) * (1.0 / gamma) * base.cdf(gamma * x[neg])
        pos = ~neg
        out[pos] = p + (2.0 / (gamma + 1.0 / gamma)) * gamma * (base.cdf(x[pos] / gamma) - 0.5)
        return out

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Inverse-cdf simulation of the (un-standardized) skewed density,
        followed by the location-scale standardisation above."""
        u = rng.uniform(size=size)
        x = self._raw_ppf(u)
        mean, std = self._raw_mean_and_std
        return (x - mean) / std

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Quantile function of the standardized skewed t: invert the raw
        cdf at the raw value corresponding to standardized level q."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        x_raw = self._raw_ppf(q)
        mean, std = self._raw_mean_and_std
        return (x_raw - mean) / std

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        mean, std = self._raw_mean_and_std
        x_raw = x * std + mean
        return self._raw_cdf(x_raw)


DISTRIBUTIONS: dict[str, dict] = {
    "normal": {"family": "normal"},
    "t5": {"family": "t", "dof": 5},
    "t3": {"family": "t", "dof": 3},
    "skt3": {"family": "skewed_t", "dof": 3, "gamma": 1.2},
}


def get_distribution(name: str):
    """Return an object exposing .rvs(size, rng), .ppf(q) and .cdf(x) for a
    standardized (mean 0, variance 1) distribution named in DISTRIBUTIONS."""
    spec = DISTRIBUTIONS[name]
    if spec["family"] == "normal":
        return _StandardNormal()
    if spec["family"] == "t":
        return StandardizedT(spec["dof"])
    if spec["family"] == "skewed_t":
        return SkewedStandardizedT(spec["dof"], spec["gamma"])
    raise ValueError(f"Unknown distribution family: {spec['family']}")


class _StandardNormal:
    name: str = "normal"

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(size)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return stats.norm.ppf(q)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.cdf(x)


Distribution = Union[_StandardNormal, StandardizedT, SkewedStandardizedT]


def value_at_risk(dist, alpha: float) -> float:
    """VaR_alpha = F^{-1}(alpha) for a standardized loss distribution."""
    return float(np.atleast_1d(dist.ppf(np.array([alpha])))[0])


def expected_shortfall(dist, alpha: float, n_points: int = 20_000) -> float:
    """ES_alpha = (1/(1-alpha)) * integral_alpha^1 VaR_u du, via a fine
    quadrature on the quantile scale (Section 1, Eq. 1.1 representation)."""
    u = np.linspace(alpha, 1.0, n_points, endpoint=False) + (1 - alpha) / (2 * n_points)
    return float(np.mean(dist.ppf(u)))

"""Abstract base class for all ProbEngine distributions.

Every distribution — discrete or continuous — inherits from
:class:`Distribution` and implements a uniform interface so the
rest of the engine never needs to know *which* distribution it
is working with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class Distribution(ABC):
    """Base class for probability distributions.

    Subclasses must implement the abstract methods below.
    Discrete distributions implement :meth:`pmf`; continuous ones
    implement :meth:`pdf`.  The default implementations raise
    ``NotImplementedError`` so that calling ``pmf`` on a continuous
    distribution gives a clear error.
    """

    # ── Identity ────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable distribution name (e.g. ``'Binomial'``)."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Distribution parameters as a dict (e.g. ``{'n': 100, 'p': 0.03}``)."""

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """``True`` for discrete distributions, ``False`` for continuous."""

    # ── Core interface ──────────────────────────────────

    def pmf(self, k: ArrayLike) -> np.floating | np.ndarray:
        """Probability mass function (discrete distributions only)."""
        raise NotImplementedError(f"{self.name} is continuous — use pdf() instead.")

    def pdf(self, x: ArrayLike) -> np.floating | np.ndarray:
        """Probability density function (continuous distributions only)."""
        raise NotImplementedError(f"{self.name} is discrete — use pmf() instead.")

    @abstractmethod
    def cdf(self, x: ArrayLike) -> np.floating | np.ndarray:
        """Cumulative distribution function: P(X ≤ x)."""

    @abstractmethod
    def sf(self, x: ArrayLike) -> np.floating | np.ndarray:
        """Survival function: P(X > x) = 1 − cdf(x)."""

    @abstractmethod
    def mean(self) -> float:
        """Expected value E[X]."""

    @abstractmethod
    def variance(self) -> float:
        """Variance Var(X)."""

    @abstractmethod
    def sample(self, n: int = 1, seed: int | None = None) -> np.ndarray:
        """Draw *n* independent samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : int or None
            Random seed for reproducibility.
        """

    # ── Convenience ─────────────────────────────────────

    def std(self) -> float:
        """Standard deviation (derived from variance)."""
        return float(np.sqrt(self.variance()))

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"

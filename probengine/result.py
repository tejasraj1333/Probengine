"""Result object returned by every ProbEngine query.

Every answer is auditable: it reports the computed value, the method used,
the reason that method was chosen, the assumptions made, and validation checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    """Immutable result returned by ``Model.query()``.

    Attributes
    ----------
    value : float
        The computed probability or expected value.
    method : str
        Human-readable label for the computation method used.
        Example: ``"exact: binomial pmf"``
    reason : str
        Why this method was chosen.
        Example: ``"single supported distribution, no dependencies"``
    assumptions : list[str]
        Assumptions that the computation relies on.
        Example: ``["fixed n=100", "independent trials", "constant p=0.03"]``
    checks : dict[str, str]
        Validation results.  Keys are check names, values are
        ``"pass"``, ``"fail"``, or ``"skipped"``.
    confidence : str
        ``"exact"`` for closed-form results, or an interval string
        like ``"approx ± 0.002"`` for sampled results.
    diagnostics : dict[str, Any]
        Extra solver-specific diagnostics (e.g. sample size, R-hat).
    """

    value: float
    method: str = "unknown"
    reason: str = ""
    assumptions: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)
    confidence: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ── Display ─────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable text report of this result."""
        lines = [
            f"Result: {self.value}",
            f"  Method:     {self.method}",
            f"  Reason:     {self.reason}",
            f"  Confidence: {self.confidence}",
        ]
        if self.assumptions:
            lines.append("  Assumptions:")
            for a in self.assumptions:
                lines.append(f"    - {a}")
        if self.checks:
            lines.append("  Checks:")
            for name, status in self.checks.items():
                lines.append(f"    {name}: {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Result(value={self.value}, method='{self.method}', confidence='{self.confidence}')"

    # ── Visualization (stub — implemented in Stage 5) ───

    def plot(self) -> None:
        """Plot the distribution with the queried event highlighted.

        .. note:: Fully implemented in Stage 5.
        """
        raise NotImplementedError("Plotting will be implemented in Stage 5.")

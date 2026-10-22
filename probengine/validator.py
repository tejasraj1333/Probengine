"""Validation layer — cross-checks every result before returning it.

.. note:: Stub for Stage 0. Validation checks implemented in Stage 3.
"""

from __future__ import annotations

from probengine.result import Result


def validate(result: Result) -> Result:
    """Run validation checks on a result and update ``result.checks``.

    Checks planned (Stage 3):
    1. Range check — probability in [0, 1].
    2. Complement check — P(A) + P(¬A) = 1.
    3. Distribution check — all outcomes sum to 1 (discrete).
    4. Cross-method check — exact vs. Monte Carlo agreement.
    5. Assumption check — flag violated assumptions.

    .. note:: Stub — marks all checks as skipped.
    """
    result.checks = {
        "range": "skipped",
        "complement": "skipped",
        "cross_method": "skipped",
    }
    return result

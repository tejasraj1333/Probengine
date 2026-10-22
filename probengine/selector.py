"""Method Selector — chooses the best solver for a given model + query.

.. note:: Stub for Stage 0. Rule-based selector implemented in Stage 3.
"""

from __future__ import annotations


def select_method(analysis: dict) -> tuple[str, str]:
    """Select a computation method based on model analysis.

    Parameters
    ----------
    analysis : dict
        Feature dict returned by :func:`probengine.analyzer.analyze`.

    Returns
    -------
    method : str
        Name of the method to use (e.g. ``"closed_form"``, ``"monte_carlo"``).
    reason : str
        Human-readable explanation of why this method was chosen.

    .. note:: Stub — always returns ``"closed_form"``. Full rules in Stage 3.
    """
    return "closed_form", "Stage 0 stub — default method"

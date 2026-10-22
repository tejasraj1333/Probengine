"""Model Analyzer — inspects a model's structure to guide method selection.

.. note:: Stub for Stage 0. Implemented in Stage 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probengine.model import Model


def analyze(model: Model) -> dict:
    """Analyze a model and return a feature dict describing its structure.

    The returned dict will contain keys such as:
    - ``n_variables``: number of variables
    - ``has_evidence``: whether evidence has been observed
    - ``all_discrete``: whether every variable is discrete
    - ``has_dependencies``: whether any variable depends on another

    .. note:: Stub — returns minimal info. Full implementation in Stage 3.
    """
    return {
        "n_variables": len(model.variables),
        "has_evidence": len(model.evidence) > 0,
    }

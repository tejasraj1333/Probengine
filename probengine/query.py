"""Query expressions for ProbEngine.

When a user writes ``defect == 4`` or ``defect >= 3``, Python's operator
overloading creates a :class:`QueryExpr` that describes *what* the user
wants to know, without computing it yet.  The Model later passes this
expression to a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probengine.model import Variable


@dataclass(frozen=True)
class QueryExpr:
    """A description of a probability question.

    Attributes
    ----------
    variable : Variable
        The random variable being queried.
    operator : str
        One of ``"=="``, ``"!="``, ``"<"``, ``"<="``, ``">"``, ``">="``.
    value : float | int
        The threshold / target value.
    """

    variable: Variable
    operator: str
    value: float | int

    def __repr__(self) -> str:
        return f"Query({self.variable.name} {self.operator} {self.value})"

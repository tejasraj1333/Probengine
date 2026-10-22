"""ProbEngine Model — the main entry point for users.

A :class:`Model` holds random variables, their distributions,
dependencies, and observed evidence.  Users describe the problem
here and ask questions via :meth:`query`.

.. note:: This is a **Stage 0 stub**. The query pipeline is wired
   up in Stages 2–4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from probengine.distributions.base import Distribution
from probengine.query import QueryExpr
from probengine.result import Result


@dataclass
class Variable:
    """A named random variable inside a model.

    Comparison operators (``==``, ``>=``, etc.) return :class:`QueryExpr`
    objects so the user can write natural expressions like ``defect == 4``.
    """

    name: str
    distribution: Distribution
    depends_on: Variable | None = None
    cpt: dict[Any, Any] | None = None

    # ── Operator overloads → QueryExpr ──────────────────

    def __eq__(self, other: object) -> QueryExpr:  # type: ignore[override]
        return QueryExpr(variable=self, operator="==", value=other)  # type: ignore[arg-type]

    def __ne__(self, other: object) -> QueryExpr:  # type: ignore[override]
        return QueryExpr(variable=self, operator="!=", value=other)  # type: ignore[arg-type]

    def __lt__(self, other: float | int) -> QueryExpr:
        return QueryExpr(variable=self, operator="<", value=other)

    def __le__(self, other: float | int) -> QueryExpr:
        return QueryExpr(variable=self, operator="<=", value=other)

    def __gt__(self, other: float | int) -> QueryExpr:
        return QueryExpr(variable=self, operator=">", value=other)

    def __ge__(self, other: float | int) -> QueryExpr:
        return QueryExpr(variable=self, operator=">=", value=other)

    def __repr__(self) -> str:
        return f"Variable('{self.name}', {self.distribution})"

    def __hash__(self) -> int:
        return hash(self.name)


class Model:
    """Probabilistic model — the user-facing entry point.

    Example (stub — returns placeholder values in Stage 0)::

        from probengine import Model, Binomial

        m = Model()
        defect = m.var("defects", Binomial(n=100, p=0.03))
        result = m.query(defect == 4)
        print(result.value)       # actual computation in Stage 2
    """

    def __init__(self) -> None:
        self._variables: dict[str, Variable] = {}
        self._evidence: list[QueryExpr] = []

    # ── Building the model ──────────────────────────────

    def var(
        self,
        name: str,
        distribution: Distribution,
        depends_on: Variable | None = None,
        cpt: dict[Any, Any] | None = None,
    ) -> Variable:
        """Register a random variable in this model.

        Parameters
        ----------
        name : str
            Unique name for the variable.
        distribution : Distribution
            The probability distribution of this variable.
        depends_on : Variable, optional
            Parent variable (for conditional models — Stage 4).
        cpt : dict, optional
            Conditional probability table (Stage 4).

        Returns
        -------
        Variable
        """
        if name in self._variables:
            raise ValueError(f"Variable '{name}' already exists in this model.")
        v = Variable(name=name, distribution=distribution, depends_on=depends_on, cpt=cpt)
        self._variables[name] = v
        return v

    def observe(self, evidence: QueryExpr) -> None:
        """Record an observation (evidence) in the model.

        Parameters
        ----------
        evidence : QueryExpr
            An expression like ``rain == 1``.

        .. note:: Evidence handling is fully implemented in Stage 4.
        """
        self._evidence.append(evidence)

    # ── Querying ────────────────────────────────────────

    def query(self, expr: QueryExpr) -> Result:
        """Ask a probability question about this model.

        Parameters
        ----------
        expr : QueryExpr
            An expression like ``defect == 4`` or ``defect >= 3``.

        Returns
        -------
        Result

        .. note:: This is a **stub**. The solver pipeline is wired up
           in Stage 2.
        """
        return Result(
            value=0.0,
            method="stub",
            reason="Stage 0 placeholder — solver not yet implemented",
            assumptions=[],
            checks={},
            confidence="stub",
        )

    # ── Introspection ───────────────────────────────────

    @property
    def variables(self) -> dict[str, Variable]:
        """All registered variables."""
        return dict(self._variables)

    @property
    def evidence(self) -> list[QueryExpr]:
        """All observed evidence."""
        return list(self._evidence)

    def __repr__(self) -> str:
        n_vars = len(self._variables)
        n_ev = len(self._evidence)
        return f"Model({n_vars} variable(s), {n_ev} evidence)"

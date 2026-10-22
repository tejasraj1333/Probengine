"""ProbEngine — A probability engine library.

Describe the problem, the engine picks the method and solves it.

Quick start::

    from probengine import Model

    m = Model()
    # defect = m.var("defects", Binomial(n=100, p=0.03))  # Stage 1
    # print(m.query(defect == 4).value)                    # Stage 2
"""

from probengine.model import Model, Variable
from probengine.query import QueryExpr
from probengine.result import Result

__all__ = [
    "Model",
    "Variable",
    "QueryExpr",
    "Result",
]

__version__ = "0.1.0"

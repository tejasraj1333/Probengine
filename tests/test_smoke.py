"""Stage 0 smoke tests — verify the foundation is wired up correctly."""

import pytest

from probengine import Model, QueryExpr, Result, Variable

# ── Result ──────────────────────────────────────────────


class TestResult:
    """Tests for the Result dataclass."""

    def test_result_creation(self):
        r = Result(value=0.5, method="test", reason="unit test")
        assert r.value == 0.5
        assert r.method == "test"
        assert r.reason == "unit test"

    def test_result_defaults(self):
        r = Result(value=0.0)
        assert r.method == "unknown"
        assert r.assumptions == []
        assert r.checks == {}
        assert r.confidence == "unknown"

    def test_result_summary(self):
        r = Result(
            value=0.1706,
            method="exact: binomial pmf",
            reason="single supported distribution",
            assumptions=["fixed n=100", "independent trials"],
            checks={"range": "pass"},
            confidence="exact",
        )
        text = r.summary()
        assert "0.1706" in text
        assert "exact: binomial pmf" in text
        assert "independent trials" in text
        assert "range: pass" in text

    def test_result_repr(self):
        r = Result(value=0.5, method="test", confidence="exact")
        assert "0.5" in repr(r)
        assert "test" in repr(r)

    def test_plot_not_implemented(self):
        r = Result(value=0.5)
        with pytest.raises(NotImplementedError):
            r.plot()


# ── QueryExpr ───────────────────────────────────────────


class TestQueryExpr:
    """Tests for query expression creation via operator overloading."""

    def setup_method(self):
        """Create a dummy variable for testing."""
        # We need a Distribution stub for Variable
        from probengine.distributions.base import Distribution

        class DummyDist(Distribution):
            @property
            def name(self) -> str:
                return "Dummy"

            @property
            def params(self) -> dict:
                return {}

            @property
            def is_discrete(self) -> bool:
                return True

            def pmf(self, k):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 1.0

            def sample(self, n=1, seed=None):
                import numpy as np
                return np.zeros(n)

        self.dist = DummyDist()

    def test_eq_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v == 5
        assert isinstance(q, QueryExpr)
        assert q.operator == "=="
        assert q.value == 5

    def test_ge_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v >= 3
        assert isinstance(q, QueryExpr)
        assert q.operator == ">="
        assert q.value == 3

    def test_lt_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v < 10
        assert q.operator == "<"
        assert q.value == 10

    def test_gt_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v > 0
        assert q.operator == ">"

    def test_le_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v <= 7
        assert q.operator == "<="

    def test_ne_creates_query(self):
        m = Model()
        v = m.var("x", self.dist)
        q = v != 2
        assert q.operator == "!="


# ── Model ───────────────────────────────────────────────


class TestModel:
    """Tests for Model creation and variable registration."""

    def setup_method(self):
        from probengine.distributions.base import Distribution

        class DummyDist(Distribution):
            @property
            def name(self) -> str:
                return "Dummy"

            @property
            def params(self) -> dict:
                return {}

            @property
            def is_discrete(self) -> bool:
                return True

            def pmf(self, k):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 1.0

            def sample(self, n=1, seed=None):
                import numpy as np
                return np.zeros(n)

        self.dist = DummyDist()

    def test_create_model(self):
        m = Model()
        assert len(m.variables) == 0
        assert len(m.evidence) == 0

    def test_add_variable(self):
        m = Model()
        v = m.var("x", self.dist)
        assert isinstance(v, Variable)
        assert v.name == "x"
        assert "x" in m.variables

    def test_duplicate_variable_raises(self):
        m = Model()
        m.var("x", self.dist)
        with pytest.raises(ValueError, match="already exists"):
            m.var("x", self.dist)

    def test_observe_evidence(self):
        m = Model()
        v = m.var("x", self.dist)
        m.observe(v == 1)
        assert len(m.evidence) == 1

    def test_query_returns_result(self):
        m = Model()
        v = m.var("x", self.dist)
        result = m.query(v == 5)
        assert isinstance(result, Result)
        assert result.method == "stub"  # Stage 0 placeholder

    def test_model_repr(self):
        m = Model()
        m.var("a", self.dist)
        m.var("b", self.dist)
        assert "2 variable(s)" in repr(m)


# ── Distribution base ──────────────────────────────────


class TestDistributionBase:
    """Tests for the abstract Distribution base class."""

    def test_cannot_instantiate_abstract(self):
        from probengine.distributions.base import Distribution

        with pytest.raises(TypeError):
            Distribution()  # type: ignore[abstract]

    def test_continuous_pmf_raises(self):
        """Calling pmf on a continuous distribution should raise."""
        from probengine.distributions.base import Distribution

        class ContinuousDummy(Distribution):
            @property
            def name(self):
                return "ContinuousDummy"

            @property
            def params(self):
                return {}

            @property
            def is_discrete(self):
                return False

            def pdf(self, x):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 1.0

            def sample(self, n=1, seed=None):
                import numpy as np
                return np.zeros(n)

        d = ContinuousDummy()
        with pytest.raises(NotImplementedError, match="continuous"):
            d.pmf(0)

    def test_discrete_pdf_raises(self):
        """Calling pdf on a discrete distribution should raise."""
        from probengine.distributions.base import Distribution

        class DiscreteDummy(Distribution):
            @property
            def name(self):
                return "DiscreteDummy"

            @property
            def params(self):
                return {}

            @property
            def is_discrete(self):
                return True

            def pmf(self, k):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 1.0

            def sample(self, n=1, seed=None):
                import numpy as np
                return np.zeros(n)

        d = DiscreteDummy()
        with pytest.raises(NotImplementedError, match="discrete"):
            d.pdf(0)

    def test_repr(self):
        from probengine.distributions.base import Distribution

        class Named(Distribution):
            @property
            def name(self):
                return "TestDist"

            @property
            def params(self):
                return {"a": 1, "b": 2}

            @property
            def is_discrete(self):
                return True

            def pmf(self, k):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 1.0

            def sample(self, n=1, seed=None):
                import numpy as np
                return np.zeros(n)

        d = Named()
        assert repr(d) == "TestDist(a=1, b=2)"

    def test_std_derived_from_variance(self):
        import numpy as np

        from probengine.distributions.base import Distribution

        class WithVariance(Distribution):
            @property
            def name(self):
                return "V"

            @property
            def params(self):
                return {}

            @property
            def is_discrete(self):
                return True

            def pmf(self, k):
                return 0.0

            def cdf(self, x):
                return 0.0

            def sf(self, x):
                return 1.0

            def mean(self):
                return 0.0

            def variance(self):
                return 4.0

            def sample(self, n=1, seed=None):
                return np.zeros(n)

        d = WithVariance()
        assert d.std() == pytest.approx(2.0)

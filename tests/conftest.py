"""Shared pytest fixtures and mock classes for chess-tuning-tools tests.

This module provides common fixtures, mock classes, and test utilities
to reduce duplication across test files and improve test organization.
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from skopt.space import Categorical, Integer, Real, Space


class MockDimension:
    """Mock skopt.space.Dimension for testing."""

    def __init__(
        self,
        categories=None,
        bounds=(0.0, 1.0),
        prior="uniform",
        transformed_size=1,
        is_constant=False,
        name=None,
    ):
        self.categories = np.array(categories) if categories else np.array([])
        self.bounds = bounds
        self.prior = prior
        self.transformed_size = transformed_size
        self.is_constant = is_constant
        self.name = name
        self.low = bounds[0]
        self.high = bounds[1]

    def transform(self, x):
        if len(self.categories) > 0:
            if isinstance(x, np.ndarray):
                indices = np.array(
                    [np.where(self.categories == val)[0][0] for val in x]
                )
                return np.eye(len(self.categories))[indices]
            else:
                idx = np.where(self.categories == x)[0][0]
                return np.eye(len(self.categories))[idx]
        return np.asarray(x) if not isinstance(x, np.ndarray) else x

    def inverse_transform(self, x):
        return x

    def __repr__(self):
        return f"MockDimension(bounds={self.bounds}, prior={self.prior})"


class MockSpace:
    """Mock skopt Space for testing."""

    def __init__(self, n_dims=2, dimensions=None):
        self.n_dims = n_dims
        self.dimensions = dimensions or [MockDimension() for _ in range(n_dims)]
        self.transformed_dims = sum(
            dim.transformed_size for dim in self.dimensions
        )

    def rvs(self, n_samples, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.uniform(0, 1, size=(n_samples, self.n_dims))

    def transform(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr

    def inverse_transform(self, X):
        return X

    @property
    def bounds(self):
        return [(dim.low, dim.high) for dim in self.dimensions]

    def __repr__(self):
        return f"MockSpace(n_dims={self.n_dims})"


class MockGPModel:
    """Mock Gaussian Process model for testing."""

    def __init__(self, noise_std=0.1, n_features=1):
        self.noise_std = noise_std
        self.n_features_in_ = n_features

    def predict(self, X, return_std=False):
        X_arr = np.asarray(X)
        y = (
            -np.sin(X_arr[:, 0] * np.pi * 2)
            if X_arr.ndim > 1
            else -np.sin(X_arr * np.pi * 2)
        )
        if return_std:
            return y, np.ones(len(y)) * self.noise_std
        return y

    def noise_set_to_zero(self):
        class ContextManager:
            def __init__(self, model):
                self.model = model

            def __enter__(self):
                return self.model

            def __exit__(self, *args):
                pass

        return ContextManager(self)

    def log_marginal_likelihood(self, theta):
        return 0.0

    @property
    def chain_(self):
        return MagicMock()


class MockOptimizer:
    """Mock skopt Optimizer for testing."""

    def __init__(self, n_dims=1):
        self.space = MockSpace(n_dims=n_dims)
        self.models = [MockGPModel(n_features=n_dims)]
        self.xi = 0.1
        self.n_initial = 0

    def ask(self):
        return [0.5] * self.space.n_dims

    def tell(self, x, y):
        pass


class MockRegression:
    """Mock regression model for testing."""

    def predict(self, X):
        return np.zeros(len(X))


class MockPolynomialFeatures:
    """Mock polynomial features transformer for testing."""

    def transform(self, X):
        return X


class MockOptimizeResult:
    """Mock scipy.optimize.OptimizeResult for testing."""

    def __init__(self, n_dims=2, n_iters=10):
        self.models = [MockGPModel(n_features=n_dims)]
        self.space = MockSpace(n_dims=n_dims)
        self.x_iters = np.random.uniform(0, 1, size=(n_iters, n_dims))
        self.func_vals = np.random.uniform(-1, 0, size=n_iters)
        self.x = [0.5] * n_dims


class MockActiveSubspaces:
    """Mock active subspaces object for testing."""

    def __init__(self, n_evals=5, n_pars=5, n_evects=2, dim=1):
        all_evals = [10.0, 5.0, 2.0, 0.5, 0.1]
        self.evals = np.array(all_evals[:n_evals])
        all_evals_br = [
            [9.0, 11.0],
            [4.0, 6.0],
            [1.5, 2.5],
            [0.3, 0.7],
            [0.05, 0.15],
        ]
        self.evals_br = np.array(all_evals_br[:n_evals])
        self.evects = np.random.randn(n_pars, n_evects)
        self.dim = dim

    def transform(self, X):
        return np.dot(X, self.evects), None


@pytest.fixture
def mock_dimension():
    """Provide a basic mock dimension."""
    return MockDimension(bounds=(0.0, 1.0))


@pytest.fixture
def mock_space_1d():
    """Provide a 1D mock space."""
    return MockSpace(n_dims=1)


@pytest.fixture
def mock_space_2d():
    """Provide a 2D mock space."""
    return MockSpace(n_dims=2)


@pytest.fixture
def mock_gp_model():
    """Provide a mock GP model."""
    return MockGPModel(n_features=1)


@pytest.fixture
def mock_optimizer():
    """Provide a mock optimizer."""
    return MockOptimizer(n_dims=1)


@pytest.fixture
def mock_optimizer_2d():
    """Provide a 2D mock optimizer."""
    return MockOptimizer(n_dims=2)


@pytest.fixture(autouse=True)
def cleanup_polyglot_config():
    """Clean up polyglot config directory after each test."""
    yield
    polyglot_dir = Path("polyglot-config")
    if polyglot_dir.exists():
        shutil.rmtree(polyglot_dir)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file operations."""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    original_cwd = Path.cwd()
    try:
        os.chdir(tmpdir)
        yield Path(tmpdir)
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmpdir)


@pytest.fixture
def sample_real_space():
    """Provide a sample Real space."""
    return Space([Real(0, 10), Real(0, 10)])


@pytest.fixture
def sample_integer_space():
    """Provide a sample Integer space."""
    return Space([Integer(0, 10), Integer(0, 10)])


@pytest.fixture
def sample_categorical_space():
    """Provide a sample Categorical space."""
    return Space([Categorical(["a", "b", "c"]), Real(0, 10)])


@pytest.fixture
def sample_mixed_space():
    """Provide a sample mixed space."""
    return Space([Integer(0, 10), Categorical(["a", "b", "c"]), Real(0, 1)])


@pytest.fixture
def mock_datetime_module(monkeypatch):
    """Mock datetime module with actual datetime classes."""
    from datetime import datetime, timezone

    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def utcnow(cls):
            return cls(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def utcfromtimestamp(cls, timestamp):
            return cls(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class MockDate:
        @classmethod
        def today(cls):
            return MockDatetime(2024, 1, 1)

    import datetime as dt_module

    monkeypatch.setattr(dt_module, "datetime", MockDatetime)
    monkeypatch.setattr(dt_module, "date", MockDate)
    return MockDatetime


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for plotting tests."""
    import matplotlib.pyplot as plt
    from unittest.mock import MagicMock, patch

    fig = MagicMock()
    ax = MagicMock()
    fig.__enter__ = MagicMock(return_value=fig)
    fig.__exit__ = MagicMock(return_value=None)
    ax.__enter__ = MagicMock(return_value=ax)
    ax.__exit__ = MagicMock(return_value=None)

    with patch.object(plt, "subplots", return_value=(fig, ax)):
        with patch.object(plt, "Figure", return_value=fig):
            with patch.object(plt, "gcf", return_value=fig):
                with patch.object(plt, "gca", return_value=ax):
                    yield {"fig": fig, "ax": ax, "plt": plt}


@pytest.fixture
def mock_corner():
    """Mock corner library for plotting tests."""
    from unittest.mock import MagicMock, patch

    mock_corner_lib = MagicMock()

    with patch.dict("sys.modules", {"corner": mock_corner_lib}):
        yield mock_corner_lib


@pytest.fixture
def sample_data_1d():
    """Provide sample 1D data for plotting tests."""
    return {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "y": np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.0]),
        "y_std": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    }


@pytest.fixture
def sample_data_2d():
    """Provide sample 2D data for plotting tests."""
    return {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "y": np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.0]),
        "y_std": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        "x2": np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
    }


@pytest.fixture
def sample_iterations():
    """Provide sample iteration data."""
    return np.array([1.0, 2.0, 3.0, 3.0, 4.0])


@pytest.fixture
def sample_penta_counts():
    """Provide sample penta counts."""
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_ldw_arrays():
    """Provide sample LDW arrays."""
    return {
        "ldw1": np.array([0.1, 0.2, 0.7]),
        "ldw2": np.array([0.2, 0.2, 0.6]),
    }


@pytest.fixture
def sample_engines_json():
    """Provide sample engines JSON structure."""
    return [
        {
            "command": "lc0",
            "name": "engine1",
            "initStrings": ["uci"],
            "protocol": "uci",
        }
    ]


@pytest.fixture
def sample_polyglot_params():
    """Provide sample polyglot parameters."""
    return [
        {"engine_command": "engine1", "book_file": ""},
        {
            "engine_command": "engine2",
            "book_file": "book.bin",
            "max_book_depth": 10,
        },
    ]


@pytest.fixture
def sample_ranges():
    """Provide sample range specifications."""
    return {"Threads": "Integer(1, 8)", "CPuct": "Real(0.0, 2.0)"}


@pytest.fixture
def sample_init_strings():
    """Provide sample init strings."""
    return [
        "uci",
        "setoption name Threads value 4",
        "setoption name Hash value 128",
    ]


@pytest.fixture
def sample_timecontrols():
    """Provide sample time controls."""
    from tune.utils import TimeControl

    return [
        TimeControl.from_string("60"),
        TimeControl.from_string("10+0.1"),
    ]


@pytest.fixture
def sample_prior_params():
    """Provide sample prior parameters."""
    return {"n_parameters": 3, "signal_scale": 1.0, "noise_scale": 0.1}

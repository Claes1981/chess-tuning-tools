"""Comprehensive tests for tune/plots.py functions.

Tests all plotting functions with mocked dependencies to document current functionality.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, create_autospec
from scipy.optimize import OptimizeResult

from tune.plots import (
    _evenly_sample,
    partial_dependence,
    plot_objective_1d,
    plot_objective,
    plot_optima,
    plot_performance,
    plot_activesubspace_eigenvalues,
    plot_activesubspace_eigenvectors,
    plot_activesubspace_sufficient_summary,
)


# =============================================================================
# Mock Classes for skopt and matplotlib objects
# =============================================================================


class MockDimension:
    """Mock skopt.space.Dimension."""

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
            # Handle both array input and scalar input
            if isinstance(x, np.ndarray):
                # Convert category values to indices
                indices = np.array(
                    [np.where(self.categories == val)[0][0] for val in x]
                )
                return np.eye(len(self.categories))[indices]
            else:
                # Single value
                idx = np.where(self.categories == x)[0][0]
                return np.eye(len(self.categories))[idx]
        return x

    def inverse_transform(self, x):
        return x


class MockSpace:
    """Mock skopt.space.Space."""

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
        return np.array(X)

    def inverse_transform(self, X):
        return X

    @property
    def bounds(self):
        return [(dim.low, dim.high) for dim in self.dimensions]


class MockGPModel:
    """Mock Gaussian Process model."""

    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def predict(self, X, return_std=False):
        y = (
            -np.sin(X[:, 0] * np.pi * 2)
            if X.ndim > 1
            else -np.sin(X * np.pi * 2)
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

    @property
    def chain_(self):
        return MagicMock()


class MockRegression:
    """Mock regression model."""

    def predict(self, X):
        return np.zeros(len(X))


class MockPolynomialFeatures:
    """Mock polynomial features transformer."""

    def transform(self, X):
        return X


class MockOptimizeResult:
    """Mock scipy.optimize.OptimizeResult."""

    def __init__(self, n_dims=2, n_iters=10):
        self.models = [MockGPModel()]
        self.space = MockSpace(n_dims=n_dims)
        self.x_iters = np.random.uniform(0, 1, size=(n_iters, n_dims))
        self.func_vals = np.random.uniform(-1, 0, size=n_iters)
        self.x = [0.5] * n_dims


class MockActiveSubspaces:
    """Mock active subspaces object."""

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


# =============================================================================
# Tests for _evenly_sample
# =============================================================================


class TestEvenlySample:
    """Tests for the _evenly_sample helper function."""

    def test_continuous_uniform(self):
        """Test sampling from continuous uniform dimension."""
        dim = MockDimension(bounds=(0.0, 1.0), prior="uniform")
        xi, xi_transformed = _evenly_sample(dim, n_points=10)

        assert len(xi) == 10
        assert len(xi_transformed) == 10
        assert np.all(xi >= 0.0)
        assert np.all(xi <= 1.0)
        assert np.all(xi_transformed >= 0.0)
        assert np.all(xi_transformed <= 1.0)

    def test_continuous_log_uniform(self):
        """Test sampling from continuous log-uniform dimension."""
        dim = MockDimension(bounds=(0.01, 100.0), prior="log-uniform")
        xi, xi_transformed = _evenly_sample(dim, n_points=10)

        assert len(xi) == 10
        assert len(xi_transformed) == 10
        assert np.all(xi >= 0.01)
        assert np.all(xi <= 100.0)

    def test_categorical(self):
        """Test sampling from categorical dimension."""
        categories = ["a", "b", "c", "d"]
        dim = MockDimension(
            categories=categories, bounds=(0, 3), transformed_size=4
        )
        xi, xi_transformed = _evenly_sample(dim, n_points=4)

        assert len(xi) == 4
        assert xi_transformed.shape == (4, 4)  # One-hot encoded

    def test_categorical_fewer_points(self):
        """Test sampling fewer points than categories."""
        categories = ["a", "b", "c", "d", "e"]
        dim = MockDimension(
            categories=categories, bounds=(0, 4), transformed_size=5
        )
        xi, xi_transformed = _evenly_sample(dim, n_points=3)

        assert len(xi) == 3
        assert xi_transformed.shape == (3, 5)

    def test_categorical_more_points(self):
        """Test sampling more points than categories (returns min)."""
        categories = ["a", "b", "c"]
        dim = MockDimension(
            categories=categories, bounds=(0, 2), transformed_size=3
        )
        xi, xi_transformed = _evenly_sample(dim, n_points=10)

        assert len(xi) == 3  # Returns min(len(cats), n_points)
        assert xi_transformed.shape == (3, 3)

    def test_single_point(self):
        """Test sampling a single point."""
        dim = MockDimension(bounds=(0.0, 1.0))
        xi, xi_transformed = _evenly_sample(dim, n_points=1)

        assert len(xi) == 1
        assert len(xi_transformed) == 1


# =============================================================================
# Tests for partial_dependence
# =============================================================================


class TestPartialDependence:
    """Tests for the partial_dependence function."""

    def test_1d_basic(self):
        """Test basic 1D partial dependence."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            n_samples=50,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20
        assert xi.shape == (20,)
        # yi is returned as a list for 1D case
        assert isinstance(yi, list)

    def test_1d_with_confidence_interval(self):
        """Test 1D partial dependence with confidence interval."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, yi_std = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            plot_confidence_interval_width=True,
            n_samples=50,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20
        assert len(yi_std) == 20

    def test_2d_basic(self):
        """Test basic 2D partial dependence."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, zi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=1,
            n_samples=50,
            n_points=15,
        )

        assert len(xi) == 15
        assert len(yi) == 15
        assert zi.shape == (15, 15)

    def test_2d_with_confidence_interval(self):
        """Test 2D partial dependence with confidence interval."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, zi, zi_std = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=1,
            plot_confidence_interval_width=True,
            n_samples=50,
            n_points=15,
        )

        assert len(xi) == 15
        assert len(yi) == 15
        assert zi.shape == (15, 15)
        assert zi_std.shape == (15, 15)

    def test_with_polynomial_regression(self):
        """Test with polynomial regression enabled."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            plot_polynomial_regression=True,
            n_samples=50,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20

    def test_with_x_eval(self):
        """Test with fixed x_eval values."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        x_eval = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            x_eval=x_eval,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20

    def test_with_sample_points(self):
        """Test with pre-sampled points."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        sample_points = np.random.uniform(0, 1, size=(100, 3))

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            sample_points=sample_points,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20


# =============================================================================
# Tests for plot_objective_1d
# =============================================================================


class TestPlotObjective1D:
    """Tests for the plot_objective_1d function."""

    @pytest.fixture
    def mock_result(self):
        """Create mock OptimizeResult for 1D plotting."""
        result = MockOptimizeResult(n_dims=1, n_iters=20)
        return result

    def test_basic_plot(self, mock_result):
        """Test basic 1D objective plot."""
        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_line = MagicMock()
            mock_ax1.plot.return_value = [mock_line]
            mock_ax2.plot.return_value = [mock_line]
            mock_subplots.return_value = (
                mock_fig,
                [mock_ax1, mock_ax2],
            )

            fig, ax = plot_objective_1d(
                result=mock_result,
                parameter_name="TestParam",
                n_points=100,
            )

            assert fig is mock_fig
            assert ax is not None

    def test_without_parameter_name(self, mock_result):
        """Test plot without parameter name label."""
        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_line = MagicMock()
            mock_ax1.plot.return_value = [mock_line]
            mock_ax2.plot.return_value = [mock_line]
            mock_subplots.return_value = (
                mock_fig,
                [mock_ax1, mock_ax2],
            )

            fig, ax = plot_objective_1d(
                result=mock_result,
                parameter_name=None,
            )

            assert fig is mock_fig

    def test_with_existing_figure(self, mock_result):
        """Test plot with existing figure and axes."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_line = MagicMock()
        mock_ax1.plot.return_value = [mock_line]
        mock_ax1.fill_between.return_value = MagicMock()
        mock_ax1.scatter.return_value = MagicMock()
        mock_ax2.plot.return_value = [mock_line]

        fig, ax = plot_objective_1d(
            result=mock_result,
            fig=mock_fig,
            ax=[mock_ax1, mock_ax2],
        )

        assert fig is mock_fig

    def test_different_confidence_levels(self, mock_result):
        """Test with different confidence levels."""
        for confidence in [0.5, 0.9, 0.95, 0.99]:
            with patch("tune.plots.plt.subplots") as mock_subplots:
                mock_fig = MagicMock()
                mock_ax1 = MagicMock()
                mock_ax2 = MagicMock()
                mock_line = MagicMock()
                mock_ax1.plot.return_value = [mock_line]
                mock_ax2.plot.return_value = [mock_line]
                mock_subplots.return_value = (
                    mock_fig,
                    [mock_ax1, mock_ax2],
                )

                fig, ax = plot_objective_1d(
                    result=mock_result,
                    confidence=confidence,
                )

                assert fig is mock_fig

    def test_custom_figsize(self, mock_result):
        """Test with custom figure size."""
        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_line = MagicMock()
            mock_ax1.plot.return_value = [mock_line]
            mock_ax2.plot.return_value = [mock_line]
            mock_subplots.return_value = (
                mock_fig,
                [mock_ax1, mock_ax2],
            )

            fig, ax = plot_objective_1d(
                result=mock_result,
                figsize=(15, 8),
            )

            assert fig is mock_fig


# =============================================================================
# Tests for plot_optima
# =============================================================================


class TestPlotOptima:
    """Tests for the plot_optima function."""

    def test_basic_plot(self):
        """Test basic optima plot."""
        iterations = np.array([1, 2, 3, 4, 5])
        optima = np.array(
            [
                [0.1, 0.5],
                [0.2, 0.4],
                [0.3, 0.3],
                [0.35, 0.35],
                [0.4, 0.38],
            ]
        )

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_axes = np.empty(2, dtype=object)
            mock_axes[0] = mock_ax1
            mock_axes[1] = mock_ax2
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig, ax = plot_optima(
                iterations=iterations,
                optima=optima,
            )

            assert fig is mock_fig
            assert ax is not None

    def test_with_parameter_names(self):
        """Test plot with parameter names."""
        iterations = np.array([1, 2, 3])
        optima = np.array([[0.1], [0.2], [0.3]])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig, ax = plot_optima(
                iterations=iterations,
                optima=optima,
                parameter_names=["Param1"],
            )

            assert fig is mock_fig

    def test_with_space(self):
        """Test plot with space for scaling."""
        iterations = np.array([1, 2, 3])
        optima = np.array([[0.1], [0.2], [0.3]])
        space = MockSpace(n_dims=1)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig, ax = plot_optima(
                iterations=iterations,
                optima=optima,
                space=space,
            )

            assert fig is mock_fig

    def test_with_log_scale_space(self):
        """Test plot with log-scale parameter."""
        iterations = np.array([1, 2, 3])
        optima = np.array([[0.01], [0.1], [1.0]])
        dim = MockDimension(bounds=(0.01, 10.0), prior="log-uniform")
        space = MockSpace(n_dims=1, dimensions=[dim])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig, ax = plot_optima(
                iterations=iterations,
                optima=optima,
                space=space,
            )

            assert fig is mock_fig

    def test_duplicate_iterations(self):
        """Test with duplicate iteration numbers (should be handled)."""
        iterations = np.array([1, 1, 2, 2, 3])
        optima = np.array(
            [
                [0.1, 0.5],
                [0.15, 0.45],
                [0.2, 0.4],
                [0.25, 0.35],
                [0.3, 0.3],
            ]
        )

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_axes = np.empty(2, dtype=object)
            mock_axes[0] = mock_ax1
            mock_axes[1] = mock_ax2
            mock_subplots.return_value = (mock_fig, mock_axes)

            fig, ax = plot_optima(
                iterations=iterations,
                optima=optima,
            )

            assert fig is mock_fig

    def test_raises_mismatched_lengths(self):
        """Test that mismatched iterations/optima lengths raise ValueError."""
        iterations = np.array([1, 2, 3])
        optima = np.array([[0.1], [0.2]])  # Only 2 points

        with pytest.raises(ValueError):
            plot_optima(iterations=iterations, optima=optima)

    def test_raises_fig_without_ax(self):
        """Test that providing fig without ax raises ValueError."""
        mock_fig = MagicMock()

        with pytest.raises(ValueError):
            plot_optima(
                iterations=np.array([1, 2]),
                optima=np.array([[0.1], [0.2]]),
                fig=mock_fig,
            )


# =============================================================================
# Tests for plot_performance
# =============================================================================


class TestPlotPerformance:
    """Tests for the plot_performance function."""

    def test_basic_plot(self):
        """Test basic performance plot."""
        performance = np.array(
            [
                [1, 500, 50],
                [2, 520, 45],
                [3, 540, 40],
                [4, 530, 42],
                [5, 550, 38],
            ]
        )

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig, ax = plot_performance(performance=performance)

            assert fig is mock_fig
            assert ax is mock_ax

    def test_with_confidence(self):
        """Test plot with custom confidence level."""
        performance = np.array(
            [
                [1, 500, 50],
                [2, 520, 45],
                [3, 540, 40],
            ]
        )

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig, ax = plot_performance(
                performance=performance,
                confidence=0.95,
            )

            assert fig is mock_fig

    def test_with_existing_figure(self):
        """Test plot with existing figure and axes."""
        performance = np.array(
            [
                [1, 500, 50],
                [2, 520, 45],
            ]
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()

        fig, ax = plot_performance(
            performance=performance,
            fig=mock_fig,
            ax=mock_ax,
        )

        assert fig is mock_fig
        assert ax is mock_ax

    def test_duplicate_iterations(self):
        """Test with duplicate iteration numbers."""
        performance = np.array(
            [
                [1, 500, 50],
                [1, 510, 48],
                [2, 520, 45],
                [2, 525, 44],
            ]
        )

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig, ax = plot_performance(performance=performance)

            assert fig is mock_fig

    def test_raises_fig_without_ax(self):
        """Test that providing fig without ax raises ValueError."""
        mock_fig = MagicMock()

        with pytest.raises(ValueError):
            plot_performance(
                performance=np.array([[1, 500, 50]]),
                fig=mock_fig,
            )


# =============================================================================
# Tests for plot_activesubspace_eigenvalues
# =============================================================================


class TestPlotActiveSubspaceEigenvalues:
    """Tests for the plot_activesubspace_eigenvalues function."""

    def test_basic_plot(self):
        """Test basic eigenvalues plot."""
        active_subspaces = MockActiveSubspaces(n_evals=5)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_eigenvalues(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvalues_axes=mock_ax,
            )

            assert ax is mock_ax

    def test_with_n_evals(self):
        """Test plot with specific number of eigenvalues."""
        active_subspaces = MockActiveSubspaces(n_evals=10)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_eigenvalues(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvalues_axes=mock_ax,
                n_evals=3,
            )

            assert ax is mock_ax

    def test_with_bootstrap_bounds(self):
        """Test plot with bootstrap bounds."""
        active_subspaces = MockActiveSubspaces(n_evals=5)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_eigenvalues(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvalues_axes=mock_ax,
            )

            assert ax is mock_ax

    def test_with_zero_eigenvalue(self):
        """Test plot with zero eigenvalue (uses semilogy)."""
        active_subspaces = MagicMock()
        active_subspaces.evals = np.array([10.0, 5.0, 0.0, 0.0])
        active_subspaces.evals_br = None

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_eigenvalues(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvalues_axes=mock_ax,
            )

            assert ax is mock_ax

    def test_with_existing_figure(self):
        """Test plot with existing figure."""
        active_subspaces = MockActiveSubspaces(n_evals=5)

        mock_fig = MagicMock()
        mock_ax = MagicMock()

        ax = plot_activesubspace_eigenvalues(
            active_subspaces_object=active_subspaces,
            active_subspace_figure=mock_fig,
            active_subspace_eigenvalues_axes=mock_ax,
        )

        assert ax is mock_ax

    def test_raises_evals_none(self):
        """Test that None evals raises TypeError."""
        active_subspaces = MagicMock()
        active_subspaces.evals = None

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            with pytest.raises(TypeError):
                plot_activesubspace_eigenvalues(
                    active_subspaces_object=active_subspaces,
                    active_subspace_eigenvalues_axes=mock_ax,
                )

    def test_raises_n_evals_too_large(self):
        """Test that n_evals > len(evals) raises TypeError."""
        active_subspaces = MagicMock()
        active_subspaces.evals = np.array([10.0, 5.0])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            with pytest.raises(TypeError):
                plot_activesubspace_eigenvalues(
                    active_subspaces_object=active_subspaces,
                    active_subspace_eigenvalues_axes=mock_ax,
                    n_evals=5,
                )


# =============================================================================
# Tests for plot_activesubspace_eigenvectors
# =============================================================================


class TestPlotActiveSubspaceEigenvectors:
    """Tests for the plot_activesubspace_eigenvectors function."""

    def test_basic_plot(self):
        """Test basic eigenvectors plot."""
        active_subspaces = MockActiveSubspaces(n_pars=5, n_evects=2)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_axes = np.empty(2, dtype=object)
            mock_axes[0] = mock_ax1
            mock_axes[1] = mock_ax2
            mock_subplots.return_value = (mock_fig, mock_axes)

            ax = plot_activesubspace_eigenvectors(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvectors_axes=mock_axes,
            )

            assert ax is not None

    def test_with_labels(self):
        """Test plot with custom labels."""
        active_subspaces = MockActiveSubspaces(n_pars=3, n_evects=1)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            ax = plot_activesubspace_eigenvectors(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvectors_axes=mock_axes,
                labels=["Param1", "Param2", "Param3"],
            )

            assert ax is not None

    def test_with_n_evects(self):
        """Test plot with specific number of eigenvectors."""
        active_subspaces = MockActiveSubspaces(n_pars=5, n_evects=4)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_axes = np.empty(2, dtype=object)
            mock_axes[0] = mock_ax1
            mock_axes[1] = mock_ax2
            mock_subplots.return_value = (mock_fig, mock_axes)

            ax = plot_activesubspace_eigenvectors(
                active_subspaces_object=active_subspaces,
                active_subspace_eigenvectors_axes=mock_axes,
                n_evects=2,
            )

            assert ax is not None

    def test_raises_evects_none(self):
        """Test that None evects raises TypeError."""
        active_subspaces = MagicMock()
        active_subspaces.evects = None

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            with pytest.raises(TypeError):
                plot_activesubspace_eigenvectors(
                    active_subspaces_object=active_subspaces,
                    active_subspace_eigenvectors_axes=mock_axes,
                )

    def test_raises_n_evects_too_large(self):
        """Test that n_evects > evects.shape[0] raises ValueError."""
        active_subspaces = MagicMock()
        active_subspaces.evects = np.random.randn(5, 2)

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_axes = np.empty(1, dtype=object)
            mock_axes[0] = mock_ax
            mock_subplots.return_value = (mock_fig, mock_axes)

            with pytest.raises(ValueError):
                plot_activesubspace_eigenvectors(
                    active_subspaces_object=active_subspaces,
                    active_subspace_eigenvectors_axes=mock_axes,
                    n_evects=6,
                )


# =============================================================================
# Tests for plot_activesubspace_sufficient_summary
# =============================================================================


class TestPlotActiveSubspaceSufficientSummary:
    """Tests for the plot_activesubspace_sufficient_summary function."""

    def test_1d_summary(self):
        """Test 1D sufficient summary plot."""
        active_subspaces = MockActiveSubspaces(dim=1)
        inputs = np.random.uniform(-1, 1, size=(100, 5))
        outputs = np.random.randn(100)
        result = MockOptimizeResult(n_dims=5)
        next_point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_sufficient_summary(
                active_subspaces_object=active_subspaces,
                inputs=inputs,
                outputs=outputs,
                result_object=result,
                next_point=next_point,
                active_subspace_sufficient_summary_axes=mock_ax,
            )

            assert ax is mock_ax

    def test_2d_summary(self):
        """Test 2D sufficient summary plot."""
        active_subspaces = MockActiveSubspaces(dim=2)
        inputs = np.random.uniform(-1, 1, size=(100, 5))
        outputs = np.random.randn(100)
        result = MockOptimizeResult(n_dims=5)
        next_point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            ax = plot_activesubspace_sufficient_summary(
                active_subspaces_object=active_subspaces,
                inputs=inputs,
                outputs=outputs,
                result_object=result,
                next_point=next_point,
                active_subspace_figure=mock_fig,
                active_subspace_sufficient_summary_axes=mock_ax,
            )

            assert ax is mock_ax

    def test_raises_dim_greater_than_2(self):
        """Test that dim > 2 raises ValueError."""
        active_subspaces = MagicMock()
        active_subspaces.dim = 3
        active_subspaces.evects = np.random.randn(5, 3)

        inputs = np.random.uniform(-1, 1, size=(100, 5))
        outputs = np.random.randn(100)
        result = MockOptimizeResult(n_dims=5)
        next_point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            with pytest.raises(ValueError):
                plot_activesubspace_sufficient_summary(
                    active_subspaces_object=active_subspaces,
                    inputs=inputs,
                    outputs=outputs,
                    result_object=result,
                    next_point=next_point,
                    active_subspace_sufficient_summary_axes=mock_ax,
                )

    def test_raises_evects_none(self):
        """Test that None evects raises TypeError."""
        active_subspaces = MagicMock()
        active_subspaces.dim = 1
        active_subspaces.evects = None

        inputs = np.random.uniform(-1, 1, size=(100, 5))
        outputs = np.random.randn(100)
        result = MockOptimizeResult(n_dims=5)
        next_point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with patch("tune.plots.plt.subplots") as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            with pytest.raises(TypeError):
                plot_activesubspace_sufficient_summary(
                    active_subspaces_object=active_subspaces,
                    inputs=inputs,
                    outputs=outputs,
                    result_object=result,
                    next_point=next_point,
                    active_subspace_sufficient_summary_axes=mock_ax,
                )

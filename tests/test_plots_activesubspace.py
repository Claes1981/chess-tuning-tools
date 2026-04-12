"""Tests for active subspace plotting functions in tune/plots.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure tests directory is in path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import MockActiveSubspaces, MockOptimizeResult  # noqa: E402
from tune.plots import (  # noqa: E402
    plot_activesubspace_eigenvalues,
    plot_activesubspace_eigenvectors,
    plot_activesubspace_sufficient_summary,
)


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

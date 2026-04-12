"""Tests for plot_optima function in tune/plots.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure tests directory is in path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import MockDimension, MockSpace  # noqa: E402
from tune.plots import plot_optima  # noqa: E402


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

"""Tests for plot_objective_1d function in tune/plots.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure tests directory is in path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import MockOptimizeResult  # noqa: E402
from tune.plots import plot_objective_1d  # noqa: E402


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

"""Tests for plot_performance function in tune/plots.py."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure tests directory is in path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from tune.plots import plot_performance  # noqa: E402


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

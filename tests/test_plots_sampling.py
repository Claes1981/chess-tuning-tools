"""Tests for sampling functions in tune/plots.py.

Tests the _evenly_sample helper function.
"""

import os
import sys

import numpy as np

# Ensure tests directory is in path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from conftest import MockDimension  # noqa: E402
from tune.plots import _evenly_sample  # noqa: E402


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

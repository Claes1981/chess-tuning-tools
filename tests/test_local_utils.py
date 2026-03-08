"""Comprehensive tests for additional tune/local.py functions.

Tests inputs_uniform, reduce_ranges, load_points_to_evaluate, and other utility functions.
"""

import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from skopt.space import Categorical, Integer, Real, Space

from tune.local import (
    inputs_uniform,
    reduce_ranges,
    load_points_to_evaluate,
)


class TestInputsUniform:
    """Tests for the inputs_uniform function."""

    def test_basic_uniform_generation(self):
        """Test basic uniform sampling."""
        n_samples = 100
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert result.shape == (n_samples, 2)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_single_dimension(self):
        """Test with single dimension."""
        n_samples = 50
        lb = np.array([0.0])
        ub = np.array([10.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert result.shape == (n_samples, 1)
        assert np.all(result >= 0)
        assert np.all(result <= 10)

    def test_three_dimensions(self):
        """Test with three dimensions."""
        n_samples = 200
        lb = np.array([0.0, -1.0, 5.0])
        ub = np.array([1.0, 1.0, 10.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert result.shape == (n_samples, 3)
        assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 1)
        assert np.all(result[:, 1] >= -1) and np.all(result[:, 1] <= 1)
        assert np.all(result[:, 2] >= 5) and np.all(result[:, 2] <= 10)

    def test_different_bounds(self):
        """Test with different lower and upper bounds per dimension."""
        n_samples = 10
        lb = np.array([-10.0, -5.0])
        ub = np.array([10.0, 5.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert np.all(result[:, 0] >= -10) and np.all(result[:, 0] <= 10)
        assert np.all(result[:, 1] >= -5) and np.all(result[:, 1] <= 5)

    def test_large_sample_size(self):
        """Test with large number of samples."""
        n_samples = 10000
        lb = np.array([0.0])
        ub = np.array([1.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert result.shape == (n_samples, 1)
        # Check that values are reasonably distributed
        assert np.abs(np.mean(result) - 0.5) < 0.1

    def test_narrow_range(self):
        """Test with very narrow range."""
        n_samples = 100
        lb = np.array([0.999])
        ub = np.array([1.001])

        result = inputs_uniform(n_samples, lb, ub)

        assert np.all(result >= 0.999)
        assert np.all(result <= 1.001)
        assert np.allclose(result, 1.0, atol=0.001)

    def test_negative_bounds(self):
        """Test with all negative bounds."""
        n_samples = 50
        lb = np.array([-100.0, -50.0])
        ub = np.array([-10.0, -5.0])

        result = inputs_uniform(n_samples, lb, ub)

        assert np.all(result < 0)
        assert np.all(result[:, 0] >= -100) and np.all(result[:, 0] <= -10)
        assert np.all(result[:, 1] >= -50) and np.all(result[:, 1] <= -5)


class TestReduceRanges:
    """Tests for the reduce_ranges function."""

    def test_no_reduction_needed(self):
        """Test when all points are within new bounds."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [1.0, 2.0, 3.0]
        noise = [0.1, 0.2, 0.3]
        space = Space([Real(0, 10), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert not reduction_needed
        assert len(X_new) == 3
        assert len(y_new) == 3
        assert len(noise_new) == 3

    def test_reduction_needed(self):
        """Test when some points are outside new bounds."""
        X = [[1.0, 2.0], [15.0, 4.0], [5.0, 6.0]]  # Second point out of bounds
        y = [1.0, 2.0, 3.0]
        noise = [0.1, 0.2, 0.3]
        space = Space([Real(0, 10), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert reduction_needed
        assert len(X_new) == 2
        assert len(y_new) == 2
        assert len(noise_new) == 2

    def test_all_points_reduced(self):
        """Test when all points are outside new bounds."""
        X = [[15.0, 2.0], [20.0, 4.0], [25.0, 6.0]]
        y = [1.0, 2.0, 3.0]
        noise = [0.1, 0.2, 0.3]
        space = Space([Real(0, 10), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert reduction_needed
        assert len(X_new) == 0
        assert len(y_new) == 0
        assert len(noise_new) == 0

    def test_integer_dimension(self):
        """Test with Integer dimension."""
        X = [[1, 2], [5, 8], [15, 6]]  # Third point out of bounds
        y = [1.0, 2.0, 3.0]
        noise = [0.1, 0.2, 0.3]
        space = Space([Integer(0, 10), Integer(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert reduction_needed
        assert len(X_new) == 2

    def test_categorical_dimension(self):
        """Test with Categorical dimension."""
        X = [["a", 1], ["b", 2], ["c", 3], ["d", 4]]  # "d" is not in categories
        y = [1.0, 2.0, 3.0, 4.0]
        noise = [0.1, 0.2, 0.3, 0.4]
        space = Space([Categorical(["a", "b", "c"]), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert reduction_needed
        assert len(X_new) == 3
        assert X_new[0][0] != "d"
        assert X_new[1][0] != "d"
        assert X_new[2][0] != "d"

    def test_mixed_dimensions(self):
        """Test with mixed dimension types."""
        X = [
            [1, "a", 0.5],
            [5, "b", 0.8],
            [15, "c", 0.3],  # Integer out of bounds
            [3, "d", 0.6],  # Categorical out of bounds
        ]
        y = [1.0, 2.0, 3.0, 4.0]
        noise = [0.1, 0.2, 0.3, 0.4]
        space = Space(
            [Integer(0, 10), Categorical(["a", "b", "c"]), Real(0, 1)]
        )

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert reduction_needed
        assert len(X_new) == 2

    def test_boundary_values(self):
        """Test with values exactly at boundaries."""
        X = [[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]]
        y = [1.0, 2.0, 3.0]
        noise = [0.1, 0.2, 0.3]
        space = Space([Real(0, 10), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert not reduction_needed
        assert len(X_new) == 3

    def test_single_point(self):
        """Test with single data point."""
        X = [[5.0, 5.0]]
        y = [1.0]
        noise = [0.1]
        space = Space([Real(0, 10), Real(0, 10)])

        reduction_needed, X_new, y_new, noise_new = reduce_ranges(
            X, y, noise, space
        )

        assert not reduction_needed
        assert len(X_new) == 1


class TestLoadPointsToEvaluate:
    """Tests for the load_points_to_evaluate function."""

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        csv_file = tmp_path / "points.csv"
        return csv_file

    @pytest.fixture
    def test_space(self):
        """Create a test space."""
        return Space([Real(0, 10), Real(0, 10)])

    def test_none_csv_file(self, test_space):
        """Test with None csv_file."""
        result = load_points_to_evaluate(space=test_space, csv_file=None)
        assert result == []

    def test_valid_csv_file(self, temp_csv_file, test_space):
        """Test with valid CSV file."""
        # Create CSV file with 2D points
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        df.to_csv(temp_csv_file, index=False, header=False)

        result = load_points_to_evaluate(
            space=test_space, csv_file=temp_csv_file
        )

        assert len(result) == 3
        assert result[0] == ([1.0, 2.0], 10)  # Default rounds
        assert result[1] == ([3.0, 4.0], 10)
        assert result[2] == ([5.0, 6.0], 10)

    def test_csv_with_rounds_column(self, temp_csv_file, test_space):
        """Test CSV file with extra rounds column."""
        df = pd.DataFrame([[1.0, 2.0, 5], [3.0, 4.0, 10], [5.0, 6.0, 15]])
        df.to_csv(temp_csv_file, index=False, header=False)

        result = load_points_to_evaluate(
            space=test_space, csv_file=temp_csv_file
        )

        assert len(result) == 3
        assert result[0] == ([1.0, 2.0], 5)
        assert result[1] == ([3.0, 4.0], 10)
        assert result[2] == ([5.0, 6.0], 15)

    def test_custom_rounds_parameter(self, temp_csv_file, test_space):
        """Test with custom rounds parameter."""
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        df.to_csv(temp_csv_file, index=False, header=False)

        result = load_points_to_evaluate(
            space=test_space, csv_file=temp_csv_file, rounds=20
        )

        assert len(result) == 2
        assert result[0] == ([1.0, 2.0], 20)
        assert result[1] == ([3.0, 4.0], 20)

    def test_wrong_number_of_columns(self, temp_csv_file, test_space):
        """Test CSV with wrong number of columns."""
        df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]])  # 4 columns for 2D space
        df.to_csv(temp_csv_file, index=False, header=False)

        with pytest.raises(ValueError, match="Number of columns"):
            load_points_to_evaluate(space=test_space, csv_file=temp_csv_file)

    def test_points_outside_bounds(self, temp_csv_file, test_space):
        """Test CSV with points outside the bounds."""
        df = pd.DataFrame([[1.0, 2.0], [15.0, 4.0]])  # 15.0 is outside [0, 10]
        df.to_csv(temp_csv_file, index=False, header=False)

        with pytest.raises(ValueError, match="outside of the specified bounds"):
            load_points_to_evaluate(space=test_space, csv_file=temp_csv_file)

    def test_boundary_points(self, temp_csv_file, test_space):
        """Test CSV with points exactly at boundaries."""
        df = pd.DataFrame([[0.0, 0.0], [10.0, 10.0]])
        df.to_csv(temp_csv_file, index=False, header=False)

        result = load_points_to_evaluate(
            space=test_space, csv_file=temp_csv_file
        )

        assert len(result) == 2
        assert result[0] == ([0.0, 0.0], 10)
        assert result[1] == ([10.0, 10.0], 10)

    def test_three_dimensional_space(self, temp_csv_file):
        """Test with 3D space."""
        space = Space([Real(0, 10), Real(0, 10), Real(0, 10)])
        df = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        df.to_csv(temp_csv_file, index=False, header=False)

        result = load_points_to_evaluate(space=space, csv_file=temp_csv_file)

        assert len(result) == 2
        assert result[0] == ([1.0, 2.0, 3.0], 10)
        assert result[1] == ([4.0, 5.0, 6.0], 10)

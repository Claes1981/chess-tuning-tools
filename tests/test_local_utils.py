"""Comprehensive tests for utility functions in tune/local.py.

Tests inputs_uniform, reduce_ranges, load_points_to_evaluate, counts_to_penta,
setup_logger, and pause_between_times.
"""

import io
import logging
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from datetime import time as datetime_time
from pathlib import Path
from unittest.mock import MagicMock, patch

from skopt.space import Categorical, Integer, Real, Space

from tune.local import (
    counts_to_penta,
    inputs_uniform,
    load_points_to_evaluate,
    pause_between_times,
    reduce_ranges,
    setup_logger,
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


class TestCountsToPenta:
    """Tests for the counts_to_penta function."""

    def test_all_draws_gives_zero_elo(self):
        """All draws (middle element only) should give Elo of 0."""
        counts = np.array([0, 0, 100, 0, 0])
        mean_elo, variance = counts_to_penta(counts)
        assert np.isclose(mean_elo, 0.0, rtol=1e-5)

    def test_more_wins_gives_negative_elo(self):
        """Penta with more wins (WW) should give negative Elo (good for ref)."""
        counts = np.array([50, 20, 10, 10, 10])
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo < 0

    def test_more_losses_gives_positive_elo(self):
        """Penta with more losses (LL) should give positive Elo (bad for ref)."""
        counts = np.array([10, 10, 10, 20, 50])
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo > 0

    def test_variance_decreases_with_larger_counts(self):
        """Larger sample sizes should give smaller variance."""
        small_counts = np.array([10, 10, 20, 10, 10])
        large_counts = np.array([1000, 1000, 2000, 1000, 1000])
        _, variance_small = counts_to_penta(small_counts)
        _, variance_large = counts_to_penta(large_counts)
        assert variance_large < variance_small

    def test_custom_score_scale(self):
        """Should accept custom score_scale parameter."""
        counts = np.array([50, 20, 10, 10, 10])
        mean_elo_default, _ = counts_to_penta(counts, score_scale=4.0)
        mean_elo_custom, _ = counts_to_penta(counts, score_scale=2.0)
        assert mean_elo_default != mean_elo_custom

    def test_random_state_reproducibility(self):
        """Same random_state should give reproducible results."""
        counts = np.array([10, 10, 20, 10, 10])
        mean1, var1 = counts_to_penta(counts, random_state=42)
        mean2, var2 = counts_to_penta(counts, random_state=42)
        assert np.isclose(mean1, mean2)
        assert np.isclose(var1, var2)

    def test_different_random_state_gives_different_variance(self):
        """Different random_state may give slightly different variance."""
        counts = np.array([5, 5, 10, 5, 5])
        _, var1 = counts_to_penta(counts, random_state=42)
        _, var2 = counts_to_penta(counts, random_state=123)
        assert abs(var1 - var2) < max(var1, var2) * 0.5

    def test_mixed_results(self):
        """Test with realistic mixed game results."""
        counts = np.array([30, 25, 50, 25, 30])
        mean_elo, variance = counts_to_penta(counts)
        assert isinstance(mean_elo, float)
        assert isinstance(variance, float)
        assert variance >= 0

    def test_very_small_counts(self):
        """Should handle very small counts (with larger variance)."""
        counts = np.array([1, 1, 2, 1, 1])
        mean_elo, variance = counts_to_penta(counts)
        assert isinstance(mean_elo, float)
        assert isinstance(variance, float)
        assert variance >= 0

    def test_very_large_counts(self):
        """Should handle very large counts (with smaller variance)."""
        counts = np.array([10000, 10000, 20000, 10000, 10000])
        mean_elo, variance = counts_to_penta(counts)
        assert isinstance(mean_elo, float)
        assert isinstance(variance, float)
        assert variance >= 0
        assert variance < 1.0


class TestSetupLogger:
    """Tests for the setup_logger function."""

    def test_creates_logger(self):
        """Test that setup_logger creates a logger."""
        logger = setup_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ChessTuner"

    def test_with_verbose_0(self):
        """Test with verbose=0."""
        logger = setup_logger(verbose=0)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

    def test_with_verbose_1(self):
        """Test with verbose=1."""
        logger = setup_logger(verbose=1)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.DEBUG

    def test_with_logfile(self, tmp_path):
        """Test with logfile parameter."""
        log_file = tmp_path / "test_log.txt"
        logger = setup_logger(logfile=str(log_file))
        assert isinstance(logger, logging.Logger)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) > 0

    def test_logger_has_handlers(self):
        """Test that logger has both file and console handlers."""
        logger = setup_logger()
        assert len(logger.handlers) >= 2
        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types
        assert logging.StreamHandler in handler_types

    def test_logger_propagate_is_false(self):
        """Logger propagate should be set to False."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name

        try:
            logger = setup_logger(verbose=0, logfile=log_path)
            assert logger.propagate is False
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_logger_writes_to_file(self):
        """Logger should write to the specified log file."""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".log"
        ) as f:
            log_path = f.name

        try:
            logger = setup_logger(verbose=0, logfile=log_path)
            logger.info("Test message")
            assert os.path.exists(log_path)
            with open(log_path, "r") as f:
                content = f.read()
            assert "Test message" in content
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_logger_returns_same_instance(self):
        """Should return the same logger instance for repeated calls."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name

        try:
            logger1 = setup_logger(verbose=0, logfile=log_path)
            logger2 = setup_logger(verbose=1, logfile=log_path)
            assert logger1 is logger2
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)


class TestPauseBetweenTimes:
    """Tests for the pause_between_times function."""

    def test_time_not_in_range(self, monkeypatch):
        """Test when current time is not in pause range."""
        start_time = datetime_time(23, 0)
        end_time = datetime_time(6, 0)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.time.return_value = datetime_time(14, 0)
        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        pause_between_times(start_time, end_time)

    def test_time_in_range_day_boundary(self, monkeypatch, capsys):
        """Test when current time is in pause range (crossing day boundary)."""
        from datetime import date, datetime as dt_class

        start_time = datetime_time(23, 0)
        end_time = datetime_time(6, 0)

        mock_now = dt_class(2024, 1, 15, 2, 0)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value = mock_now
        mock_datetime.combine = dt_class.combine

        monkeypatch.setattr("tune.local.datetime", mock_datetime)
        monkeypatch.setattr("tune.local.time", MagicMock())

        pause_between_times(start_time, end_time)

    def test_same_start_end_time(self, monkeypatch):
        """Test with same start and end time."""
        start_time = datetime_time(12, 0)
        end_time = datetime_time(12, 0)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value.time.return_value = datetime_time(14, 0)
        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        pause_between_times(start_time, end_time)

    def test_time_range_not_crossing_midnight(self, monkeypatch):
        """Test with time range not crossing midnight."""
        from datetime import datetime as dt_class

        start_time = datetime_time(10, 0)
        end_time = datetime_time(11, 0)

        mock_now = dt_class(2024, 1, 15, 10, 30)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value = mock_now
        mock_datetime.combine = dt_class.combine

        monkeypatch.setattr("tune.local.datetime", mock_datetime)
        monkeypatch.setattr("tune.local.time", MagicMock())

        pause_between_times(start_time, end_time)

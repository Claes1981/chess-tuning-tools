"""Tests for additional tune/local.py functions.

Tests functions that are not covered in test_local.py.
"""

import logging
import numpy as np
import pytest
from datetime import time as datetime_time, timedelta
from unittest.mock import MagicMock, patch

from tune.local import (
    counts_to_penta,
    elo_to_prob,
    inputs_uniform,
    pause_between_times,
    prob_to_elo,
    setup_logger,
)


class TestEloToProb:
    """Tests for the elo_to_prob function."""

    def test_basic_conversion(self):
        """Test basic ELO to probability conversion."""
        elo = np.array([0.0, 100.0, -100.0, 50.0, -50.0])
        prob = elo_to_prob(elo)

        assert isinstance(prob, np.ndarray)
        assert len(prob) == len(elo)
        assert np.all(prob >= 0)
        assert np.all(prob <= 1)
        assert np.isclose(prob[0], 0.5)  # 0 ELO = 50%

    def test_zero_elo_gives_0_5(self):
        """Test that 0 ELO gives 0.5 probability."""
        elo = np.array([0.0])
        prob = elo_to_prob(elo)

        assert np.isclose(prob[0], 0.5)

    def test_positive_elo_gives_greater_than_0_5(self):
        """Test that positive ELO gives probability > 0.5."""
        elo = np.array([100.0, 200.0, 500.0])
        prob = elo_to_prob(elo)

        assert np.all(prob > 0.5)

    def test_negative_elo_gives_less_than_0_5(self):
        """Test that negative ELO gives probability < 0.5."""
        elo = np.array([-100.0, -200.0, -500.0])
        prob = elo_to_prob(elo)

        assert np.all(prob < 0.5)

    def test_symmetry(self):
        """Test that elo_to_prob(-elo) = 1 - elo_to_prob(elo)."""
        elo = np.array([50.0, 100.0, 200.0])
        prob_pos = elo_to_prob(elo)
        prob_neg = elo_to_prob(-elo)

        assert np.allclose(prob_neg, 1 - prob_pos)

    def test_custom_k_parameter(self):
        """Test with custom k parameter."""
        elo = np.array([100.0])
        prob_k4 = elo_to_prob(elo, k=4.0)
        prob_k8 = elo_to_prob(elo, k=8.0)

        assert prob_k4 != prob_k8  # Different k should give different results

    def test_large_elo_values(self):
        """Test with very large ELO values."""
        elo = np.array([1000.0, -1000.0, 10000.0, -10000.0])
        prob = elo_to_prob(elo)

        assert np.all(prob >= 0)
        assert np.all(prob <= 1)
        assert prob[0] > 0.9  # Very high ELO
        assert prob[1] < 0.1  # Very low ELO

    def test_scalar_input_returns_array(self):
        """Test that scalar input returns array (np.atleast_1d)."""
        elo = 100.0
        prob = elo_to_prob(elo)

        assert isinstance(prob, np.ndarray)
        assert len(prob) == 1
        assert prob[0] > 0.5


class TestProbToElo:
    """Tests for the prob_to_elo function."""

    def test_basic_conversion(self):
        """Test basic probability to ELO conversion."""
        prob = np.array([0.5, 0.75, 0.25, 0.9, 0.1])
        elo = prob_to_elo(prob)

        assert isinstance(elo, np.ndarray)
        assert len(elo) == len(prob)
        assert np.isclose(elo[0], 0.0)  # 0.5 = 0 ELO

    def test_0_5_gives_zero_elo(self):
        """Test that 0.5 probability gives 0 ELO."""
        prob = np.array([0.5])
        elo = prob_to_elo(prob)

        assert np.isclose(elo[0], 0.0)

    def test_greater_than_0_5_gives_positive_elo(self):
        """Test that probability > 0.5 gives positive ELO."""
        prob = np.array([0.6, 0.75, 0.9, 0.99])
        elo = prob_to_elo(prob)

        assert np.all(elo > 0)

    def test_less_than_0_5_gives_negative_elo(self):
        """Test that probability < 0.5 gives negative ELO."""
        prob = np.array([0.4, 0.25, 0.1, 0.01])
        elo = prob_to_elo(prob)

        assert np.all(elo < 0)

    def test_inverse_of_elo_to_prob(self):
        """Test that prob_to_elo is inverse of elo_to_prob for moderate values."""
        # Use moderate values that don't cause numerical issues
        elo_original = np.array([0.0, 50.0, -50.0])
        prob = elo_to_prob(elo_original)
        elo_recovered = prob_to_elo(prob)

        assert np.allclose(elo_original, elo_recovered)

    def test_custom_k_parameter(self):
        """Test with custom k parameter."""
        prob = np.array([0.75])
        elo_k4 = prob_to_elo(prob, k=4.0)
        elo_k8 = prob_to_elo(prob, k=8.0)

        assert elo_k4 != elo_k8  # Different k should give different results

    def test_extreme_probabilities(self):
        """Test with extreme probabilities close to 0 and 1."""
        prob = np.array([0.001, 0.999])
        elo = prob_to_elo(prob)

        assert elo[0] < 0  # Very low prob
        assert elo[1] > 0  # Very high prob
        assert np.isclose(
            abs(elo[0]), abs(elo[1])
        )  # Symmetric (with tolerance)

    def test_scalar_input_returns_array(self):
        """Test that scalar input returns array (np.atleast_1d)."""
        prob = 0.75
        elo = prob_to_elo(prob)

        assert isinstance(elo, np.ndarray)
        assert len(elo) == 1
        assert elo[0] > 0

    def test_extreme_values_return_inf_or_nan(self):
        """Test that extreme probabilities (0.0 or 1.0) return -inf or NaN."""
        prob = np.array([0.0, 1.0])
        elo = prob_to_elo(prob)

        assert np.isinf(elo[0])
        assert elo[0] < 0  # 0.0 probability gives -inf
        assert np.isnan(elo[1])  # 1.0 probability gives NaN due to log10(0)


class TestCountsToPenta:
    """Tests for the counts_to_penta function."""

    def test_basic_conversion(self):
        """Test basic counts to penta conversion returns tuple."""
        counts = np.array([10, 5, 3, 2, 0])
        result = counts_to_penta(counts)

        assert isinstance(result, tuple)
        assert len(result) == 2
        mean, std = result
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_all_wins(self):
        """Test with all wins."""
        counts = np.array([10, 0, 0, 0, 0])
        result = counts_to_penta(counts)
        mean, std = result

        assert mean < 0  # All wins is negative (good) for minimizer
        assert std >= 0

    def test_all_losses(self):
        """Test with all losses."""
        counts = np.array([0, 0, 0, 0, 10])
        result = counts_to_penta(counts)
        mean, std = result

        assert mean > 0  # All losses is positive (bad) for minimizer
        assert std >= 0

    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        counts = np.array([1, 1, 1, 1, 1])
        result = counts_to_penta(counts)
        mean, std = result

        assert np.isclose(mean, 0.0)  # Uniform = 0 ELO
        assert std >= 0

    def test_zero_counts_no_error(self):
        """Test that zero total counts does not raise error."""
        counts = np.array([0, 0, 0, 0, 0])

        # Should not raise ZeroDivisionError
        result = counts_to_penta(counts)
        mean, std = result
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_large_counts(self):
        """Test with large count values."""
        counts = np.array([1000, 2000, 3000, 2500, 1500])
        result = counts_to_penta(counts)
        mean, std = result

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std >= 0

    def test_custom_prior_counts(self):
        """Test with custom prior counts."""
        counts = np.array([10, 10, 10, 10, 10])
        prior = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = counts_to_penta(counts, prior_counts=prior)
        mean, std = result

        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_invalid_prior_counts_raises(self):
        """Test that invalid prior counts raises ValueError."""
        counts = np.array([10, 10, 10, 10, 10])
        prior = [1.0, 1.0, 1.0]  # Only 3 elements

        with pytest.raises(ValueError, match="should contain 5 elements"):
            counts_to_penta(counts, prior_counts=prior)


class TestInputsUniform:
    """Tests for the inputs_uniform function."""

    def test_basic_generation(self):
        """Test basic uniform input generation."""
        n_samples = 10
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        inputs = inputs_uniform(n_samples, lb, ub)

        assert isinstance(inputs, np.ndarray)
        assert len(inputs) == n_samples
        assert inputs.shape[1] == 2

    def test_values_within_bounds(self):
        """Test that generated values are within bounds."""
        n_samples = 100
        lb = np.array([0.0, -1.0])
        ub = np.array([1.0, 1.0])

        inputs = inputs_uniform(n_samples, lb, ub)

        assert np.all(inputs[:, 0] >= lb[0])
        assert np.all(inputs[:, 0] <= ub[0])
        assert np.all(inputs[:, 1] >= lb[1])
        assert np.all(inputs[:, 1] <= ub[1])

    def test_single_dimension(self):
        """Test with single dimension."""
        n_samples = 10
        lb = np.array([0.0])
        ub = np.array([1.0])

        inputs = inputs_uniform(n_samples, lb, ub)

        assert inputs.shape == (n_samples, 1)

    def test_multiple_dimensions(self):
        """Test with multiple dimensions."""
        n_samples = 10
        lb = np.array([0.0, 0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0, 1.0])

        inputs = inputs_uniform(n_samples, lb, ub)

        assert inputs.shape == (n_samples, 4)

    def test_different_bounds(self):
        """Test with different lower and upper bounds."""
        n_samples = 10
        lb = np.array([-10.0, 5.0])
        ub = np.array([10.0, 15.0])

        inputs = inputs_uniform(n_samples, lb, ub)

        assert np.all(inputs[:, 0] >= -10.0)
        assert np.all(inputs[:, 0] <= 10.0)
        assert np.all(inputs[:, 1] >= 5.0)
        assert np.all(inputs[:, 1] <= 15.0)


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
        # Check that file handler was added
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


class TestPauseBetweenTimes:
    """Tests for the pause_between_times function."""

    def test_time_not_in_range(self, monkeypatch):
        """Test when current time is not in pause range."""
        start_time = datetime_time(23, 0)  # 11 PM
        end_time = datetime_time(6, 0)  # 6 AM

        # Mock current time to be 2 PM (not in range)
        mock_datetime = MagicMock()
        mock_datetime.now.return_value.time.return_value = datetime_time(14, 0)
        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        # Should not raise or pause
        pause_between_times(start_time, end_time)

    def test_time_in_range_day_boundary(self, monkeypatch, capsys):
        """Test when current time is in pause range (crossing day boundary)."""
        from datetime import date, datetime as dt_class

        start_time = datetime_time(23, 0)  # 11 PM
        end_time = datetime_time(6, 0)  # 6 AM

        # Create a proper mock datetime object
        mock_now = dt_class(2024, 1, 15, 2, 0)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value = mock_now
        # Mock combine to return a proper datetime
        mock_datetime.combine = dt_class.combine

        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        # Mock time.sleep to avoid actually sleeping
        monkeypatch.setattr("tune.local.time", MagicMock())

        # Should not raise
        pause_between_times(start_time, end_time)

    def test_same_start_end_time(self, monkeypatch):
        """Test with same start and end time."""
        start_time = datetime_time(12, 0)  # Noon
        end_time = datetime_time(12, 0)  # Noon

        # Mock current time
        mock_datetime = MagicMock()
        mock_datetime.now.return_value.time.return_value = datetime_time(14, 0)
        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        pause_between_times(start_time, end_time)

    def test_time_range_not_crossing_midnight(self, monkeypatch):
        """Test with time range not crossing midnight."""
        from datetime import datetime as dt_class

        start_time = datetime_time(10, 0)  # 10 AM
        end_time = datetime_time(11, 0)  # 11 AM

        # Create a proper mock datetime object
        mock_now = dt_class(2024, 1, 15, 10, 30)

        mock_datetime = MagicMock()
        mock_datetime.now.return_value = mock_now
        # Mock combine to return a proper datetime
        mock_datetime.combine = dt_class.combine

        monkeypatch.setattr("tune.local.datetime", mock_datetime)

        # Mock time.sleep to avoid actually sleeping
        monkeypatch.setattr("tune.local.time", MagicMock())

        pause_between_times(start_time, end_time)

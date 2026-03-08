"""Comprehensive tests for counts_to_penta and setup_logger functions."""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tune.local import counts_to_penta, setup_logger


class TestCountsToPenta:
    """Tests for counts_to_penta function.

    The function takes a 5-element penta distribution array (WW, WD, WL/DD, LD, LL)
    and returns a tuple of (mean_elo, variance).

    Note: The scoring is such that:
    - WW (double win) = 0.0 score
    - WD (win-draw) = 0.25 score
    - WL/DD (win-loss or double draw) = 0.5 score
    - LD (draw-loss) = 0.75 score
    - LL (double loss) = 1.0 score

    This means more WW gives negative Elo (good for reference engine)
    and more LL gives positive Elo (bad for reference engine).
    """

    def test_returns_tuple_of_mean_and_variance(self):
        """Function should return tuple of (mean_elo, variance)."""
        counts = np.array([10, 10, 20, 10, 10])  # Symmetric penta
        result = counts_to_penta(counts)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # mean elo
        assert isinstance(result[1], float)  # variance

    def test_symmetric_penta_gives_zero_elo(self):
        """Symmetric penta distribution should give Elo near 0."""
        counts = np.array([10, 10, 20, 10, 10])
        mean_elo, variance = counts_to_penta(counts)
        assert abs(mean_elo) < 1.0  # Should be very close to 0

    def test_all_draws_gives_zero_elo(self):
        """All draws (middle element only) should give Elo of 0."""
        counts = np.array([0, 0, 100, 0, 0])
        mean_elo, variance = counts_to_penta(counts)
        assert_allclose(mean_elo, 0.0, rtol=1e-5)

    def test_more_wins_gives_negative_elo(self):
        """Penta with more wins (WW) should give negative Elo (good for ref)."""
        counts = np.array([50, 20, 10, 10, 10])  # More WW and WD
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo < 0

    def test_more_losses_gives_positive_elo(self):
        """Penta with more losses (LL) should give positive Elo (bad for ref)."""
        counts = np.array([10, 10, 10, 20, 50])  # More LD and LL
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo > 0

    def test_variance_is_positive(self):
        """Variance should always be non-negative."""
        counts = np.array([10, 10, 20, 10, 10])
        mean_elo, variance = counts_to_penta(counts)
        assert variance >= 0

    def test_variance_decreases_with_larger_counts(self):
        """Larger sample sizes should give smaller variance."""
        small_counts = np.array([10, 10, 20, 10, 10])
        large_counts = np.array([1000, 1000, 2000, 1000, 1000])
        _, variance_small = counts_to_penta(small_counts)
        _, variance_large = counts_to_penta(large_counts)
        assert variance_large < variance_small

    def test_custom_prior_counts(self):
        """Should accept custom prior counts."""
        counts = np.array([10, 10, 20, 10, 10])
        prior_counts = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
        mean_elo, variance = counts_to_penta(counts, prior_counts=prior_counts)
        assert isinstance(mean_elo, float)
        assert isinstance(variance, float)

    def test_raises_error_for_invalid_prior_counts_length(self):
        """Should raise ValueError if prior_counts doesn't have 5 elements."""
        counts = np.array([10, 10, 20, 10, 10])
        invalid_prior = np.array([1.0, 1.0, 1.0])  # Only 3 elements
        with pytest.raises(
            ValueError, match="prior_counts should contain 5 elements"
        ):
            counts_to_penta(counts, prior_counts=invalid_prior)

    def test_custom_score_scale(self):
        """Should accept custom score_scale parameter."""
        counts = np.array([50, 20, 10, 10, 10])
        mean_elo_default, _ = counts_to_penta(counts, score_scale=4.0)
        mean_elo_custom, _ = counts_to_penta(counts, score_scale=2.0)
        # Different score scales should give different results
        assert mean_elo_default != mean_elo_custom

    def test_random_state_reproducibility(self):
        """Same random_state should give reproducible results."""
        counts = np.array([10, 10, 20, 10, 10])
        mean1, var1 = counts_to_penta(counts, random_state=42)
        mean2, var2 = counts_to_penta(counts, random_state=42)
        assert_allclose(mean1, mean2)
        assert_allclose(var1, var2)

    def test_different_random_state_gives_different_variance(self):
        """Different random_state may give slightly different variance."""
        counts = np.array([5, 5, 10, 5, 5])  # Small counts for more variance
        _, var1 = counts_to_penta(counts, random_state=42)
        _, var2 = counts_to_penta(counts, random_state=123)
        # Variance estimates may differ slightly due to different samples
        # They should be in the same ballpark
        assert abs(var1 - var2) < max(var1, var2) * 0.5

    def test_all_wins_gives_negative_elo(self):
        """All WW (double wins) should give negative Elo (very good for ref)."""
        counts = np.array([100, 0, 0, 0, 0])
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo < 0

    def test_all_losses_gives_positive_elo(self):
        """All LL (double losses) should give positive Elo (very bad for ref)."""
        counts = np.array([0, 0, 0, 0, 100])
        mean_elo, variance = counts_to_penta(counts)
        assert mean_elo > 0

    def test_mixed_results(self):
        """Test with realistic mixed game results."""
        # Realistic distribution: some of each outcome
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
        # Large counts should give very small variance
        assert variance < 1.0


class TestSetupLogger:
    """Tests for setup_logger function.

    The function takes verbose (int) and logfile (str) parameters,
    and always uses the global LOGGER name "ChessTuner".
    """

    def test_creates_logger_with_chess_tuner_name(self):
        """Logger should always be created with name "ChessTuner"."""
        logger = setup_logger(verbose=0, logfile="test.log")
        assert logger.name == "ChessTuner"

    def test_verbose_0_gives_info_level(self):
        """verbose=0 should give INFO level."""
        logger = setup_logger(verbose=0, logfile="test.log")
        assert logger.level == logging.INFO

    def test_verbose_greater_than_0_gives_debug_level(self):
        """verbose>0 should give DEBUG level."""
        logger = setup_logger(verbose=1, logfile="test.log")
        assert logger.level == logging.DEBUG

    def test_logger_has_handlers(self):
        """Logger should have handlers (file and console)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name

        try:
            logger = setup_logger(verbose=0, logfile=log_path)
            # Should have at least file and console handlers
            assert len(logger.handlers) >= 2
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

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
            # Check if file was written
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
            # Should return the same logger instance (ChessTuner)
            assert logger1 is logger2
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

    def test_logger_with_default_parameters(self):
        """Should work with default parameters."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name

        try:
            # Use custom logfile to avoid polluting current directory
            logger = setup_logger(logfile=log_path)
            assert logger.name == "ChessTuner"
            assert logger.level == logging.INFO
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)

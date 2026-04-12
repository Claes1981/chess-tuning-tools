"""Comprehensive tests for tune/summary.py functions.

Tests the confidence_intervals function and its helper functions.
"""

import pytest
from unittest.mock import MagicMock

from tune.summary import confidence_intervals


class TestConfidenceIntervals:
    """Tests for the confidence_intervals function."""

    @pytest.fixture
    def mock_optimizer_1d(self):
        """Create a mock optimizer with 1D space."""
        optimizer = MagicMock()
        optimizer.space.dimensions = [MagicMock()]
        optimizer.space.n_dims = 1

        # Mock the optimum_intervals method
        def mock_optimum_intervals(**kwargs):
            return [[[0.5, 1.5]]]

        optimizer.optimum_intervals = mock_optimum_intervals
        return optimizer

    @pytest.fixture
    def mock_optimizer_2d(self):
        """Create a mock optimizer with 2D space."""
        optimizer = MagicMock()
        optimizer.space.dimensions = [MagicMock(), MagicMock()]
        optimizer.space.n_dims = 2

        def mock_optimum_intervals(**kwargs):
            return [[[0.5, 1.5], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]

        optimizer.optimum_intervals = mock_optimum_intervals
        return optimizer

    @pytest.fixture
    def mock_optimizer_multimodal(self):
        """Create a mock optimizer with multimodal results."""
        optimizer = MagicMock()
        optimizer.space.dimensions = [MagicMock()]
        optimizer.space.n_dims = 1

        def mock_optimum_intervals(**kwargs):
            # Return multiple modes
            return [[[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]]

        optimizer.optimum_intervals = mock_optimum_intervals
        return optimizer

    def test_basic_1d(self, mock_optimizer_1d):
        """Test basic 1D confidence interval computation."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            prob=0.95,
        )

        assert isinstance(result, str)
        assert "param1" in result
        assert "Lower bound" in result
        assert "Upper bound" in result
        assert "0.5" in result or "1.5" in result

    def test_2d_parameters(self, mock_optimizer_2d):
        """Test with 2D parameter space."""
        result = confidence_intervals(
            optimizer=mock_optimizer_2d,
            param_names=["param1", "param2"],
            prob=0.9,
        )

        assert isinstance(result, str)
        assert "param1" in result
        assert "param2" in result
        assert "Lower bound" in result
        assert "Upper bound" in result

    def test_custom_param_names(self, mock_optimizer_1d):
        """Test with custom parameter names."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["MyCustomParameter"],
        )

        assert "MyCustomParameter" in result

    def test_default_param_names(self, mock_optimizer_1d):
        """Test with default parameter names when param_names is None."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=None,
        )

        assert "Parameter 0" in result

    def test_multimodal_true(self, mock_optimizer_multimodal):
        """Test with multimodal=True."""
        result = confidence_intervals(
            optimizer=mock_optimizer_multimodal,
            param_names=["param1"],
            multimodal=True,
        )

        assert isinstance(result, str)
        # Should contain multiple intervals for multimodal case

    def test_multimodal_false(self, mock_optimizer_1d):
        """Test with multimodal=False."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            multimodal=False,
        )

        assert isinstance(result, str)

    def test_different_hdi_prob(self, mock_optimizer_1d):
        """Test with different probability values."""
        for prob_value in [0.5, 0.9, 0.95, 0.99]:
            result = confidence_intervals(
                optimizer=mock_optimizer_1d,
                param_names=["param1"],
                prob=prob_value,
            )
            assert isinstance(result, str)
            # Note: prob is not included in the output string
            assert "param1" in result

    def test_opt_samples_parameter(self, mock_optimizer_1d):
        """Test with different opt_samples values."""
        for opt_samples in [100, 500, 1000, 2000]:
            result = confidence_intervals(
                optimizer=mock_optimizer_1d,
                param_names=["param1"],
                opt_samples=opt_samples,
            )
            assert isinstance(result, str)

    def test_space_samples_parameter(self, mock_optimizer_1d):
        """Test with different space_samples values."""
        for space_samples in [100, 500, 1000, 5000]:
            result = confidence_intervals(
                optimizer=mock_optimizer_1d,
                param_names=["param1"],
                space_samples=space_samples,
            )
            assert isinstance(result, str)

    def test_only_mean_true(self, mock_optimizer_1d):
        """Test with only_mean=True."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            only_mean=True,
        )
        assert isinstance(result, str)

    def test_only_mean_false(self, mock_optimizer_1d):
        """Test with only_mean=False."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            only_mean=False,
        )
        assert isinstance(result, str)

    def test_random_state_int(self, mock_optimizer_1d):
        """Test with random_state as integer."""
        result1 = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            random_state=42,
        )
        result2 = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            random_state=42,
        )
        # Same seed should produce same result (though with mock this may not matter)
        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_random_state_none(self, mock_optimizer_1d):
        """Test with random_state=None."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            random_state=None,
        )
        assert isinstance(result, str)

    def test_max_precision_parameter(self, mock_optimizer_1d):
        """Test with different max_precision values."""
        for max_precision in [8, 16, 32, 64]:
            result = confidence_intervals(
                optimizer=mock_optimizer_1d,
                param_names=["param1"],
                max_precision=max_precision,
            )
            assert isinstance(result, str)

    def test_threshold_parameter(self, mock_optimizer_1d):
        """Test with different threshold values."""
        for threshold in [0.001, 0.01, 0.1]:
            result = confidence_intervals(
                optimizer=mock_optimizer_1d,
                param_names=["param1"],
                threshold=threshold,
            )
            assert isinstance(result, str)

    def test_output_format(self, mock_optimizer_1d):
        """Test the output format of confidence_intervals."""
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=["param1"],
            prob=0.95,
        )

        # Check that output contains expected sections
        lines = result.strip().split("\n")
        assert len(lines) >= 3  # Header, separator, at least one row

        # First line should contain headers
        assert "Parameter" in lines[0]
        assert "Lower bound" in lines[0]
        assert "Upper bound" in lines[0]

        # Second line should be a separator
        assert "-" in lines[1]

    def test_long_parameter_names(self, mock_optimizer_1d):
        """Test with very long parameter names."""
        long_name = "ThisIsAVeryLongParameterNameForTesting"
        result = confidence_intervals(
            optimizer=mock_optimizer_1d,
            param_names=[long_name],
        )

        assert long_name in result

    def test_multiple_modes_formatting(self, mock_optimizer_multimodal):
        """Test formatting when multiple modes exist."""
        result = confidence_intervals(
            optimizer=mock_optimizer_multimodal,
            param_names=["param1"],
            multimodal=True,
        )

        # Should have multiple rows for the same parameter
        lines = result.strip().split("\n")
        # Header + separator + multiple mode rows
        assert len(lines) >= 4


class TestRoundInterval:
    """Tests for the internal _round_interval function."""

    def test_round_simple_interval(self):
        """Test rounding of a simple interval."""
        from tune.summary import _round_interval

        interval = (1.234567, 2.345678)
        result = _round_interval(interval, threshold=0.01)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] < result[1]

    def test_round_zero_diff(self):
        """Test rounding when interval has zero difference."""
        from tune.summary import _round_interval

        interval = (1.5, 1.5)
        result = _round_interval(interval)

        assert result == (1.5, 1.5)

    def test_round_high_precision(self):
        """Test rounding with high precision requirement."""
        from tune.summary import _round_interval

        interval = (0.123456789, 0.987654321)
        result = _round_interval(interval, threshold=0.001, max_precision=64)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_round_low_threshold(self):
        """Test rounding with very low threshold."""
        from tune.summary import _round_interval

        interval = (1.0, 2.0)
        result = _round_interval(interval, threshold=0.0001)

        assert isinstance(result, tuple)


class TestRoundAllIntervals:
    """Tests for the internal _round_all_intervals function."""

    def test_round_1d_intervals(self):
        """Test rounding of 1D intervals."""
        from tune.summary import _round_all_intervals

        intervals = [[(1.234, 2.345)]]
        result = _round_all_intervals(intervals)

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_round_2d_intervals(self):
        """Test rounding of 2D intervals."""
        from tune.summary import _round_all_intervals

        intervals = [
            [(1.234, 2.345), (3.456, 4.567)],
            [(5.678, 6.789), (7.890, 8.901)],
        ]
        result = _round_all_intervals(intervals)

        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_round_with_custom_params(self):
        """Test rounding with custom threshold and max_precision."""
        from tune.summary import _round_all_intervals

        intervals = [[(1.234567, 2.345678)]]
        result = _round_all_intervals(
            intervals, threshold=0.001, max_precision=16
        )

        assert isinstance(result, list)

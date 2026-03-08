"""Tests for print_results function in local.py."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tune.local import print_results


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer object."""
    optimizer = MagicMock()
    optimizer.gp.chain_ = np.array([[0.1], [0.2], [0.3]])
    optimizer.gp.noise_set_to_zero.return_value.__enter__ = lambda self: (
        optimizer.gp
    )
    optimizer.gp.noise_set_to_zero.return_value.__exit__ = lambda self, *args: (
        None
    )
    optimizer.gp.predict.return_value = (np.array([-0.05]), np.array([0.01]))
    optimizer.space.transform = lambda x: x
    return optimizer


@pytest.fixture
def mock_result_object():
    """Create a mock result object."""
    result = MagicMock(spec=OptimizeResult)
    result.x = np.array([0.5])
    result.fun = -0.05
    return result


@pytest.fixture
def parameter_names():
    """Create parameter names."""
    return ["param1"]


def test_print_results_basic(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test print_results with basic inputs."""
    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
    ):
        mock_ucb.return_value = (np.array([0.5]), -0.05)
        mock_conf.return_value = 1.645
        mock_conf_int.return_value = "param1: 0.5 [0.3, 0.7]"

        best_point, elo, std = print_results(
            mock_optimizer, mock_result_object, parameter_names, confidence=0.9
        )

        assert isinstance(best_point, np.ndarray)
        assert isinstance(elo, float)
        assert isinstance(std, float)
        assert elo == 5.0
        assert std == 1.0


def test_print_results_confidence_interval(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test confidence interval calculation in print_results."""
    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
    ):
        mock_ucb.return_value = (np.array([0.5]), -0.1)
        mock_conf.return_value = 1.96
        mock_conf_int.return_value = "param1: 0.5 [0.3, 0.7]"

        best_point, elo, std = print_results(
            mock_optimizer, mock_result_object, parameter_names, confidence=0.95
        )

        assert elo == 10.0
        assert std == 1.0


def test_print_results_value_error(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test that ValueError is raised when expected_ucb fails."""
    with patch("tune.local.expected_ucb") as mock_ucb:
        mock_ucb.side_effect = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            print_results(mock_optimizer, mock_result_object, parameter_names)


def test_print_results_logging(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test that print_results logs correctly."""
    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
        patch("tune.local.logging") as mock_logging,
    ):
        mock_ucb.return_value = (np.array([0.5]), -0.05)
        mock_conf.return_value = 1.645
        mock_conf_int.return_value = "param1: 0.5 [0.3, 0.7]"
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger

        print_results(mock_optimizer, mock_result_object, parameter_names)

        # Check that logging calls were made
        assert mock_logger.info.call_count >= 3


def test_print_results_multiple_parameters(mock_optimizer, mock_result_object):
    """Test print_results with multiple parameters."""
    parameter_names = ["param1", "param2", "param3"]

    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
    ):
        mock_ucb.return_value = (np.array([0.1, 0.2, 0.3]), -0.08)
        mock_conf.return_value = 1.645
        mock_conf_int.return_value = "param1: 0.1 [0.0, 0.2]\nparam2: 0.2 [0.1, 0.3]\nparam3: 0.3 [0.2, 0.4]"

        best_point, elo, std = print_results(
            mock_optimizer, mock_result_object, parameter_names
        )

        assert len(best_point) == 3
        assert elo == 8.0
        assert std == 1.0


def test_print_results_high_confidence(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test print_results with high confidence level."""
    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
    ):
        mock_ucb.return_value = (np.array([0.5]), -0.05)
        mock_conf.return_value = 3.0  # Very high confidence
        mock_conf_int.return_value = "param1: 0.5 [0.3, 0.7]"

        best_point, elo, std = print_results(
            mock_optimizer,
            mock_result_object,
            parameter_names,
            confidence=0.999,
        )

        assert elo == 5.0
        # Check that higher confidence mult is used
        mock_conf.assert_called_with(0.999)


def test_print_results_zero_std(
    mock_optimizer, mock_result_object, parameter_names
):
    """Test print_results when standard deviation is zero."""
    optimizer = mock_optimizer
    optimizer.gp.predict.return_value = (np.array([-0.05]), np.array([0.0]))

    with (
        patch("tune.local.expected_ucb") as mock_ucb,
        patch("tune.local.confidence_to_mult") as mock_conf,
        patch("tune.local.confidence_intervals") as mock_conf_int,
    ):
        mock_ucb.return_value = (np.array([0.5]), -0.05)
        mock_conf.return_value = 1.645
        mock_conf_int.return_value = "param1: 0.5 [0.5, 0.5]"

        best_point, elo, std = print_results(
            optimizer, mock_result_object, parameter_names
        )

        assert elo == 5.0
        assert std == 0.0

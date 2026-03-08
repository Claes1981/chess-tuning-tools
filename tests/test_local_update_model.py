"""Tests for update_model function in local.py."""

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tune.local import update_model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer object."""
    optimizer = MagicMock()
    optimizer.gp.chain_ = None  # Initial state
    optimizer.gp_priors = None
    return optimizer


def test_update_model_initial_fit(mock_optimizer):
    """Test update_model with initial fit (chain_ is None)."""
    point = [0.5, 0.3]
    score = -0.05
    variance = 0.01

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 5),
        ]

        update_model(
            mock_optimizer,
            point,
            score,
            variance,
            gp_initial_burnin=100,
            gp_initial_samples=300,
        )

        # Verify tell was called with initial parameters
        mock_optimizer.tell.assert_called_once()
        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["gp_burnin"] == 100
        assert call_kwargs["gp_samples"] == 300
        assert call_kwargs["x"] == point
        assert call_kwargs["y"] == score


def test_update_model_subsequent_fit(mock_optimizer):
    """Test update_model with subsequent fit (chain_ is not None)."""
    optimizer = mock_optimizer
    optimizer.gp.chain_ = np.array([[0.1], [0.2]])  # Not None

    point = [0.5, 0.3]
    score = -0.05
    variance = 0.01

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 3),
        ]

        update_model(
            optimizer,
            point,
            score,
            variance,
            gp_burnin=5,
            gp_samples=300,
        )

        # Verify tell was called with subsequent parameters
        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["gp_burnin"] == 5
        assert call_kwargs["gp_samples"] == 300


def test_update_model_noise_scaling(mock_optimizer):
    """Test update_model with noise scaling coefficient."""
    point = [0.5]
    score = -0.05
    variance = 0.01
    noise_scaling = 2.0

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 1),
        ]

        update_model(
            mock_optimizer,
            point,
            score,
            variance,
            noise_scaling_coefficient=noise_scaling,
        )

        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["noise_vector"] == noise_scaling * variance


def test_update_model_value_error_retry(mock_optimizer):
    """Test update_model retries on ValueError."""
    optimizer = mock_optimizer
    optimizer.gp.chain_ = np.array([[0.1]])  # Not None

    point = [0.5]
    score = -0.05
    variance = 0.01

    # Make tell raise ValueError once, then succeed
    call_count = [0]

    def mock_tell_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Test error")
        # Second call succeeds

    optimizer.tell.side_effect = mock_tell_side_effect

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),  # First attempt
            datetime(2024, 1, 1, 12, 0, 0),  # After sample
            datetime(2024, 1, 1, 12, 0, 1),  # Second attempt
            datetime(2024, 1, 1, 12, 0, 2),  # After success
        ]

        update_model(
            optimizer, point, score, variance, gp_burnin=5, gp_samples=100
        )

        # Verify sample was called after ValueError
        optimizer.gp.sample.assert_called_once()
        sample_kwargs = optimizer.gp.sample.call_args.kwargs
        assert sample_kwargs["n_burnin"] == 11
        assert sample_kwargs["priors"] is None

        # Verify tell was called twice
        assert optimizer.tell.call_count == 2


def test_update_model_custom_acq_function_samples(mock_optimizer):
    """Test update_model with custom acq_function_samples."""
    point = [0.5]
    score = -0.05
    variance = 0.01

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 1),
        ]

        update_model(
            mock_optimizer, point, score, variance, acq_function_samples=10
        )

        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["n_samples"] == 10


def test_update_model_custom_walkers(mock_optimizer):
    """Test update_model with custom gp_walkers_per_thread."""
    point = [0.5]
    score = -0.05
    variance = 0.01

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 1),
        ]

        update_model(
            mock_optimizer, point, score, variance, gp_walkers_per_thread=50
        )

        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["n_walkers_per_thread"] == 50


def test_update_model_logging(mock_optimizer):
    """Test update_model logs correctly."""
    point = [0.5]
    score = -0.05
    variance = 0.01

    with (
        patch("tune.local.datetime") as mock_datetime,
        patch("tune.local.logging") as mock_logging,
    ):
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 5),
        ]
        mock_logger = MagicMock()
        mock_logging.getLogger.return_value = mock_logger

        update_model(mock_optimizer, point, score, variance)

        # Check logging calls
        mock_logger.info.assert_called()
        assert "GP sampling finished" in mock_logger.info.call_args[0][0]


def test_update_model_zero_variance(mock_optimizer):
    """Test update_model with zero variance."""
    point = [0.5]
    score = -0.05
    variance = 0.0

    with patch("tune.local.datetime") as mock_datetime:
        mock_datetime.now.side_effect = [
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 1),
        ]

        update_model(mock_optimizer, point, score, variance)

        call_kwargs = mock_optimizer.tell.call_args.kwargs
        assert call_kwargs["noise_vector"] == 0.0

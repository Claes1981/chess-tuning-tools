"""Tests for optimizer initialization and model update in local.py.

Tests initialize_optimizer and update_model functions with various configurations.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from skopt.space import Integer, Real

from tune.local import initialize_optimizer, update_model


class TestInitializeOptimizer:
    """Tests for the initialize_optimizer function."""

    @pytest.fixture
    def basic_configs(self):
        """Create basic configuration dictionaries."""
        return {
            "X": [],
            "y": [],
            "noise": [],
            "parameter_ranges": [
                Integer(0, 100, name="param1"),
                Real(0.0, 1.0, name="param2"),
            ],
            "gp_config": {
                "normalize_y": True,
                "warp_inputs": False,
                "nu": 2.5,
            },
            "acq_function_config": {
                "function": "ei",
                "lcb_alpha": 1.96,
                "n_points": 10000,
                "n_initial_points": 10,
                "non_uncert_acq_function_evaluation_points": 5000,
            },
            "resume_config": {
                "resume": False,
                "model_path": None,
                "fast_resume": False,
            },
            "gp_priors": None,
            "random_seed": 42,
        }

    def test_rand_acq_function(self, basic_configs):
        """Test with rand acquisition function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "rand"

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            acq_func = call_kwargs["acq_func"]
            assert acq_func in ["ts", "lcb", "pvrs", "mes", "ei", "mean"]

    def test_inf_lcb_alpha(self, basic_configs):
        """Test with infinite lcb_alpha."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["lcb_alpha"] = float("inf")

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            assert call_kwargs["acq_func_kwargs"]["alpha"] == "inf"

    def test_pvrs_acq_function(self, basic_configs):
        """Test with pvrs acquisition function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "pvrs"

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            assert call_kwargs["n_points"] == 5000

    def test_ts_acq_function(self, basic_configs):
        """Test with ts acquisition function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "ts"

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            assert call_kwargs["n_points"] == 5000

    def test_vr_acq_function(self, basic_configs):
        """Test with vr acquisition function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "vr"

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            assert call_kwargs["n_points"] == 5000

    def test_lcb_acq_function(self, basic_configs):
        """Test with lcb acquisition function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "lcb"

        with patch("tune.local.Optimizer") as mock_opt:
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            optimizer = initialize_optimizer(**configs)

            call_kwargs = mock_opt.call_args.kwargs
            assert call_kwargs["n_points"] == 10000
            assert call_kwargs["acq_func"] == "lcb"

    def test_resume_with_gp_priors(self, basic_configs):
        """Test with resume and gp_priors."""
        configs = basic_configs.copy()
        configs["resume_config"]["resume"] = True
        configs["resume_config"]["model_path"] = "/fake/path/model.pkl"
        configs["resume_config"]["fast_resume"] = True
        configs["gp_priors"] = {"length_scale": "uniform(0, 1)"}
        configs["X"] = [[0.5, 0.3]]
        configs["y"] = [-0.05]
        configs["noise"] = [0.01]

        with (
            patch("tune.local.pathlib.Path") as mock_path,
            patch("tune.local.dill") as mock_dill,
            patch("tune.local.Optimizer") as mock_opt,
        ):
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True

            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            old_opt = MagicMock()
            old_opt.space = mock_inst.space
            mock_dill.load.return_value = old_opt

            with patch("builtins.open", MagicMock()):
                optimizer = initialize_optimizer(**configs)

                assert old_opt.gp_priors == {"length_scale": "uniform(0, 1)"}

    def test_resume_space_mismatch(self, basic_configs):
        """Test when resumed optimizer has different space."""
        configs = basic_configs.copy()
        configs["resume_config"]["resume"] = True
        configs["resume_config"]["model_path"] = "/fake/path/model.pkl"
        configs["resume_config"]["fast_resume"] = True
        configs["gp_priors"] = {"length_scale": "uniform(0, 1)"}

        with (
            patch("tune.local.pathlib.Path") as mock_path,
            patch("tune.local.dill") as mock_dill,
            patch("tune.local.Optimizer") as mock_opt,
        ):
            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True

            old_opt = MagicMock()
            old_opt.space = "different_space"

            mock_dill.load.return_value = old_opt

            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            with patch("builtins.open", MagicMock()):
                optimizer = initialize_optimizer(**configs)

                assert mock_inst.gp_priors == {"length_scale": "uniform(0, 1)"}

    def test_rand_acq_debug_log(self, basic_configs):
        """Test debug logging for rand acq function."""
        configs = basic_configs.copy()
        configs["acq_function_config"]["function"] = "rand"
        configs["resume_config"]["resume"] = True
        configs["X"] = [[0.5, 0.3]]
        configs["y"] = [-0.05]
        configs["noise"] = [0.01]

        with (
            patch("tune.local.Optimizer") as mock_opt,
            patch("tune.local.logging") as mock_logging,
        ):
            mock_inst = MagicMock()
            mock_inst.space.n_dims = 2
            mock_opt.return_value = mock_inst

            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger

            optimizer = initialize_optimizer(**configs)

            assert mock_logger.debug.call_count > 0


class TestUpdateModel:
    """Tests for the update_model function."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer object."""
        optimizer = MagicMock()
        optimizer.gp.chain_ = None
        optimizer.gp_priors = None
        return optimizer

    def test_initial_fit(self, mock_optimizer):
        """Test with initial fit (chain_ is None)."""
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

            mock_optimizer.tell.assert_called_once()
            call_kwargs = mock_optimizer.tell.call_args.kwargs
            assert call_kwargs["gp_burnin"] == 100
            assert call_kwargs["gp_samples"] == 300
            assert call_kwargs["x"] == point
            assert call_kwargs["y"] == score

    def test_subsequent_fit(self, mock_optimizer):
        """Test with subsequent fit (chain_ is not None)."""
        optimizer = mock_optimizer
        optimizer.gp.chain_ = np.array([[0.1], [0.2]])

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

            call_kwargs = mock_optimizer.tell.call_args.kwargs
            assert call_kwargs["gp_burnin"] == 5
            assert call_kwargs["gp_samples"] == 300

    def test_noise_scaling(self, mock_optimizer):
        """Test with noise scaling coefficient."""
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

    def test_value_error_retry(self, mock_optimizer):
        """Test retries on ValueError."""
        optimizer = mock_optimizer
        optimizer.gp.chain_ = np.array([[0.1]])

        point = [0.5]
        score = -0.05
        variance = 0.01

        call_count = [0]

        def mock_tell_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")

        optimizer.tell.side_effect = mock_tell_side_effect

        with patch("tune.local.datetime") as mock_datetime:
            mock_datetime.now.side_effect = [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 1),
                datetime(2024, 1, 1, 12, 0, 2),
            ]

            update_model(
                optimizer, point, score, variance, gp_burnin=5, gp_samples=100
            )

            optimizer.gp.sample.assert_called_once()
            sample_kwargs = optimizer.gp.sample.call_args.kwargs
            assert sample_kwargs["n_burnin"] == 11
            assert sample_kwargs["priors"] is None

            assert optimizer.tell.call_count == 2

    def test_custom_acq_function_samples(self, mock_optimizer):
        """Test with custom acq_function_samples."""
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

    def test_custom_walkers(self, mock_optimizer):
        """Test with custom gp_walkers_per_thread."""
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

    def test_logging(self, mock_optimizer):
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

            mock_logger.info.assert_called()
            assert "GP sampling finished" in mock_logger.info.call_args[0][0]

    def test_zero_variance(self, mock_optimizer):
        """Test with zero variance."""
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

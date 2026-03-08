"""Tests for initialize_optimizer edge cases in local.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from skopt.space import Integer, Real

from tune.local import initialize_optimizer


@pytest.fixture
def basic_configs():
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


def test_initialize_optimizer_rand_acq_function(basic_configs):
    """Test initialize_optimizer with rand acquisition function (line 564)."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["function"] = "rand"

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that a random acquisition function was chosen
        call_kwargs = mock_opt.call_args.kwargs
        acq_func = call_kwargs["acq_func"]
        assert acq_func in ["ts", "lcb", "pvrs", "mes", "ei", "mean"]


def test_initialize_optimizer_inf_lcb_alpha(basic_configs):
    """Test initialize_optimizer with infinite lcb_alpha (line 572)."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["lcb_alpha"] = float("inf")

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that alpha is converted to string "inf"
        call_kwargs = mock_opt.call_args.kwargs
        assert call_kwargs["acq_func_kwargs"]["alpha"] == "inf"


def test_initialize_optimizer_pvrs_acq_function(basic_configs):
    """Test initialize_optimizer with pvrs acquisition function (line 578)."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["function"] = "pvrs"

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that non_uncert_acq_function_evaluation_points is used
        call_kwargs = mock_opt.call_args.kwargs
        assert call_kwargs["n_points"] == 5000


def test_initialize_optimizer_ts_acq_function(basic_configs):
    """Test initialize_optimizer with ts acquisition function (line 578)."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["function"] = "ts"

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that non_uncert_acq_function_evaluation_points is used
        call_kwargs = mock_opt.call_args.kwargs
        assert call_kwargs["n_points"] == 5000


def test_initialize_optimizer_resume_with_gp_priors(basic_configs):
    """Test initialize_optimizer with resume and gp_priors (line 623)."""
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

            # Verify that gp_priors was set on the old optimizer
            assert old_opt.gp_priors == {"length_scale": "uniform(0, 1)"}


def test_initialize_optimizer_resume_space_mismatch(basic_configs):
    """Test initialize_optimizer when resumed optimizer has different space (line 623 path)."""
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
        old_opt.space = "different_space"  # Different space

        mock_dill.load.return_value = old_opt

        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        with patch("builtins.open", MagicMock()):
            optimizer = initialize_optimizer(**configs)

            # Verify that gp_priors was set on the new optimizer
            assert mock_inst.gp_priors == {"length_scale": "uniform(0, 1)"}


def test_initialize_optimizer_rand_acq_debug_log(basic_configs):
    """Test initialize_optimizer logs debug message for rand acq function (line 631)."""
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

        # Verify debug log was called - the actual acq_func used is captured
        assert mock_logger.debug.call_count > 0


def test_initialize_optimizer_vr_acq_function(basic_configs):
    """Test initialize_optimizer with vr acquisition function (line 578)."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["function"] = "vr"

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that non_uncert_acq_function_evaluation_points is used
        call_kwargs = mock_opt.call_args.kwargs
        assert call_kwargs["n_points"] == 5000


def test_initialize_optimizer_lcb_acq_function(basic_configs):
    """Test initialize_optimizer with lcb acquisition function."""
    configs = basic_configs.copy()
    configs["acq_function_config"]["function"] = "lcb"

    with patch("tune.local.Optimizer") as mock_opt:
        mock_inst = MagicMock()
        mock_inst.space.n_dims = 2
        mock_opt.return_value = mock_inst

        optimizer = initialize_optimizer(**configs)

        # Verify that n_points is used (not non_uncert_acq_function_evaluation_points)
        call_kwargs = mock_opt.call_args.kwargs
        assert call_kwargs["n_points"] == 10000
        assert call_kwargs["acq_func"] == "lcb"

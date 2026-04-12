"""Integration tests for uncovered areas in local.py, plots.py, and cli.py.

This module provides integration tests for areas that were previously
uncovered or had minimal test coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tune import cli
from tune.local import (
    _construct_engine_conf,
    check_if_pause,
    pause_between_times,
    plot_results,
    run_match,
)
from tune.plots import (
    partial_dependence,
)


class TestPlotResultsIntegration:
    """Integration tests for plot_results function."""

    @pytest.fixture
    def mock_optimizer_with_data(self):
        """Create a mock optimizer with sample data."""
        from tests.conftest import MockDimension, MockSpace

        optimizer = MagicMock()
        mock_dim = MockDimension(bounds=(0.0, 1.0), transformed_size=1)
        mock_space = MockSpace(n_dims=1, dimensions=[mock_dim])
        optimizer.space = mock_space
        optimizer.Xi = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        optimizer.yi = [1.0, 2.0, 1.5, 2.5, 3.0]
        optimizer.noisei = [0.1, 0.1, 0.1, 0.1, 0.1]
        optimizer._next_x = np.array([0.5])
        optimizer.gp = MagicMock()
        optimizer.gp.chain_ = np.array([[0.1], [0.2], [0.3]])
        optimizer.gp.kernel_.k1.k2.nu = 2.5
        optimizer.gp.predict.return_value = (np.array([0.5]), np.array([[0.1]]))
        return optimizer

    @pytest.fixture
    def mock_result_object(self):
        """Create a mock result object."""
        from tests.conftest import MockDimension, MockSpace

        result = MagicMock()
        result.x_iters = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        result.fun = np.array([1.0, 2.0, 1.5, 2.5, 3.0])
        mock_dim = MockDimension(bounds=(0.0, 1.0), transformed_size=1)
        result.space = MockSpace(n_dims=1, dimensions=[mock_dim])
        return result

    @patch("tune.local.ActiveSubspaces")
    @patch("tune.local.inputs_uniform")
    @patch("tune.local.corner.corner")
    @patch("tune.local.plot_objective")
    @patch("tune.local.plot_objective_1d")
    @patch("tune.local.plot_optima")
    @patch("tune.local.plot_performance")
    @patch("tune.local.plot_activesubspace_eigenvalues")
    @patch("tune.local.plot_activesubspace_eigenvectors")
    @patch("tune.local.plot_activesubspace_sufficient_summary")
    @patch("tune.local.plt")
    def test_plot_results_1d(
        self,
        mock_plt,
        mock_plot_sufficient_summary,
        mock_plot_eigenvectors,
        mock_plot_eigenvalues,
        mock_plot_performance,
        mock_plot_optima,
        mock_plot_objective_1d,
        mock_plot_objective,
        mock_corner,
        mock_inputs_uniform,
        mock_active_subspaces,
        mock_optimizer_with_data,
        mock_result_object,
        tmp_path,
    ):
        """Test plot_results with 1D optimizer."""
        mock_inputs_uniform.return_value = np.array([[0.1], [0.2], [0.3]])
        mock_active_subspaces_instance = MagicMock()
        mock_active_subspaces_instance.activity_scores = np.array([0.5])
        mock_active_subspaces_instance.evects = np.array([[1.0], [0.0]])
        mock_active_subspaces.return_value = mock_active_subspaces_instance

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.get_size_inches.return_value = (6.0, 6.0)
        mock_ax.get_ylim.return_value = (0.0, 1.0)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.Figure.return_value = mock_fig
        mock_plt.gcf.return_value = mock_fig
        mock_plt.gca.return_value = mock_ax
        mock_plot_objective_1d.return_value = (mock_fig, mock_ax)
        mock_plot_objective.return_value = (mock_fig, mock_ax)
        mock_plot_optima.return_value = (MagicMock(), MagicMock())
        mock_plot_performance.return_value = (MagicMock(), MagicMock())
        mock_plot_eigenvalues.return_value = MagicMock()
        mock_plot_eigenvectors.return_value = MagicMock()
        mock_plot_sufficient_summary.return_value = MagicMock()

        plot_config = {
            "path": str(tmp_path),
            "parameter_names": ["param1"],
            "iterations": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "elos": np.array([[100.0], [150.0], [120.0], [180.0], [200.0]]),
            "optima": np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]),
            "confidence": 0.9,
        }

        plot_results(
            optimizer=mock_optimizer_with_data,
            result_object=mock_result_object,
            plot_config=plot_config,
        )

        assert mock_corner.called
        assert mock_plot_objective_1d.called
        assert (
            not mock_plot_objective.called
        )  # plot_objective is only called for n_dims > 1
        assert mock_plot_optima.called
        assert mock_plot_performance.called

    @patch("tune.local.ActiveSubspaces")
    @patch("tune.local.inputs_uniform")
    @patch("tune.local.corner.corner")
    @patch("tune.local.plot_objective")
    @patch("tune.local.plot_optima")
    @patch("tune.local.plot_performance")
    @patch("tune.local.plot_activesubspace_eigenvalues")
    @patch("tune.local.plot_activesubspace_eigenvectors")
    @patch("tune.local.plot_activesubspace_sufficient_summary")
    def test_plot_results_2d(
        self,
        mock_plot_sufficient_summary,
        mock_plot_eigenvectors,
        mock_plot_eigenvalues,
        mock_plot_performance,
        mock_plot_optima,
        mock_plot_objective,
        mock_corner,
        mock_inputs_uniform,
        mock_active_subspaces,
        mock_optimizer_with_data,
        mock_result_object,
        tmp_path,
    ):
        """Test plot_results with 2D optimizer."""
        mock_inputs_uniform.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        mock_active_subspaces_instance = MagicMock()
        mock_active_subspaces_instance.activity_scores = np.array([0.5, 0.3])
        mock_active_subspaces_instance.evects = np.array(
            [[1.0, 0.0], [0.0, 1.0]]
        )
        mock_active_subspaces.return_value = mock_active_subspaces_instance

        mock_optimizer_with_data.space.n_dims = 2
        mock_optimizer_with_data.Xi = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
        )
        mock_result_object.space.n_dims = 2
        mock_result_object.x_iters = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plot_objective.return_value = (
            (mock_fig, mock_ax),
            (mock_fig, mock_ax),
        )
        mock_plot_optima.return_value = (MagicMock(), MagicMock())
        mock_plot_performance.return_value = (MagicMock(), MagicMock())
        mock_plot_eigenvalues.return_value = MagicMock()
        mock_plot_eigenvectors.return_value = MagicMock()
        mock_plot_sufficient_summary.return_value = MagicMock()

        plot_config = {
            "path": str(tmp_path),
            "parameter_names": ["param1", "param2"],
            "iterations": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "elos": np.array([[100.0], [150.0], [120.0], [180.0], [200.0]]),
            "optima": np.array(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 0.1]]
            ),
            "confidence": 0.9,
        }

        plot_results(
            optimizer=mock_optimizer_with_data,
            result_object=mock_result_object,
            plot_config=plot_config,
        )

        assert mock_corner.called
        assert mock_plot_objective.called

    @patch("tune.local.ActiveSubspaces")
    @patch("tune.local.inputs_uniform")
    @patch("tune.local.corner.corner")
    @patch("tune.local.plot_objective")
    @patch("tune.local.plot_optima")
    @patch("tune.local.plot_performance")
    @patch("tune.local.plot_activesubspace_eigenvalues")
    @patch("tune.local.plot_activesubspace_eigenvectors")
    @patch("tune.local.plot_activesubspace_sufficient_summary")
    def test_plot_results_with_polynomial_regression(
        self,
        mock_plot_sufficient_summary,
        mock_plot_eigenvectors,
        mock_plot_eigenvalues,
        mock_plot_performance,
        mock_plot_optima,
        mock_plot_objective,
        mock_corner,
        mock_inputs_uniform,
        mock_active_subspaces,
        mock_optimizer_with_data,
        mock_result_object,
        tmp_path,
    ):
        """Test plot_results includes polynomial regression when enough data points."""
        # Polynomial regression only runs for n_dims > 1
        mock_optimizer_with_data.space.n_dims = 2
        mock_optimizer_with_data.Xi = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 0.1],
                [0.2, 0.3],
                [0.4, 0.5],
                [0.6, 0.7],
                [0.8, 0.9],
            ]
        )
        mock_optimizer_with_data.yi = [
            1.0,
            2.0,
            1.5,
            2.5,
            3.0,
            1.2,
            1.8,
            2.2,
            2.8,
        ]
        mock_optimizer_with_data.noisei = [0.1] * 9
        mock_result_object.space.n_dims = 2
        mock_result_object.x_iters = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 0.1],
                [0.2, 0.3],
                [0.4, 0.5],
                [0.6, 0.7],
                [0.8, 0.9],
            ]
        )

        mock_inputs_uniform.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        mock_active_subspaces_instance = MagicMock()
        mock_active_subspaces_instance.activity_scores = np.array([0.5, 0.3])
        mock_active_subspaces_instance.evects = np.array(
            [[1.0, 0.0], [0.0, 1.0]]
        )
        mock_active_subspaces.return_value = mock_active_subspaces_instance

        mock_plot_objective.return_value = (MagicMock(), MagicMock())
        mock_plot_optima.return_value = (MagicMock(), MagicMock())
        mock_plot_performance.return_value = (MagicMock(), MagicMock())
        mock_plot_eigenvalues.return_value = MagicMock()
        mock_plot_eigenvectors.return_value = MagicMock()
        mock_plot_sufficient_summary.return_value = MagicMock()

        plot_config = {
            "path": str(tmp_path),
            "parameter_names": ["param1", "param2"],
            "iterations": np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            ),
            "elos": np.array(
                [
                    [100.0],
                    [150.0],
                    [120.0],
                    [180.0],
                    [200.0],
                    [110.0],
                    [140.0],
                    [160.0],
                    [190.0],
                ]
            ),
            "optima": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6],
                    [0.7, 0.8],
                    [0.9, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                    [0.6, 0.7],
                    [0.8, 0.9],
                ]
            ),
            "confidence": 0.9,
        }

        plot_results(
            optimizer=mock_optimizer_with_data,
            result_object=mock_result_object,
            plot_config=plot_config,
        )

        assert mock_corner.called
        assert mock_plot_objective.called


class TestRunMatchGenerator:
    """Integration tests for run_match generator."""

    @patch("tune.local.iter")
    @patch("subprocess.Popen")
    def test_run_match_generator_yields_output_lines(
        self, mock_popen, mock_iter
    ):
        """Test that run_match yields output from cutechess-cli."""
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        mock_iter.return_value = iter(
            [
                "[Info] Starting match\n",
                "[Game 1] Result: 1-0\n",
                "[Game 2] Result: 0-1\n",
            ]
        )

        output_lines = list(
            run_match(
                cutechesscli_command="cutechess-cli",
                rounds=2,
                engine1_tc="60+0.1",
                engine2_tc="60+0.1",
                tuning_config_name="test_config",
            )
        )

        assert len(output_lines) == 3
        assert "[Info] Starting match\n" in output_lines[0]

    @patch("tune.local.iter")
    @patch("subprocess.Popen")
    def test_run_match_builds_correct_command(self, mock_popen, mock_iter):
        """Test that run_match builds the correct cutechess-cli command."""
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        mock_iter.return_value = iter([])

        list(
            run_match(
                cutechesscli_command="/usr/local/bin/cutechess-cli",
                rounds=4,
                engine1_tc="10+1",
                engine2_tc="10+1",
                engine1_ponder=True,
                timemargin=500,
                tuning_config_name="test_config",
            )
        )

        call_args = mock_popen.call_args[0][0]
        assert "/usr/local/bin/cutechess-cli" in call_args
        assert "tc=10+1" in call_args


class TestPauseFunctions:
    """Integration tests for pause functions."""

    @patch("tune.local.time.sleep")
    def test_pause_between_times_sleeps_during_interval(self, mock_sleep):
        """Test pause_between_times sleeps when current time is within interval."""
        from datetime import datetime, time as datetime_time

        with patch("tune.local.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 8, 0)
            mock_datetime.combine.return_value = datetime(2024, 1, 1, 10, 0)

            pause_between_times(
                start_time=datetime_time(7, 0),
                end_time=datetime_time(10, 0),
            )

            mock_sleep.assert_called_once()

    @patch("tune.local.time.sleep")
    def test_pause_between_times_no_sleep_outside_interval(self, mock_sleep):
        """Test pause_between_times does not sleep when outside interval."""
        from datetime import datetime, time as datetime_time

        with patch("tune.local.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)

            pause_between_times(
                start_time=datetime_time(7, 0),
                end_time=datetime_time(10, 0),
            )

            mock_sleep.assert_not_called()

    def test_check_if_pause_reads_intervals_from_file(self):
        """Test check_if_pause correctly reads pause intervals from file."""
        from datetime import time as datetime_time

        with patch("tune.local.open", MagicMock()) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = "07:00-10:00\n14:30-18:30"
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("tune.local.pause_between_times") as mock_pause:
                check_if_pause()

                assert mock_pause.call_count == 2
                mock_pause.assert_any_call(
                    datetime_time(7, 0),
                    datetime_time(10, 0),
                )
                mock_pause.assert_any_call(
                    datetime_time(14, 30),
                    datetime_time(18, 30),
                )

    def test_check_if_pause_handles_midnight_spanning_intervals(self):
        """Test check_if_pause correctly handles intervals spanning midnight."""
        from datetime import time as datetime_time

        with patch("tune.local.open", MagicMock()) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = "22:00-06:00"
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("tune.local.pause_between_times") as mock_pause:
                check_if_pause()

                assert mock_pause.call_count == 2
                mock_pause.assert_any_call(
                    datetime_time(22, 0),
                    datetime_time(23, 59, 59),
                )
                mock_pause.assert_any_call(
                    datetime_time(0, 0),
                    datetime_time(6, 0),
                )


class TestConstructEngineConfEdgeCases:
    """Integration tests for _construct_engine_conf edge cases."""

    def test_construct_engine_conf_with_time_control(self):
        """Test _construct_engine_conf with time control parameter."""
        result = _construct_engine_conf(
            id=1,
            engine_tc="60+0.1",
        )

        assert "-engine" in result
        assert "conf=engine1" in result
        assert "tc=60+0.1" in result

    def test_construct_engine_conf_with_nodes_per_minute(self):
        """Test _construct_engine_conf with nodes per minute."""
        result = _construct_engine_conf(
            id=1,
            engine_npm=100000,
        )

        assert "-engine" in result
        assert "tc=inf" in result
        assert "nodes=100000" in result

    def test_construct_engine_conf_with_time_per_move(self):
        """Test _construct_engine_conf with time per move."""
        result = _construct_engine_conf(
            id=1,
            engine_st=5,
        )

        assert "-engine" in result
        assert "st=5" in result

    def test_construct_engine_conf_with_depth(self):
        """Test _construct_engine_conf with depth limit."""
        result = _construct_engine_conf(
            id=1,
            engine_depth=20,
        )

        assert "-engine" in result
        assert "tc=inf" in result
        assert "depth=20" in result

    def test_construct_engine_conf_with_ponder_and_timemargin(self):
        """Test _construct_engine_conf with ponder and timemargin options."""
        result = _construct_engine_conf(
            id=1,
            engine_tc="60+0.1",
            engine_ponder=True,
            timemargin=1000,
        )

        assert "-engine" in result
        assert "ponder" in result
        assert "timemargin=1000" in result

    def test_construct_engine_conf_raises_without_time_control(self):
        """Test _construct_engine_conf raises ValueError when no time control specified."""
        with pytest.raises(
            ValueError, match="No engine time control specified"
        ):
            _construct_engine_conf(id=1)


class TestPartialDependenceIntegration:
    """Integration tests for partial_dependence function."""

    @pytest.fixture
    def mock_space(self):
        """Create a mock space for partial dependence tests."""

        dimension1 = MagicMock()
        dimension1.transformed_size = 1
        dimension1.prior = "uniform"
        dimension1.bounds = (0, 1)
        dimension1.transform = lambda x: x
        dimension1.inverse_transform = lambda x: x

        dimension2 = MagicMock()
        dimension2.transformed_size = 1
        dimension2.prior = "uniform"
        dimension2.bounds = (0, 1)
        dimension2.transform = lambda x: x
        dimension2.inverse_transform = lambda x: x

        space = MagicMock()
        space.n_dims = 2
        space.dimensions = [dimension1, dimension2]
        space.transform = lambda x: np.array(x) if isinstance(x, list) else x
        space.inverse_transform = lambda x: (
            np.array(x) if isinstance(x, list) else x
        )
        space.rvs = lambda n_samples: np.random.rand(n_samples, 2)
        return space

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for partial dependence tests."""
        model = MagicMock()
        model.predict.return_value = np.array([1.5, 2.0, 1.8, 2.2])

        context_manager = MagicMock()
        context_manager.__enter__ = lambda self: model
        context_manager.__exit__ = lambda self, *args: None
        model.noise_set_to_zero.return_value = context_manager
        return model

    @patch("tune.plots._evenly_sample")
    def test_partial_dependence_1d(
        self, mock_evenly_sample, mock_space, mock_model
    ):
        """Test partial_dependence with 1D calculation."""
        xi_values = np.linspace(0, 1, 10)
        xi_transformed = np.linspace(0, 1, 10)
        mock_evenly_sample.return_value = (xi_values, xi_transformed)

        mock_model.predict.return_value = np.array([1.5, 2.0, 1.8, 2.2, 1.6])

        xi, yi = partial_dependence(
            space=mock_space,
            model=mock_model,
            regression_object=None,
            polynomial_features_object=None,
            i=0,
            j=None,
            n_points=10,
            n_samples=100,
        )

        assert len(xi) == 10
        assert len(yi) == 10
        assert mock_model.predict.called

    @patch("tune.plots._evenly_sample")
    def test_partial_dependence_2d(
        self, mock_evenly_sample, mock_space, mock_model
    ):
        """Test partial_dependence with 2D calculation."""
        xi_values = np.linspace(0, 1, 10)
        xi_transformed = np.linspace(0, 1, 10)
        mock_evenly_sample.return_value = (xi_values, xi_transformed)

        mock_model.predict.return_value = np.array([1.5, 2.0, 1.8, 2.2, 1.6])

        xi, yi, zi = partial_dependence(
            space=mock_space,
            model=mock_model,
            regression_object=None,
            polynomial_features_object=None,
            i=0,
            j=1,
            n_points=10,
            n_samples=100,
        )

        assert len(xi) == 10
        assert len(yi) == 10
        assert zi.shape == (10, 10)
        assert mock_model.predict.called

    @patch("tune.plots._evenly_sample")
    def test_partial_dependence_with_custom_x_eval(
        self, mock_evenly_sample, mock_space, mock_model
    ):
        """Test partial_dependence with custom evaluation points."""
        custom_x_eval = [0.1, 0.3, 0.5, 0.7, 0.9]
        xi_values = np.array(custom_x_eval)
        xi_transformed = np.array(custom_x_eval)
        mock_evenly_sample.return_value = (xi_values, xi_transformed)

        mock_model.predict.return_value = np.array([1.5, 2.0, 1.8, 2.2, 1.6])

        xi, yi = partial_dependence(
            space=mock_space,
            model=mock_model,
            regression_object=None,
            polynomial_features_object=None,
            i=0,
            j=None,
            n_points=5,
            n_samples=100,
            x_eval=custom_x_eval,
        )

        assert len(xi) == 5
        assert len(yi) == 5
        assert mock_model.predict.called


class TestCliMainFunctions:
    """Integration tests for CLI main functions."""

    def test_cli_subcommands_registered(self):
        """Test that all expected subcommands are registered."""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli.cli, ["--help"])

        assert result.exit_code == 0
        assert "run-client" in result.output
        assert "run-server" in result.output
        assert "local" in result.output

    def test_run_client_command_exists(self):
        """Test that run-client command is accessible."""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli.cli, ["run-client", "--help"])

        assert result.exit_code == 0
        assert "--verbose" in result.output

    def test_run_server_command_exists(self):
        """Test that run-server command is accessible."""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli.cli, ["run-server", "--help"])

        assert result.exit_code == 0
        assert "COMMAND" in result.output
        assert "EXPERIMENT_FILE" in result.output

    def test_local_command_exists(self):
        """Test that local command is accessible."""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli.cli, ["local", "--help"])

        assert result.exit_code == 0
        assert "--tuning-config" in result.output
        assert "local" in result.output

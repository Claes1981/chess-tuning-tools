"""Tests for the CLI module."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tune.cli import ACQUISITION_FUNC, cli


@pytest.fixture
def cli_runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def mock_dbconfig(cli_runner):
    """Create a temporary dbconfig file."""
    with cli_runner.isolated_filesystem():
        dbconfig = {"host": "localhost", "user": "test", "password": "test"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dbconfig, f)
            yield f.name


@pytest.fixture
def mock_experiment_file(cli_runner):
    """Create a temporary experiment file."""
    with cli_runner.isolated_filesystem():
        experiment = {
            "name": "test",
            "engines": [],
            "parameters": [],
            "n_games": 100,
            "time_control": "10+0.1",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(experiment, f)
            yield f.name


@pytest.fixture
def mock_tuning_config(cli_runner):
    """Create a temporary tuning config file."""
    with cli_runner.isolated_filesystem():
        config = {
            "name": "test",
            "engines": [{"name": "engine1", "path": "/path/to/engine"}],
            "parameters": [],
            "n_games": 100,
            "time_control": "10+0.1",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            yield f.name


class TestCliGroup:
    """Test the CLI group."""

    def test_cli_help(self, cli_runner):
        """Test CLI help shows available commands."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run-client" in result.output
        assert "run-server" in result.output
        assert "local" in result.output

    def test_cli_no_command(self, cli_runner):
        """Test CLI with no command shows help and exits with error."""
        result = cli_runner.invoke(cli)
        # Click shows help and exits with code 2 when no command is provided
        assert result.exit_code == 2
        assert "Usage" in result.output


class TestRunClient:
    """Test the run_client command."""

    def test_run_client_basic(self, cli_runner, mock_dbconfig):
        """Test basic run_client invocation."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            _ = cli_runner.invoke(cli, ["run-client", mock_dbconfig])

            # Exit code may be non-zero if TuningClient.run() raises
            assert mock_client.called
            mock_client.assert_called_with(
                dbconfig_path=mock_dbconfig,
                terminate_after=0,
                clientconfig=None,
                only_run_once=False,
                skip_benchmark=False,
            )
            mock_instance.run.assert_called_once()

    def test_run_client_with_verbose(self, cli_runner, mock_dbconfig):
        """Test run_client with verbose flag."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            _ = cli_runner.invoke(
                cli, ["run-client", "--verbose", mock_dbconfig]
            )

            assert mock_client.called

    def test_run_client_with_logfile(self, cli_runner, mock_dbconfig):
        """Test run_client with logfile option."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=False
            ) as log_file:
                _ = cli_runner.invoke(
                    cli,
                    ["run-client", "--logfile", log_file.name, mock_dbconfig],
                )

            assert mock_client.called

    def test_run_client_with_terminate_after(self, cli_runner, mock_dbconfig):
        """Test run_client with terminate-after option."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            _ = cli_runner.invoke(
                cli, ["run-client", "--terminate-after", "10", mock_dbconfig]
            )

            assert mock_client.called
            mock_client.assert_called_with(
                dbconfig_path=mock_dbconfig,
                terminate_after=10,
                clientconfig=None,
                only_run_once=False,
                skip_benchmark=False,
            )

    def test_run_client_with_run_only_once(self, cli_runner, mock_dbconfig):
        """Test run_client with run-only-once flag."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            _ = cli_runner.invoke(
                cli, ["run-client", "--run-only-once", mock_dbconfig]
            )

            assert mock_client.called
            mock_client.assert_called_with(
                dbconfig_path=mock_dbconfig,
                terminate_after=0,
                clientconfig=None,
                only_run_once=True,
                skip_benchmark=False,
            )

    def test_run_client_with_skip_benchmark(self, cli_runner, mock_dbconfig):
        """Test run_client with skip-benchmark flag."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            _ = cli_runner.invoke(
                cli, ["run-client", "--skip-benchmark", mock_dbconfig]
            )

            assert mock_client.called
            mock_client.assert_called_with(
                dbconfig_path=mock_dbconfig,
                terminate_after=0,
                clientconfig=None,
                only_run_once=False,
                skip_benchmark=True,
            )

    def test_run_client_with_clientconfig(self, cli_runner, mock_dbconfig):
        """Test run_client with clientconfig option."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False
            ) as cc:
                _ = cli_runner.invoke(
                    cli,
                    ["run-client", "--clientconfig", cc.name, mock_dbconfig],
                )

            assert mock_client.called

    def test_run_client_all_options(self, cli_runner, mock_dbconfig):
        """Test run_client with all options."""
        with patch("tune.cli.TuningClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=False
            ) as log_file:
                with tempfile.NamedTemporaryFile(
                    suffix=".json", delete=False
                ) as clientconfig:
                    _ = cli_runner.invoke(
                        cli,
                        [
                            "run-client",
                            "--verbose",
                            "--logfile",
                            log_file.name,
                            "--terminate-after",
                            "5",
                            "--run-only-once",
                            "--skip-benchmark",
                            "--clientconfig",
                            clientconfig.name,
                            mock_dbconfig,
                        ],
                    )

            assert mock_client.called

    def test_run_client_missing_dbconfig(self, cli_runner):
        """Test run_client without required dbconfig argument."""
        result = cli_runner.invoke(cli, ["run-client"])
        assert result.exit_code != 0
        assert (
            "Missing argument" in result.output or "DBCONFIG" in result.output
        )


class TestRunServer:
    """Test the run_server command."""

    def test_run_server_run_command(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with 'run' command."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            _ = cli_runner.invoke(
                cli,
                ["run-server", "run", mock_experiment_file, mock_dbconfig],
            )

            assert mock_server.called
            mock_instance.run.assert_called_once()

    def test_run_server_deactivate_command(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with 'deactivate' command."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            _ = cli_runner.invoke(
                cli,
                [
                    "run-server",
                    "deactivate",
                    mock_experiment_file,
                    mock_dbconfig,
                ],
            )

            assert mock_server.called
            mock_instance.deactivate.assert_called_once()

    def test_run_server_reactivate_command(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with 'reactivate' command."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            _ = cli_runner.invoke(
                cli,
                [
                    "run-server",
                    "reactivate",
                    mock_experiment_file,
                    mock_dbconfig,
                ],
            )

            assert mock_server.called
            mock_instance.reactivate.assert_called_once()

    def test_run_server_invalid_command(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with invalid command raises ValueError."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            result = cli_runner.invoke(
                cli,
                ["run-server", "invalid", mock_experiment_file, mock_dbconfig],
            )

            # The ValueError is raised and caught by Click, stored in result.exception
            assert result.exit_code != 0
            assert result.exception is not None
            assert isinstance(result.exception, ValueError)
            assert "not recognized" in str(result.exception)

    def test_run_server_with_verbose(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with verbose flag."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            _ = cli_runner.invoke(
                cli,
                [
                    "run-server",
                    "--verbose",
                    "run",
                    mock_experiment_file,
                    mock_dbconfig,
                ],
            )

            assert mock_server.called

    def test_run_server_with_logfile(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with logfile option."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=False
            ) as log_file:
                _ = cli_runner.invoke(
                    cli,
                    [
                        "run-server",
                        "--logfile",
                        log_file.name,
                        "run",
                        mock_experiment_file,
                        mock_dbconfig,
                    ],
                )

            assert mock_server.called

    def test_run_server_all_options(
        self, cli_runner, mock_dbconfig, mock_experiment_file
    ):
        """Test run_server with all options."""
        with patch("tune.cli.TuningServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance

            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=False
            ) as log_file:
                _ = cli_runner.invoke(
                    cli,
                    [
                        "run-server",
                        "--verbose",
                        "--logfile",
                        log_file.name,
                        "run",
                        mock_experiment_file,
                        mock_dbconfig,
                    ],
                )

            assert mock_server.called

    def test_run_server_missing_command(self, cli_runner, mock_dbconfig):
        """Test run_server without required command argument."""
        result = cli_runner.invoke(cli, ["run-server"])
        assert result.exit_code != 0

    def test_run_server_missing_experiment_file(
        self, cli_runner, mock_dbconfig
    ):
        """Test run_server without required experiment_file argument."""
        result = cli_runner.invoke(cli, ["run-server", "run"])
        assert result.exit_code != 0

    def test_run_server_missing_dbconfig(
        self, cli_runner, mock_experiment_file
    ):
        """Test run_server without required dbconfig argument."""
        result = cli_runner.invoke(
            cli, ["run-server", "run", mock_experiment_file]
        )
        assert result.exit_code != 0


class TestLocal:
    """Test the local command."""

    def test_local_help(self, cli_runner):
        """Test local command help."""
        result = cli_runner.invoke(cli, ["local", "--help"])
        assert result.exit_code == 0
        assert "tuning-config" in result.output
        assert "acq-function" in result.output

    def test_local_missing_tuning_config(self, cli_runner):
        """Test local without required tuning-config option."""
        result = cli_runner.invoke(cli, ["local"])
        assert result.exit_code != 0
        assert "Missing option" in result.output
        assert "tuning-config" in result.output

    def test_local_with_basic_options(self, cli_runner, mock_tuning_config):
        """Test local with basic options."""
        with patch("tune.cli.json.load") as mock_json_load:
            with patch("tune.cli.load_tuning_config") as mock_load_config:
                with patch("tune.cli.setup_logger"):
                    with patch("tune.cli.initialize_data") as mock_init_data:
                        with patch(
                            "tune.cli.initialize_optimizer"
                        ) as mock_init_optimizer:
                            with patch(
                                "tune.cli.load_points_to_evaluate"
                            ) as mock_load_points:
                                with patch(
                                    "tune.cli.check_if_pause"
                                ) as mock_check_pause:
                                    mock_check_pause.return_value = (
                                        True  # Exit loop immediately
                                    )

                                    mock_json_load.return_value = {}
                                    mock_load_config.return_value = (
                                        {},
                                        [],
                                        [],
                                        {},
                                        {},
                                        [],
                                    )
                                    mock_init_data.return_value = ({}, {}, {})
                                    mock_init_optimizer.return_value = (
                                        MagicMock()
                                    )
                                    mock_load_points.return_value = []

                                    _ = cli_runner.invoke(
                                        cli,
                                        [
                                            "local",
                                            "-c",
                                            mock_tuning_config,
                                        ],
                                    )

                                    assert mock_json_load.called
                                    assert mock_load_config.called

    def test_local_with_acq_function(self, cli_runner, mock_tuning_config):
        """Test local with different acquisition functions."""
        with patch("tune.cli.json.load") as mock_json_load:
            with patch("tune.cli.load_tuning_config") as mock_load_config:
                with patch("tune.cli.setup_logger"):
                    with patch("tune.cli.initialize_data") as mock_init_data:
                        with patch(
                            "tune.cli.initialize_optimizer"
                        ) as mock_init_optimizer:
                            with patch(
                                "tune.cli.load_points_to_evaluate"
                            ) as mock_load_points:
                                with patch(
                                    "tune.cli.check_if_pause"
                                ) as mock_check_pause:
                                    mock_check_pause.return_value = True

                                    mock_json_load.return_value = {}
                                    mock_load_config.return_value = (
                                        {},
                                        [],
                                        [],
                                        {},
                                        {},
                                        [],
                                    )
                                    mock_init_data.return_value = ({}, {}, {})
                                    mock_init_optimizer.return_value = (
                                        MagicMock()
                                    )
                                    mock_load_points.return_value = []

                                    for acq_func in [
                                        "ei",
                                        "lcb",
                                        "mean",
                                        "mes",
                                        "pvrs",
                                        "ts",
                                        "ttei",
                                        "vr",
                                    ]:
                                        _ = cli_runner.invoke(
                                            cli,
                                            [
                                                "local",
                                                "-c",
                                                mock_tuning_config,
                                                "-a",
                                                acq_func,
                                            ],
                                        )

                                        assert mock_json_load.called
                                        assert mock_load_config.called

    def test_local_with_n_points(self, cli_runner, mock_tuning_config):
        """Test local with n-points option."""
        with patch("tune.cli.json.load") as mock_json_load:
            with patch("tune.cli.load_tuning_config") as mock_load_config:
                with patch("tune.cli.setup_logger"):
                    with patch("tune.cli.initialize_data") as mock_init_data:
                        with patch(
                            "tune.cli.initialize_optimizer"
                        ) as mock_init_optimizer:
                            with patch(
                                "tune.cli.load_points_to_evaluate"
                            ) as mock_load_points:
                                with patch(
                                    "tune.cli.check_if_pause"
                                ) as mock_check_pause:
                                    mock_check_pause.return_value = True

                                    mock_json_load.return_value = {}
                                    mock_load_config.return_value = (
                                        {},
                                        [],
                                        [],
                                        {},
                                        {},
                                        [],
                                    )
                                    mock_init_data.return_value = ({}, {}, {})
                                    mock_init_optimizer.return_value = (
                                        MagicMock()
                                    )
                                    mock_load_points.return_value = []

                                    _ = cli_runner.invoke(
                                        cli,
                                        [
                                            "local",
                                            "-c",
                                            mock_tuning_config,
                                            "--n-points",
                                            "1000",
                                        ],
                                    )

                                    assert mock_json_load.called
                                    assert mock_load_config.called


class TestAcquisitionFunc:
    """Test the ACQUISITION_FUNC dictionary."""

    def test_acquisition_func_keys(self):
        """Test that ACQUISITION_FUNC has expected keys."""
        expected_keys = {"ei", "lcb", "mean", "mes", "pvrs", "ts", "ttei", "vr"}
        assert set(ACQUISITION_FUNC.keys()) == expected_keys

    def test_acquisition_func_values(self):
        """Test that ACQUISITION_FUNC values are acquisition objects."""
        from bask import acquisition

        for value in ACQUISITION_FUNC.values():
            assert isinstance(value, acquisition.Acquisition)

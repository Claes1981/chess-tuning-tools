"""Tests for data parsing and log checking in local.py.

Tests parse_experiment_result, check_log_for_errors, is_debug_log, and print_results functions.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.optimize import OptimizeResult

from tune.local import (
    check_log_for_errors,
    is_debug_log,
    parse_experiment_result,
    print_results,
)


class TestParseExperimentResult:
    """Tests for the parse_experiment_result function."""

    def test_basic_match(self):
        """Test parsing basic match results with wins and losses."""
        cutechess_output = """Indexing opening suite...
Started game 1 of 100 (lc0 vs sf)
Finished game 1 (lc0 vs sf): 0-1 {Black mates}
Score of lc0 vs sf: 0 - 1 - 0  [0.000] 1
Started game 2 of 100 (sf vs lc0)
Finished game 2 (sf vs lc0): 0-1 {Black mates}
Score of lc0 vs sf: 1 - 1 - 0  [0.500] 2
Elo difference: -31.4 +/- 57.1, LOS: 13.9 %, DrawRatio: 31.0 %
Finished match
"""
        score, variance, draw_rate, _ = parse_experiment_result(
            cutechess_output, n_dirichlet_samples=1000, random_state=0
        )
        assert_almost_equal(score, 0.0)
        assert_almost_equal(variance, 0.887797821633887)
        assert_almost_equal(draw_rate, 1 / 4)

    def test_cutechess_1_2_0(self):
        """Test parsing cutechess 1.2.0 output format."""
        cutechess_output = """Started game 1 of 4 (engine1 vs engine2)
    Finished game 1 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 0 - 1 - 0  [0.000] 1
    Started game 2 of 4 (engine2 vs engine1)
    Finished game 2 (engine2 vs engine1): 1/2-1/2 {Draw by stalemate}
    Score of engine1 vs engine2: 0 - 1 - 1  [0.250] 2
    Started game 3 of 4 (engine1 vs engine2)
    Finished game 3 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 0 - 2 - 1  [0.167] 3
    Started game 4 of 4 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 2 - 1  [0.375] 4
    ...      engine1 playing White: 0 - 2 - 0  [0.000] 2
    ...      engine1 playing Black: 1 - 0 - 1  [0.750] 2
    ...      White vs Black: 0 - 3 - 1  [0.125] 4
    Elo difference: -88.7 +/- nan, LOS: 28.2 %, DrawRatio: 25.0 %
    Finished match
    """
        score, variance, draw_rate, counts_array = parse_experiment_result(
            cutechess_output, n_dirichlet_samples=1000, random_state=0
        )
        assert_almost_equal(score, 0.38764005203222596)
        assert_almost_equal(variance, 0.6255020676255081)
        assert_almost_equal(draw_rate, 1.5 / 5)

    def test_with_adjudications(self):
        """Test parsing results with various adjudication types."""
        cutechess_output = """Indexing opening suite...
    Started game 1 of 40 (engine1 vs engine2)
    Finished game 1 (engine1 vs engine2): 1/2-1/2 {Draw by 3-fold repetition}
    Score of engine1 vs engine2: 1 - 0 - 0  [0.500] 1
    Started game 2 of 40 (engine2 vs engine1)
    Finished game 2 (engine2 vs engine1): 1/2-1/2 {Draw by adjudication: SyzygyTB}
    Score of engine1 vs engine2: 2 - 0 - 0  [0.500] 2
    Started game 3 of 40 (engine1 vs engine2)
    Finished game 3 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 3 - 0 - 0  [0.333] 3
    Started game 4 of 40 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 4 - 0 - 0  [0.500] 4
    Started game 5 of 40 (engine1 vs engine2)
    Finished game 5 (engine1 vs engine2): 1-0 {White wins by adjudication}
    Score of engine1 vs engine2: 5 - 0 - 0  [0.600] 5
    Started game 6 of 40 (engine2 vs engine1)
    Finished game 6 (engine2 vs engine1): 1-0 {White wins by adjudication}
    Score of engine1 vs engine2: 6 - 0 - 0  [0.500] 6
    Started game 7 of 40 (engine1 vs engine2)
    Finished game 7 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 7 - 0 - 0  [0.429] 7
    Started game 8 of 40 (engine2 vs engine1)
    Finished game 8 (engine2 vs engine1): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 8 - 0 - 0  [0.500] 8
    Started game 9 of 40 (engine1 vs engine2)
    Finished game 9 (engine1 vs engine2): 0-1 {Black wins by adjudication}
    Score of engine1 vs engine2: 9 - 0 - 0  [0.444] 9
    Started game 10 of 40 (engine2 vs engine1)
    Finished game 10 (engine2 vs engine1): 1/2-1/2 {Draw by adjudication}
    Score of engine1 vs engine2: 10 - 0 - 0  [0.450] 10
    """
        score, variance, draw_rate, counts_array = parse_experiment_result(
            cutechess_output, n_dirichlet_samples=1000, random_state=0
        )
        assert_almost_equal(score, -2.7958800173440745)
        assert_almost_equal(variance, 1.9952678343378125)
        assert_almost_equal(draw_rate, 1 / 8)

    def test_non_linear_game_order(self):
        """Test parsing when games finish out of order (concurrent games)."""
        cutechess_output = """Started game 1 of 4 (engine1 vs engine2)
    Started game 2 of 4 (engine2 vs engine1)
    Started game 3 of 4 (engine1 vs engine2)
    Started game 4 of 4 (engine2 vs engine1)
    Finished game 4 (engine2 vs engine1): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 0 - 0  [0.375] 1
    Finished game 1 (engine1 vs engine2): 1/2-1/2 {Draw by stalemate}
    Score of engine1 vs engine2: 1 - 0 - 1  [0.000] 2
    Finished game 2 (engine2 vs engine1): 1-0 {White mates}
    Score of engine1 vs engine2: 1 - 1 - 1  [0.250] 3
    Finished game 3 (engine1 vs engine2): 0-1 {Black mates}
    Score of engine1 vs engine2: 1 - 2 - 1  [0.167] 4
    ...      engine1 playing White: 0 - 2 - 0  [0.000] 2
    ...      engine1 playing Black: 1 - 0 - 1  [0.750] 2
    ...      White vs Black: 0 - 3 - 1  [0.125] 4
    Elo difference: -88.7 +/- nan, LOS: 28.2 %, DrawRatio: 25.0 %
    Finished match
    """
        score, variance, draw_rate, counts_array = parse_experiment_result(
            cutechess_output, n_dirichlet_samples=1000, random_state=0
        )
        assert_almost_equal(score, 0.38764005203222596)
        assert_almost_equal(variance, 0.6255020676255081)
        assert_almost_equal(draw_rate, 1.5 / 5)


class TestIsDebugLog:
    """Tests for the is_debug_log function."""

    @pytest.mark.parametrize(
        "log_line,expected",
        [
            (
                "3287 <engine1(0): info depth 1 seldepth 1 time 2 nodes 1 score cp -8502 "
                "tbhits 0 pv a7a8n",
                True,
            ),
            ("2018 >lc0(1): ucinewgame", True),
            (
                "Finished game 1 (engine1 vs engine2): 0-1 {White loses on time}",
                False,
            ),
            ("Finished game 3287 <engine1(0): ", False),
            ("...      White vs Black: 0 - 1 - 0  [0.000] 1", False),
        ],
    )
    def test_is_debug_log(self, log_line, expected):
        assert is_debug_log(log_line) == expected


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger propagate setting before each test."""
    logger = logging.getLogger("ChessTuner")
    original_propagate = logger.propagate
    yield
    logger.propagate = original_propagate


class TestCheckLogForErrors:
    """Tests for the check_log_for_errors function."""

    def test_detects_time_loss(self, caplog):
        """Test that time loss errors are detected and logged correctly."""
        logger = logging.getLogger("ChessTuner")
        logger.propagate = True
        caplog.set_level(logging.WARNING, logger="ChessTuner")
        cutechess_output = """Started game 1 of 2 (SFDev vs Lc0.11198)
Finished game 1 (SFDev vs Lc0.11198): 1-0 {Black loses on time}
Score of SFDev vs Lc0.11198: 1 - 0 - 0  [1.000] 1
Started game 2 of 2 (Lc0.11198 vs SFDev)
Finished game 2 (Lc0.11198 vs SFDev): 0-1 {White loses on time}
Score of SFDev vs Lc0.11198: 2 - 0 - 0  [1.000] 2
Elo difference: inf +/- nan
Finished match""".split("\n")

        check_log_for_errors(list(cutechess_output))
        assert "Engine Lc0.11198 lost on time as Black." in caplog.text

    def test_detects_connection_stalls(self, caplog):
        """Test that connection stall errors are detected and logged correctly."""
        logger = logging.getLogger("ChessTuner")
        logger.propagate = True
        caplog.set_level(logging.ERROR, logger="ChessTuner")
        cutechess_output = """Terminating process of engine lc0(0)
16994 >lc0(1): isready
16994 <lc0(1): readyok
...      lc0 playing White: 0 - 1 - 0  [0.000] 1
...      White vs Black: 0 - 1 - 0  [0.000] 1
Elo difference: -inf +/- nan, LOS: 15.9 %, DrawRatio: 0.0 %
Finished match
Finished game 1 (lc0 vs lc0): 0-1 {White's connection stalls}
Score of lc0 vs lc0: 0 - 1 - 0  [0.000] 1
...      lc0 playing White: 0 - 1 - 0  [0.000] 1
...      White vs Black: 0 - 1 - 0  [0.000] 1
Elo difference: -inf +/- nan, LOS: 15.9 %, DrawRatio: 0.0 %
Finished match
16995 >lc0(1): quit""".split("\n")

        check_log_for_errors(list(cutechess_output))
        assert (
            "lc0's connection stalled as White. Game result is unreliable."
            in caplog.text
        )

    def test_forward_cutechess_cli_errors(self, caplog):
        """Test that cutechess-cli errors are forwarded to logs."""
        logger = logging.getLogger("ChessTuner")
        logger.propagate = True
        caplog.set_level(logging.WARNING, logger="ChessTuner")
        cutechess_output = [
            "797 <lc0(0): error The cuda backend requires a network file."
        ]

        check_log_for_errors(cutechess_output)
        assert (
            "cutechess-cli error: The cuda backend requires a network file."
            in caplog.text
        )

    def test_unknown_uci_options(self, caplog):
        """Test that unknown UCI option errors are detected and logged."""
        logger = logging.getLogger("ChessTuner")
        logger.propagate = True
        caplog.set_level(logging.ERROR, logger="ChessTuner")
        cutechess_output = ["424 <lc0(1): error Unknown option: UnknownOption"]

        check_log_for_errors(cutechess_output)
        assert (
            "UCI option UnknownOption was unknown to the engine. Check if the spelling "
            "is correct." in caplog.text
        )


class TestPrintResults:
    """Tests for the print_results function."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer object."""
        optimizer = MagicMock()
        optimizer.gp.chain_ = np.array([[0.1], [0.2], [0.3]])
        optimizer.gp.noise_set_to_zero.return_value.__enter__ = lambda self: (
            optimizer.gp
        )
        optimizer.gp.noise_set_to_zero.return_value.__exit__ = (
            lambda self, *args: None
        )
        optimizer.gp.predict.return_value = (
            np.array([-0.05]),
            np.array([0.01]),
        )
        optimizer.space.transform = lambda x: x
        return optimizer

    @pytest.fixture
    def mock_result_object(self):
        """Create a mock result object."""
        result = MagicMock(spec=OptimizeResult)
        result.x = np.array([0.5])
        result.fun = -0.05
        return result

    @pytest.fixture
    def parameter_names(self):
        """Create parameter names."""
        return ["param1"]

    def test_basic(self, mock_optimizer, mock_result_object, parameter_names):
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
                mock_optimizer,
                mock_result_object,
                parameter_names,
                confidence=0.9,
            )

            assert isinstance(best_point, np.ndarray)
            assert isinstance(elo, float)
            assert isinstance(std, float)
            assert elo == 5.0
            assert std == 1.0

    def test_confidence_interval(
        self, mock_optimizer, mock_result_object, parameter_names
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
                mock_optimizer,
                mock_result_object,
                parameter_names,
                confidence=0.95,
            )

            assert elo == 10.0
            assert std == 1.0

    def test_value_error(
        self, mock_optimizer, mock_result_object, parameter_names
    ):
        """Test that ValueError is raised when expected_ucb fails."""
        with patch("tune.local.expected_ucb") as mock_ucb:
            mock_ucb.side_effect = ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                print_results(
                    mock_optimizer, mock_result_object, parameter_names
                )

    def test_logging(self, mock_optimizer, mock_result_object, parameter_names):
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

            assert mock_logger.info.call_count >= 3

    def test_multiple_parameters(self, mock_optimizer, mock_result_object):
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

    def test_high_confidence(
        self, mock_optimizer, mock_result_object, parameter_names
    ):
        """Test print_results with high confidence level."""
        with (
            patch("tune.local.expected_ucb") as mock_ucb,
            patch("tune.local.confidence_to_mult") as mock_conf,
            patch("tune.local.confidence_intervals") as mock_conf_int,
        ):
            mock_ucb.return_value = (np.array([0.5]), -0.05)
            mock_conf.return_value = 3.0
            mock_conf_int.return_value = "param1: 0.5 [0.3, 0.7]"

            best_point, elo, std = print_results(
                mock_optimizer,
                mock_result_object,
                parameter_names,
                confidence=0.999,
            )

            assert elo == 5.0
            mock_conf.assert_called_with(0.999)

    def test_zero_std(
        self, mock_optimizer, mock_result_object, parameter_names
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

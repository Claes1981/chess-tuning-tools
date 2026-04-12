"""Tests for engine configuration functions in local.py.

Tests _construct_engine_conf function with various time controls and options.
"""

import pytest

from tune.local import _construct_engine_conf
from tune.utils import TimeControl


class TestConstructEngineConf:
    """Tests for the _construct_engine_conf function."""

    # Time control tests
    def test_npm_int(self):
        """Test with nodes per minute as int."""
        result = _construct_engine_conf(id=1, engine_npm=1000)

        assert result == [
            "-engine",
            "conf=engine1",
            "restart=on",
            "tc=inf",
            "nodes=1000",
        ]

    def test_npm_str(self):
        """Test with nodes per minute as string."""
        result = _construct_engine_conf(id=1, engine_npm="2000")

        assert result == [
            "-engine",
            "conf=engine1",
            "restart=on",
            "tc=inf",
            "nodes=2000",
        ]

    def test_st_int(self):
        """Test with seconds as int."""
        result = _construct_engine_conf(id=2, engine_st=60)

        assert result == ["-engine", "conf=engine2", "restart=on", "st=60"]

    def test_st_str(self):
        """Test with seconds as string."""
        result = _construct_engine_conf(id=2, engine_st="120")

        assert result == ["-engine", "conf=engine2", "restart=on", "st=120"]

    def test_depth_int(self):
        """Test with depth as int."""
        result = _construct_engine_conf(id=3, engine_depth=20)

        assert result == [
            "-engine",
            "conf=engine3",
            "restart=on",
            "tc=inf",
            "depth=20",
        ]

    def test_depth_str(self):
        """Test with depth as string."""
        result = _construct_engine_conf(id=3, engine_depth="15")

        assert result == [
            "-engine",
            "conf=engine3",
            "restart=on",
            "tc=inf",
            "depth=15",
        ]

    def test_tc_string(self):
        """Test with time control as string."""
        result = _construct_engine_conf(id=4, engine_tc="10+0.5")

        assert result == ["-engine", "conf=engine4", "restart=on", "tc=10+0.5"]

    def test_tc_timecontrol(self):
        """Test with TimeControl object."""
        tc = TimeControl(time=60, increment=1)
        result = _construct_engine_conf(id=5, engine_tc=tc)

        assert result == ["-engine", "conf=engine5", "restart=on", "tc=60+1"]

    def test_tc_with_zero_increment(self):
        """Test with time control with zero increment."""
        result = _construct_engine_conf(id=1, engine_tc="10+0")

        assert "tc=10" in result

    # Priority tests
    def test_priority_npm_over_tc(self):
        """Test that npm takes priority over tc."""
        result = _construct_engine_conf(
            id=1, engine_npm=1000, engine_tc="10+0.5"
        )

        assert "nodes=1000" in result
        assert "tc=10+0.5" not in result

    def test_priority_st_over_depth(self):
        """Test that st takes priority over depth."""
        result = _construct_engine_conf(id=1, engine_st=60, engine_depth=20)

        assert "st=60" in result
        assert "depth=20" not in result

    def test_priority_depth_over_tc(self):
        """Test that depth takes priority over tc."""
        result = _construct_engine_conf(
            id=1, engine_depth=20, engine_tc="10+0.5"
        )

        assert "depth=20" in result
        assert "tc=10+0.5" not in result

    # Optional parameters tests
    def test_with_timemargin(self):
        """Test with timemargin."""
        result = _construct_engine_conf(
            id=1, engine_tc="10+0.5", timemargin=500
        )

        assert "timemargin=500" in result

    def test_with_tscale(self):
        """Test with timeout scale factor."""
        result = _construct_engine_conf(
            id=1, engine_tc="10+0.5", engine_timeout_scale_factor=1.5
        )

        assert "tscale=1.5" in result

    def test_with_tscale_str(self):
        """Test with timeout scale factor as string."""
        result = _construct_engine_conf(
            id=1, engine_tc="10+0.5", engine_timeout_scale_factor="2.0"
        )

        assert "tscale=2.0" in result

    def test_with_ponder(self):
        """Test with ponder enabled."""
        result = _construct_engine_conf(
            id=1, engine_tc="10+0.5", engine_ponder=True
        )

        assert "ponder" in result

    def test_restart_off(self):
        """Test with restart=off."""
        result = _construct_engine_conf(
            id=1, engine_tc="10+0.5", engine_restart="off"
        )

        assert "restart=off" in result

    # Edge cases
    def test_no_tc_raises_error(self):
        """Test that ValueError is raised when no time control is specified."""
        with pytest.raises(
            ValueError, match="No engine time control specified for engine 1"
        ):
            _construct_engine_conf(id=1)

    def test_high_id(self):
        """Test with high engine id."""
        result = _construct_engine_conf(id=99, engine_tc="10+0.5")

        assert "conf=engine99" in result

    def test_all_options(self):
        """Test with all options."""
        result = _construct_engine_conf(
            id=1,
            engine_tc="10+0.5",
            engine_ponder=True,
            engine_restart="on",
            engine_timeout_scale_factor=1.2,
            timemargin=1000,
        )

        assert "-engine" in result
        assert "conf=engine1" in result
        assert "restart=on" in result
        assert "timemargin=1000" in result
        assert "tscale=1.2" in result
        assert "ponder" in result
        assert "tc=10+0.5" in result

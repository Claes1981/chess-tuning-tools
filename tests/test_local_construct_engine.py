"""Tests for _construct_engine_conf function in local.py."""

import pytest

from tune.local import _construct_engine_conf
from tune.utils import TimeControl


def test_construct_engine_conf_npm_int():
    """Test _construct_engine_conf with nodes per minute as int."""
    result = _construct_engine_conf(id=1, engine_npm=1000)

    assert result == [
        "-engine",
        "conf=engine1",
        "restart=on",
        "tc=inf",
        "nodes=1000",
    ]


def test_construct_engine_conf_npm_str():
    """Test _construct_engine_conf with nodes per minute as string."""
    result = _construct_engine_conf(id=1, engine_npm="2000")

    assert result == [
        "-engine",
        "conf=engine1",
        "restart=on",
        "tc=inf",
        "nodes=2000",
    ]


def test_construct_engine_conf_st_int():
    """Test _construct_engine_conf with seconds as int."""
    result = _construct_engine_conf(id=2, engine_st=60)

    assert result == ["-engine", "conf=engine2", "restart=on", "st=60"]


def test_construct_engine_conf_st_str():
    """Test _construct_engine_conf with seconds as string."""
    result = _construct_engine_conf(id=2, engine_st="120")

    assert result == ["-engine", "conf=engine2", "restart=on", "st=120"]


def test_construct_engine_conf_depth_int():
    """Test _construct_engine_conf with depth as int."""
    result = _construct_engine_conf(id=3, engine_depth=20)

    assert result == [
        "-engine",
        "conf=engine3",
        "restart=on",
        "tc=inf",
        "depth=20",
    ]


def test_construct_engine_conf_depth_str():
    """Test _construct_engine_conf with depth as string."""
    result = _construct_engine_conf(id=3, engine_depth="15")

    assert result == [
        "-engine",
        "conf=engine3",
        "restart=on",
        "tc=inf",
        "depth=15",
    ]


def test_construct_engine_conf_tc_string():
    """Test _construct_engine_conf with time control as string."""
    result = _construct_engine_conf(id=4, engine_tc="10+0.5")

    assert result == ["-engine", "conf=engine4", "restart=on", "tc=10+0.5"]


def test_construct_engine_conf_tc_timecontrol():
    """Test _construct_engine_conf with TimeControl object."""
    tc = TimeControl(time=60, increment=1)
    result = _construct_engine_conf(id=5, engine_tc=tc)

    assert result == ["-engine", "conf=engine5", "restart=on", "tc=60+1"]


def test_construct_engine_conf_with_timemargin():
    """Test _construct_engine_conf with timemargin."""
    result = _construct_engine_conf(id=1, engine_tc="10+0.5", timemargin=500)

    assert "timemargin=500" in result


def test_construct_engine_conf_with_tscale():
    """Test _construct_engine_conf with timeout scale factor."""
    result = _construct_engine_conf(
        id=1, engine_tc="10+0.5", engine_timeout_scale_factor=1.5
    )

    assert "tscale=1.5" in result


def test_construct_engine_conf_with_tscale_str():
    """Test _construct_engine_conf with timeout scale factor as string."""
    result = _construct_engine_conf(
        id=1, engine_tc="10+0.5", engine_timeout_scale_factor="2.0"
    )

    assert "tscale=2.0" in result


def test_construct_engine_conf_with_ponder():
    """Test _construct_engine_conf with ponder enabled."""
    result = _construct_engine_conf(
        id=1, engine_tc="10+0.5", engine_ponder=True
    )

    assert "ponder" in result


def test_construct_engine_conf_restart_off():
    """Test _construct_engine_conf with restart=off."""
    result = _construct_engine_conf(
        id=1, engine_tc="10+0.5", engine_restart="off"
    )

    assert "restart=off" in result


def test_construct_engine_conf_no_tc_raises_error():
    """Test _construct_engine_conf raises ValueError when no time control is specified."""
    with pytest.raises(
        ValueError, match="No engine time control specified for engine 1"
    ):
        _construct_engine_conf(id=1)


def test_construct_engine_conf_priority_npm_over_tc():
    """Test that npm takes priority over tc."""
    result = _construct_engine_conf(id=1, engine_npm=1000, engine_tc="10+0.5")

    assert "nodes=1000" in result
    assert "tc=10+0.5" not in result


def test_construct_engine_conf_priority_st_over_depth():
    """Test that st takes priority over depth."""
    result = _construct_engine_conf(id=1, engine_st=60, engine_depth=20)

    assert "st=60" in result
    assert "depth=20" not in result


def test_construct_engine_conf_priority_depth_over_tc():
    """Test that depth takes priority over tc."""
    result = _construct_engine_conf(id=1, engine_depth=20, engine_tc="10+0.5")

    assert "depth=20" in result
    assert "tc=10+0.5" not in result


def test_construct_engine_conf_all_options():
    """Test _construct_engine_conf with all options."""
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


def test_construct_engine_conf_high_id():
    """Test _construct_engine_conf with high engine id."""
    result = _construct_engine_conf(id=99, engine_tc="10+0.5")

    assert "conf=engine99" in result


def test_construct_engine_conf_tc_with_zero_increment():
    """Test _construct_engine_conf with time control with zero increment."""
    result = _construct_engine_conf(id=1, engine_tc="10+0")

    # TimeControl.__str__ omits +0 when increment is 0
    assert "tc=10" in result

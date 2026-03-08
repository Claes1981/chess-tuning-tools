"""Tests for write_polyglot_ini function in io.py."""

import configparser
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tune.io import write_polyglot_ini


@pytest.fixture(autouse=True)
def cleanup_polyglot_config():
    """Clean up polyglot config directory after each test."""
    yield
    if Path("polyglot-config").exists():
        import shutil

        shutil.rmtree("polyglot-config")


def test_write_polyglot_ini_basic():
    """Test basic write_polyglot_ini functionality."""
    polyglot_params = [
        {"engine_command": "engine1", "book_file": ""},
        {
            "engine_command": "engine2",
            "book_file": "book.bin",
            "max_book_depth": 10,
        },
    ]

    write_polyglot_ini(polyglot_params)

    # Check that config file was created
    config_path = Path("polyglot-config/polyglot.ini")
    assert config_path.exists()

    # Read and verify config
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    assert "PolyGlot" in config
    assert "Engine" in config
    assert config["PolyGlot"]["EngineCommand"] == "engine2"
    assert config["PolyGlot"]["UCI"] == "true"
    assert config["PolyGlot"]["Book"] == "true"
    assert config["PolyGlot"]["BookFile"] == "book.bin"
    assert config["PolyGlot"]["BookTreshold"] == "0"
    # BookDepth should be between 1 and max_book_depth (10)
    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert 1 <= book_depth <= 10


def test_write_polyglot_ini_default_max_book_depth():
    """Test write_polyglot_ini with default max_book_depth."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "book_file": "book.bin"},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    # Default max_book_depth is 256
    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert 1 <= book_depth <= 256


def test_write_polyglot_ini_min_book_depth():
    """Test that BookDepth is at least 1."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "max_book_depth": 0},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert book_depth >= 1


def test_write_polyglot_ini_empty_engine_command():
    """Test write_polyglot_ini with empty engine_command."""
    polyglot_params = [
        {},
        {"book_file": "book.bin"},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    assert config["PolyGlot"]["EngineCommand"] == ""


def test_write_polyglot_ini_empty_book_file():
    """Test write_polyglot_ini with empty book_file."""
    polyglot_params = [
        {},
        {"engine_command": "engine2"},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    assert config["PolyGlot"]["BookFile"] == ""


def test_write_polyglot_ini_creates_directory():
    """Test that write_polyglot_ini creates the directory if it doesn't exist."""
    # Ensure directory doesn't exist
    if Path("polyglot-config").exists():
        import shutil

        shutil.rmtree("polyglot-config")

    polyglot_params = [
        {},
        {"engine_command": "engine2"},
    ]

    write_polyglot_ini(polyglot_params)

    assert Path("polyglot-config").exists()
    assert Path("polyglot-config").is_dir()


def test_write_polyglot_ini_case_sensitivity():
    """Test that config is case-sensitive."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "BookFile": "book.bin"},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    # Check that keys are preserved as written
    assert "BookFile" in config["PolyGlot"]


def test_write_polyglot_ini_high_max_book_depth():
    """Test write_polyglot_ini with high max_book_depth."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "max_book_depth": 1000},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert 1 <= book_depth <= 1000


def test_write_polyglot_ini_float_max_book_depth():
    """Test write_polyglot_ini with float max_book_depth (should be converted to int)."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "max_book_depth": 10.9},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert 1 <= book_depth <= 10  # Should be truncated to 10


def test_write_polyglot_ini_negative_max_book_depth():
    """Test write_polyglot_ini with negative max_book_depth (should default to 1)."""
    polyglot_params = [
        {},
        {"engine_command": "engine2", "max_book_depth": -5},
    ]

    write_polyglot_ini(polyglot_params)

    config_path = Path("polyglot-config/polyglot.ini")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)

    book_depth = int(config["PolyGlot"]["BookDepth"])
    assert book_depth == 1  # max(1, -5) = 1

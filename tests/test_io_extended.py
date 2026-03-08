"""Extended tests for io.py functions."""

import json
import tempfile
from pathlib import Path

import pytest
from skopt.space import Categorical, Integer, Real

from tune.io import (
    InitStrings,
    parse_ranges,
    prepare_engines_json,
    uci_tuple,
    write_engines_json,
)


class TestUciTuple:
    """Tests for uci_tuple function."""

    def test_parses_simple_uci_string(self):
        """Parse simple UCI string with numeric value."""
        uci_string = "name Threads value 4"
        name, value = uci_tuple(uci_string)
        assert name == "Threads"
        assert value == 4.0

    def test_parses_uci_string_with_float_value(self):
        """Parse UCI string with float value."""
        uci_string = "name CPuct value 1.5"
        name, value = uci_tuple(uci_string)
        assert name == "CPuct"
        assert value == 1.5

    def test_parses_uci_string_with_string_value(self):
        """Parse UCI string with string value."""
        uci_string = "name SkillLevel value beginner"
        name, value = uci_tuple(uci_string)
        assert name == "SkillLevel"
        assert value == "beginner"

    def test_parses_uci_string_with_integer_value(self):
        """Parse UCI string with integer value."""
        uci_string = "name MultiPV value 5"
        name, value = uci_tuple(uci_string)
        assert name == "MultiPV"
        assert value == 5.0

    def test_parses_uci_string_with_negative_value(self):
        """Parse UCI string with negative value."""
        uci_string = "name EloMating value -100"
        name, value = uci_tuple(uci_string)
        assert name == "EloMating"
        assert value == -100.0

    def test_parses_uci_string_with_scientific_notation(self):
        """Parse UCI string with scientific notation."""
        uci_string = "name CPuct value 1.5e-2"
        name, value = uci_tuple(uci_string)
        assert name == "CPuct"
        assert value == 0.015

    def test_raises_for_invalid_format(self):
        """Raise error for invalid UCI format."""
        uci_string = "invalid format"
        with pytest.raises(SystemExit):
            uci_tuple(uci_string)

    def test_raises_for_missing_name(self):
        """Raise error for missing name field."""
        uci_string = "value 4"
        with pytest.raises(SystemExit):
            uci_tuple(uci_string)

    def test_raises_for_missing_value(self):
        """Raise error for missing value field."""
        uci_string = "name Threads"
        with pytest.raises(SystemExit):
            uci_tuple(uci_string)

    def test_parses_uci_string_with_spaces_in_value(self):
        """Parse UCI string with spaces in value."""
        uci_string = "name BookPath value /path/to/book"
        name, value = uci_tuple(uci_string)
        assert name == "BookPath"
        assert value == "/path/to/book"


class TestInitStrings:
    """Tests for InitStrings class."""

    def test_initializes_with_empty_list(self):
        """Initialize with empty list."""
        init_strings = InitStrings([])
        assert len(init_strings) == 0

    def test_initializes_with_uci_only(self):
        """Initialize with just uci command."""
        init_strings = InitStrings(["uci"])
        assert len(init_strings) == 1
        assert "uci" in init_strings._init_strings

    def test_initializes_with_single_entry(self):
        """Initialize with single entry."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        assert len(init_strings) == 2
        assert init_strings["Threads"] == 4.0

    def test_initializes_with_multiple_entries(self):
        """Initialize with multiple entries."""
        init_strings = InitStrings(
            [
                "uci",
                "setoption name Threads value 4",
                "setoption name Hash value 128",
            ]
        )
        assert len(init_strings) == 3
        assert init_strings["Threads"] == 4.0
        assert init_strings["Hash"] == 128.0

    def test_get_returns_value(self):
        """Get returns correct value."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        assert init_strings["Threads"] == 4.0

    def test_set_updates_value(self):
        """Set updates value."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        init_strings["Threads"] = 8
        assert init_strings["Threads"] == 8.0

    def test_set_adds_new_entry(self):
        """Set adds new entry if key doesn't exist."""
        init_strings = InitStrings(["uci"])
        init_strings["Threads"] = 4
        assert "Threads" in init_strings
        assert init_strings["Threads"] == 4.0

    def test_del_removes_entry(self):
        """Del removes entry."""
        init_strings = InitStrings(
            [
                "uci",
                "setoption name Threads value 4",
                "setoption name Hash value 128",
            ]
        )
        del init_strings["Threads"]
        assert "Threads" not in init_strings
        assert len(init_strings) == 2

    def test_iterates_over_keys(self):
        """Iteration yields keys."""
        init_strings = InitStrings(
            [
                "uci",
                "setoption name Threads value 4",
                "setoption name Hash value 128",
            ]
        )
        keys = list(init_strings)
        assert set(keys) == {"Threads", "Hash"}

    def test_contains_returns_true_for_existing_key(self):
        """Contains returns True for existing key."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        assert "Threads" in init_strings

    def test_contains_returns_false_for_missing_key(self):
        """Contains returns False for missing key."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        assert "Hash" not in init_strings

    def test_raises_keyerror_for_missing_key(self):
        """Raise KeyError for missing key."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        with pytest.raises(KeyError):
            _ = init_strings["Hash"]

    def test_repr_representation(self):
        """Repr returns internal list representation."""
        init_strings = InitStrings(["uci", "setoption name Threads value 4"])
        repr_str = repr(init_strings)
        assert "uci" in repr_str
        assert "Threads" in repr_str

    def test_update_with_dict(self):
        """Update with dictionary adds/updates entries."""
        init_strings = InitStrings(["uci"])
        init_strings.update({"Threads": 4, "Hash": 128})
        assert init_strings["Threads"] == 4.0
        assert init_strings["Hash"] == 128.0


class TestParseRanges:
    """Tests for parse_ranges function."""

    def test_parses_integer_range(self):
        """Parse integer range specification."""
        ranges = {"Threads": "Integer(1, 8)"}
        result = parse_ranges(ranges)
        assert "Threads" in result
        assert isinstance(result["Threads"], Integer)
        assert result["Threads"].low == 1
        assert result["Threads"].high == 8

    def test_parses_real_range(self):
        """Parse real range specification."""
        ranges = {"CPuct": "Real(0.0, 2.0)"}
        result = parse_ranges(ranges)
        assert "CPuct" in result
        assert isinstance(result["CPuct"], Real)
        assert result["CPuct"].low == 0.0
        assert result["CPuct"].high == 2.0

    def test_parses_categorical_range(self):
        """Parse categorical range specification using list syntax."""
        # Use JSON string format to pass a list
        ranges = {"Skill": '["beginner", "expert"]'}
        result = parse_ranges(ranges)
        assert "Skill" in result
        # The result should be a Categorical dimension
        assert hasattr(result["Skill"], "categories")
        assert result["Skill"].categories == ("beginner", "expert")

    def test_parses_multiple_ranges(self):
        """Parse multiple range specifications."""
        ranges = {
            "Threads": "Integer(1, 8)",
            "CPuct": "Real(0.0, 2.0)",
        }
        result = parse_ranges(ranges)
        assert len(result) == 2
        assert "Threads" in result
        assert "CPuct" in result

    def test_parses_integer_with_prior(self):
        """Parse integer range with prior (unquoted)."""
        ranges = {"Threads": "Integer(1, 8, prior=uniform)"}
        result = parse_ranges(ranges)
        assert "Threads" in result
        assert isinstance(result["Threads"], Integer)
        assert result["Threads"].prior == "uniform"

    def test_parses_real_with_prior(self):
        """Parse real range with prior (unquoted)."""
        # log-uniform prior requires strictly positive bounds
        ranges = {"CPuct": "Real(1.0, 2.0, prior=log-uniform)"}
        result = parse_ranges(ranges)
        assert "CPuct" in result
        assert isinstance(result["CPuct"], Real)
        assert result["CPuct"].prior == "log-uniform"

    def test_raises_for_invalid_range_type(self):
        """Raise error for invalid range type."""
        ranges = {"Threads": "Invalid(1, 8)"}
        with pytest.raises(ValueError):
            parse_ranges(ranges)

    def test_raises_for_missing_bounds(self):
        """Raise error for missing bounds."""
        ranges = {"Threads": "Integer()"}
        with pytest.raises(TypeError):
            parse_ranges(ranges)


class TestPrepareEnginesJson:
    """Tests for prepare_engines_json function."""

    def test_creates_engines_json_with_single_engine(self):
        """Create engines JSON with single engine."""
        commands = ["lc0"]
        directories = ["/tmp"]
        fixed_params = [{"Threads": 4}]
        result = prepare_engines_json(commands, directories, fixed_params)
        assert len(result) == 1
        assert result[0]["command"] == "lc0"
        assert result[0]["name"] == "engine1"
        assert result[0]["workingDirectory"] == "/tmp"
        assert result[0]["protocol"] == "uci"
        # Check that fixed params were applied via InitStrings
        uci = InitStrings(result[0]["initStrings"])
        assert uci["Threads"] == 4.0

    def test_creates_engines_json_with_multiple_engines(self):
        """Create engines JSON with multiple engines."""
        commands = ["lc0", "sf"]
        directories = ["/tmp/lc0", "/tmp/sf"]
        fixed_params = [{"Threads": 4}, {"Hash": 128}]
        result = prepare_engines_json(commands, directories, fixed_params)
        assert len(result) == 2
        assert result[0]["command"] == "lc0"
        assert result[1]["command"] == "sf"
        assert result[0]["name"] == "engine1"
        assert result[1]["name"] == "engine2"

    def test_creates_engines_json_with_empty_fixed_params(self):
        """Create engines JSON with empty fixed parameters."""
        commands = ["lc0"]
        directories = ["/tmp"]
        fixed_params = [{}]
        result = prepare_engines_json(commands, directories, fixed_params)
        assert len(result) == 1
        # Should only have "uci" in initStrings
        assert result[0]["initStrings"] == ["uci"]

    def test_creates_engines_json_with_multiple_params(self):
        """Create engines JSON with multiple fixed parameters."""
        commands = ["lc0"]
        directories = ["/tmp"]
        fixed_params = [{"Threads": 4, "Hash": 128, "MultiPV": 1}]
        result = prepare_engines_json(commands, directories, fixed_params)
        assert len(result) == 1
        uci = InitStrings(result[0]["initStrings"])
        assert uci["Threads"] == 4.0
        assert uci["Hash"] == 128.0
        assert uci["MultiPV"] == 1.0


class TestWriteEnginesJson:
    """Tests for write_engines_json function."""

    def test_writes_engines_json_to_file(self):
        """Write engines JSON to file with point dict."""
        import os

        # Create a mock engine_json structure
        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"Threads": 4}

        # Change to temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                write_engines_json(engine_json, point_dict)

                # Read and verify
                with open("engines.json", "r") as f:
                    data = json.load(f)

                assert len(data) == 1
                assert data[0]["command"] == "lc0"
                # Check that Threads was added to initStrings
                uci = InitStrings(data[0]["initStrings"])
                assert uci["Threads"] == 4.0
            finally:
                os.chdir(original_cwd)

    def test_writes_multiple_params_to_file(self):
        """Write engines JSON with multiple parameters."""
        import os

        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"Threads": 4, "Hash": 128}

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                write_engines_json(engine_json, point_dict)

                with open("engines.json", "r") as f:
                    data = json.load(f)

                uci = InitStrings(data[0]["initStrings"])
                assert uci["Threads"] == 4.0
                assert uci["Hash"] == 128.0
            finally:
                os.chdir(original_cwd)

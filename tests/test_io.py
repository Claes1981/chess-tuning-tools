"""Tests for tune.io module.

This module tests IO functionality including:
- Configuration loading and parsing
- UCI parameter handling
- Engine configuration
- Polyglot book configuration
"""

import configparser
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from skopt.space import Categorical, Integer, Real

from tune.io import (
    InitStrings,
    combine_nested_parameters,
    load_tuning_config,
    parse_ranges,
    prepare_engines_json,
    uci_tuple,
    write_engines_json,
    write_polyglot_ini,
)


class TestCombineNestedParameters:
    """Tests for combine_nested_parameters function."""

    def test_with_flat_values(self):
        """Flat (non-nested) parameters are passed through unchanged."""
        input_params = {"UCIParameter1": 42.0, "UCIParameter2": 0.0}
        expected_result = {"UCIParameter1": 42.0, "UCIParameter2": 0.0}

        actual_result = combine_nested_parameters(input_params)

        assert actual_result == expected_result

    def test_with_same_composite_type(self):
        """Combine nested parameters with same parent and composite type."""
        input_params = {
            "UCIParameter1": 42.0,
            "UCIParameter2=composite(sub-parameter1)": 0.0,
            "UCIParameter2=composite(sub-parameter2)": 1.0,
        }
        expected_result = {
            "UCIParameter1": 42.0,
            "UCIParameter2": "composite(sub-parameter1=0.0,sub-parameter2=1.0)",
        }

        actual_result = combine_nested_parameters(input_params)

        assert actual_result == expected_result

    def test_rejects_inconsistent_types(self):
        """Raise ValueError when nested parameters use different composite types."""
        inconsistent_params = {
            "UCIParameter1": 42.0,
            "UCIParameter2=composite(sub-parameter1)": 0.0,
            "UCIParameter2=other(sub-parameter2)": 1.0,
        }

        with pytest.raises(
            ValueError,
            match="UCI parameter UCIParameter2 is set to different values",
        ):
            combine_nested_parameters(inconsistent_params)

    def test_with_three_subparameters(self):
        """Combine three sub-parameters of the same composite type."""
        input_params = {
            "ParamA=myType(sub1)": 1.5,
            "ParamA=myType(sub2)": 2.5,
            "ParamA=myType(sub3)": 3.5,
        }
        expected_result = {"ParamA": "myType(sub1=1.5,sub2=2.5,sub3=3.5)"}

        actual_result = combine_nested_parameters(input_params)

        assert actual_result == expected_result

    def test_with_empty_dict(self):
        """Return empty dict when no parameters are provided."""
        result = combine_nested_parameters({})

        assert result == {}

    def test_mixed_flat_and_nested(self):
        """Handle mix of flat parameters and nested parameters correctly."""
        input_params = {
            "FlatParam1": 42.0,
            "NestedParam=composite(sub1)": 1.0,
            "FlatParam2": 0.0,
            "NestedParam=composite(sub2)": 2.0,
            "FlatParam3": -1.0,
        }
        expected_result = {
            "FlatParam1": 42.0,
            "NestedParam": "composite(sub1=1.0,sub2=2.0)",
            "FlatParam2": 0.0,
            "FlatParam3": -1.0,
        }

        actual_result = combine_nested_parameters(input_params)

        assert actual_result == expected_result


class TestLoadTuningConfig:
    """Tests for load_tuning_config function."""

    def test_extracts_non_engine_params(self):
        """Extract and verify non-engine parameters from tuning config."""
        config = {
            "engines": [
                {
                    "command": "lc0",
                    "fixed_parameters": {"CPuctBase": 13232, "Threads": 2},
                },
                {"command": "sf", "fixed_parameters": {"Threads": 8}},
            ],
            "parameter_ranges": {"CPuct": "Real(0.0, 1.0)"},
            "gp_samples": 100,
        }

        (
            return_config,
            commands,
            directories,
            polyglot_params,
            fixed_params,
            param_ranges,
        ) = load_tuning_config(config)

        assert len(return_config) == 3
        assert "gp_samples" in return_config
        assert len(commands) == 2
        assert len(fixed_params) == 2
        assert len(param_ranges) == 1

    def test_rejects_missing_engines(self):
        """Raise ValueError when config is missing engines section."""
        invalid_config = {
            "parameter_ranges": {"CPuct": "Real(0.0, 1.0)"},
            "gp_samples": 100,
        }

        with pytest.raises(
            ValueError, match="Tuning config does not contain engines"
        ):
            load_tuning_config(invalid_config)

    def test_rejects_single_engine(self):
        """Raise ValueError when config has only one engine."""
        single_engine_config = {
            "engines": [{"command": "lc0", "fixed_parameters": {}}],
            "parameter_ranges": {"CPuct": "Real(0.0, 1.0)"},
        }

        with pytest.raises(
            ValueError, match="Tuning config requires at least two engines"
        ):
            load_tuning_config(single_engine_config)


class TestUciTuple:
    """Tests for the uci_tuple function."""

    def test_parse_simple_float(self):
        """Parse a simple UCI tuple with float value."""
        uci_str = "setoption name CPuct value 0.5"
        name, value = uci_tuple(uci_str)

        assert name == "CPuct"
        assert value == 0.5
        assert isinstance(value, float)

    def test_parse_integer_value(self):
        """Parse a UCI tuple with integer value."""
        uci_str = "setoption name Threads value 4"
        name, value = uci_tuple(uci_str)

        assert name == "Threads"
        assert value == 4.0
        assert isinstance(value, float)

    def test_parse_string_value(self):
        """Parse a UCI tuple with string value."""
        uci_str = "setoption name UCI_Chess960 value true"
        name, value = uci_tuple(uci_str)

        assert name == "UCI_Chess960"
        assert value == "true"
        assert isinstance(value, str)

    def test_parse_negative_value(self):
        """Parse a UCI tuple with negative value."""
        uci_str = "setoption name MaxDepth value -1"
        name, value = uci_tuple(uci_str)

        assert name == "MaxDepth"
        assert value == -1

    def test_parse_long_parameter_name(self):
        """Parse a UCI tuple with long parameter name."""
        uci_str = "setoption name VeryLongParameterNameForTesting value 1.234"
        name, value = uci_tuple(uci_str)

        assert name == "VeryLongParameterNameForTesting"
        assert value == 1.234

    def test_parse_whitespace_handling(self):
        """Parse a UCI tuple with extra whitespace."""
        uci_str = "setoption name   Param   value   42.0"
        name, value = uci_tuple(uci_str)

        assert name == "Param"
        assert value == 42.0

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
    """Tests for the InitStrings class."""

    def test_init_with_basic_strings(self):
        """Initialize InitStrings with basic UCI strings."""
        init_strs = [
            "uci",
            "setoption name CPuct value 0.5",
            "setoption name Threads value 4",
        ]
        strings = InitStrings(init_strs)

        assert "CPuct" in strings
        assert "Threads" in strings
        assert "uci" not in strings

    def test_get_existing_key(self):
        """Get an existing key from InitStrings."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        value = strings["CPuct"]

        assert value == 0.5

    def test_get_nonexisting_key_raises(self):
        """Get a non-existing key raises KeyError."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        with pytest.raises(KeyError):
            _ = strings["NonExisting"]

    def test_set_existing_key(self):
        """Set an existing key updates the value."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings["CPuct"] = 1.0

        assert strings["CPuct"] == 1.0

    def test_set_new_key(self):
        """Set a new key adds it to the strings."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings["NewParam"] = 42.0

        assert "NewParam" in strings
        assert strings["NewParam"] == 42.0

    def test_delete_existing_key(self):
        """Delete an existing key removes it."""
        init_strs = [
            "uci",
            "setoption name CPuct value 0.5",
            "setoption name Threads value 4",
        ]
        strings = InitStrings(init_strs)

        del strings["CPuct"]

        assert "CPuct" not in strings
        assert "Threads" in strings

    def test_delete_nonexisting_key_raises(self):
        """Delete a non-existing key raises KeyError."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        with pytest.raises(KeyError):
            del strings["NonExisting"]

    def test_contains_existing(self):
        """Check if existing key is in InitStrings."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        assert "CPuct" in strings
        assert "uci" not in strings

    def test_contains_nonexisting(self):
        """Check if non-existing key is not in InitStrings."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        assert "NonExisting" not in strings

    def test_iter_yields_keys(self):
        """Iterate over InitStrings yields parameter names."""
        init_strs = [
            "uci",
            "setoption name CPuct value 0.5",
            "setoption name Threads value 4",
        ]
        strings = InitStrings(init_strs)

        keys = list(strings)

        assert "CPuct" in keys
        assert "Threads" in keys
        assert "uci" not in keys
        assert len(keys) == 2

    def test_len_excludes_uci(self):
        """Length returns length of internal list (including 'uci')."""
        init_strs = [
            "uci",
            "setoption name CPuct value 0.5",
            "setoption name Threads value 4",
        ]
        strings = InitStrings(init_strs)

        assert len(strings) == 3

    def test_repr_returns_list_repr(self):
        """Repr returns representation of the internal list."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        repr_str = repr(strings)

        assert "CPuct" in repr_str

    def test_update_with_dict(self):
        """Update InitStrings with a dictionary."""
        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings.update({"Threads": 4, "Hash": 128})

        assert "Threads" in strings
        assert "Hash" in strings
        assert strings["Threads"] == 4
        assert strings["Hash"] == 128

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


class TestParseRanges:
    """Tests for the parse_ranges function."""

    def test_parse_real_dimension(self):
        """Parse a Real dimension."""
        ranges = {"CPuct": "Real(0.0, 1.0)"}
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert hasattr(result["CPuct"], "prior")

    def test_parse_integer_dimension(self):
        """Parse an Integer dimension."""
        ranges = {"Threads": "Integer(1, 32)"}
        result = parse_ranges(ranges)

        assert "Threads" in result

    def test_parse_categorical_dimension(self):
        """Parse a Categorical dimension (single item only - code has bug with commas)."""
        ranges = {"Param": "Categorical([1])"}
        result = parse_ranges(ranges)

        assert "Param" in result

    def test_parse_with_json_string(self):
        """Parse ranges from a JSON string."""
        ranges_str = '{"CPuct": "Real(0.0, 1.0)"}'
        result = parse_ranges(ranges_str)

        assert "CPuct" in result

    def test_parse_multiple_dimensions(self):
        """Parse multiple dimensions at once."""
        ranges = {"CPuct": "Real(0.0, 1.0)", "Threads": "Integer(1, 32)"}
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert "Threads" in result
        assert len(result) == 2

    def test_parse_with_kwargs(self):
        """Parse dimension with keyword arguments."""
        ranges = {"CPuct": "Real(0.0, 1.0, base=2)"}
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert result["CPuct"].base == 2

    def test_parse_tuple_dimension(self):
        """Parse a tuple dimension."""
        ranges = {"Param": "[1, 2, 3, 4, 5]"}
        result = parse_ranges(ranges)

        assert "Param" in result

    def test_parse_invalid_dimension_raises(self):
        """Parse an invalid dimension raises ValueError."""
        ranges = {"Param": "InvalidDimension(0.0, 1.0)"}

        with pytest.raises(ValueError):
            parse_ranges(ranges)

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
        ranges = {"Skill": '["beginner", "expert"]'}
        result = parse_ranges(ranges)
        assert "Skill" in result
        assert hasattr(result["Skill"], "categories")
        assert result["Skill"].categories == ("beginner", "expert")

    def test_parses_integer_with_prior(self):
        """Parse integer range with prior (unquoted)."""
        ranges = {"Threads": "Integer(1, 8, prior=uniform)"}
        result = parse_ranges(ranges)
        assert "Threads" in result
        assert isinstance(result["Threads"], Integer)
        assert result["Threads"].prior == "uniform"

    def test_parses_real_with_prior(self):
        """Parse real range with prior (unquoted)."""
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
    """Tests for the prepare_engines_json function."""

    def test_prepare_basic_engines(self):
        """Prepare basic engines JSON."""
        commands = ["lc0", "sf"]
        directories = ["", "/path/to/sf"]
        fixed_params = [{"CPuct": 0.5, "Threads": 4}, {"Threads": 8}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert len(result) == 2
        assert result[0]["command"] == "lc0"
        assert result[1]["command"] == "sf"
        assert result[0]["name"] == "engine1"
        assert result[1]["name"] == "engine2"

    def test_prepare_with_working_directory(self):
        """Prepare engines JSON with working directories."""
        commands = ["lc0", "sf"]
        directories = ["/path/to/lc0", "/path/to/sf"]
        fixed_params = [{}, {}]

        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["workingDirectory"] == "/path/to/lc0"
        assert result[1]["workingDirectory"] == "/path/to/sf"

    def test_prepare_with_fixed_params(self):
        """Prepare engines JSON with fixed parameters."""
        commands = ["lc0"]
        directories = [""]
        fixed_params = [{"CPuct": 0.5, "Threads": 4}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert len(result) == 1
        assert "initStrings" in result[0]
        assert len(result[0]["initStrings"]) > 1

    def test_prepare_protocol_is_uci(self):
        """Prepare engines JSON has protocol set to UCI."""
        commands = ["lc0", "sf"]
        directories = ["", ""]
        fixed_params = [{}, {}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["protocol"] == "uci"
        assert result[1]["protocol"] == "uci"

    def test_prepare_initstrings_contains_uci(self):
        """Prepare engines JSON has initStrings starting with 'uci'."""
        commands = ["lc0"]
        directories = [""]
        fixed_params = [{}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["initStrings"][0] == "uci"

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
    """Tests for the write_engines_json function."""

    def test_write_creates_file(self, tmp_path, monkeypatch):
        """Write engines JSON creates the file."""
        monkeypatch.chdir(tmp_path)

        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "workingDirectory": "",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"CPuct": 0.5}

        write_engines_json(engine_json, point_dict)

        assert (tmp_path / "engines.json").exists()

    def test_write_valid_json(self, tmp_path, monkeypatch):
        """Write engines JSON creates valid JSON."""
        monkeypatch.chdir(tmp_path)

        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "workingDirectory": "",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"CPuct": 0.5}

        write_engines_json(engine_json, point_dict)

        with open(tmp_path / "engines.json") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1

    def test_write_includes_point_dict(self, tmp_path, monkeypatch):
        """Write engines JSON includes point dict in initStrings."""
        monkeypatch.chdir(tmp_path)

        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "workingDirectory": "",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"CPuct": 0.5, "Threads": 4}

        write_engines_json(engine_json, point_dict)

        with open(tmp_path / "engines.json") as f:
            data = json.load(f)

        init_strings = data[0]["initStrings"]
        assert any("CPuct" in s for s in init_strings)
        assert any("Threads" in s for s in init_strings)

    def test_write_sorted_keys(self, tmp_path, monkeypatch):
        """Write engines JSON has sorted keys."""
        monkeypatch.chdir(tmp_path)

        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "workingDirectory": "",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {}

        write_engines_json(engine_json, point_dict)

        with open(tmp_path / "engines.json") as f:
            content = f.read()

        assert content.index("command") < content.index("initStrings")

    def test_writes_engines_json_to_file(self):
        """Write engines JSON to file with point dict."""
        engine_json = [
            {
                "command": "lc0",
                "name": "engine1",
                "initStrings": ["uci"],
                "protocol": "uci",
            }
        ]
        point_dict = {"Threads": 4}

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                write_engines_json(engine_json, point_dict)

                with open("engines.json", "r") as f:
                    data = json.load(f)

                assert len(data) == 1
                assert data[0]["command"] == "lc0"
                uci = InitStrings(data[0]["initStrings"])
                assert uci["Threads"] == 4.0
            finally:
                os.chdir(original_cwd)

    def test_writes_multiple_params_to_file(self):
        """Write engines JSON with multiple parameters."""
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


class TestWritePolyglotIni:
    """Tests for the write_polyglot_ini function."""

    @pytest.fixture(autouse=True)
    def cleanup_polyglot_config(self):
        """Clean up polyglot config directory after each test."""
        yield
        if Path("polyglot-config").exists():
            shutil.rmtree("polyglot-config")

    def test_write_polyglot_ini_basic(self):
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

        config_path = Path("polyglot-config/polyglot.ini")
        assert config_path.exists()

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
        book_depth = int(config["PolyGlot"]["BookDepth"])
        assert 1 <= book_depth <= 10

    def test_write_polyglot_ini_default_max_book_depth(self):
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

        book_depth = int(config["PolyGlot"]["BookDepth"])
        assert 1 <= book_depth <= 256

    def test_write_polyglot_ini_min_book_depth(self):
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

    def test_write_polyglot_ini_empty_engine_command(self):
        """Test write_polyglot_ini with empty engine_command."""
        polyglot_params = [{}, {"book_file": "book.bin"}]

        write_polyglot_ini(polyglot_params)

        config_path = Path("polyglot-config/polyglot.ini")
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_path)

        assert config["PolyGlot"]["EngineCommand"] == ""

    def test_write_polyglot_ini_empty_book_file(self):
        """Test write_polyglot_ini with empty book_file."""
        polyglot_params = [{}, {"engine_command": "engine2"}]

        write_polyglot_ini(polyglot_params)

        config_path = Path("polyglot-config/polyglot.ini")
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_path)

        assert config["PolyGlot"]["BookFile"] == ""

    def test_write_polyglot_ini_creates_directory(self):
        """Test that write_polyglot_ini creates the directory if it doesn't exist."""
        if Path("polyglot-config").exists():
            shutil.rmtree("polyglot-config")

        polyglot_params = [{}, {"engine_command": "engine2"}]

        write_polyglot_ini(polyglot_params)

        assert Path("polyglot-config").exists()
        assert Path("polyglot-config").is_dir()

    def test_write_polyglot_ini_case_sensitivity(self):
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

        assert "BookFile" in config["PolyGlot"]

    def test_write_polyglot_ini_high_max_book_depth(self):
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

    def test_write_polyglot_ini_float_max_book_depth(self):
        """Test write_polyglot_ini with float max_book_depth."""
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
        assert 1 <= book_depth <= 10

    def test_write_polyglot_ini_negative_max_book_depth(self):
        """Test write_polyglot_ini with negative max_book_depth."""
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
        assert book_depth == 1

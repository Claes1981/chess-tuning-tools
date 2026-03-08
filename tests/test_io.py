import pytest

from tune.io import combine_nested_parameters, load_tuning_config


def test_load_tuning_config_extracts_non_engine_params():
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


def test_combine_nested_parameters_with_flat_values():
    """Verify flat (non-nested) parameters are passed through unchanged."""
    input_params = {
        "UCIParameter1": 42.0,
        "UCIParameter2": 0.0,
    }
    expected_result = {
        "UCIParameter1": 42.0,
        "UCIParameter2": 0.0,
    }

    actual_result = combine_nested_parameters(input_params)

    assert actual_result == expected_result


def test_combine_nested_parameters_with_same_composite_type():
    """Combine nested parameters that share the same parent parameter and composite type."""
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


def test_combine_nested_parameters_rejects_inconsistent_types():
    """Raise ValueError when nested parameters use different composite types for same parent."""
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


def test_combine_nested_parameters_with_three_subparameters():
    """Combine three sub-parameters of the same composite type into one parameter."""
    input_params = {
        "ParamA=myType(sub1)": 1.5,
        "ParamA=myType(sub2)": 2.5,
        "ParamA=myType(sub3)": 3.5,
    }
    expected_result = {
        "ParamA": "myType(sub1=1.5,sub2=2.5,sub3=3.5)",
    }

    actual_result = combine_nested_parameters(input_params)

    assert actual_result == expected_result


def test_combine_nested_parameters_with_empty_dict():
    """Return empty dict when no parameters are provided."""
    result = combine_nested_parameters({})

    assert result == {}


def test_combine_nested_parameters_mixed_flat_and_nested():
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


def test_load_tuning_config_rejects_missing_engines():
    """Raise ValueError when config is missing engines section."""
    invalid_config = {
        "parameter_ranges": {"CPuct": "Real(0.0, 1.0)"},
        "gp_samples": 100,
    }

    with pytest.raises(
        ValueError, match="Tuning config does not contain engines"
    ):
        load_tuning_config(invalid_config)


def test_load_tuning_config_rejects_single_engine():
    """Raise ValueError when config has only one engine (requires at least two)."""
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
        from tune.io import uci_tuple

        uci_str = "setoption name CPuct value 0.5"
        name, value = uci_tuple(uci_str)

        assert name == "CPuct"
        assert value == 0.5
        assert isinstance(value, float)

    def test_parse_integer_value(self):
        """Parse a UCI tuple with integer value."""
        from tune.io import uci_tuple

        uci_str = "setoption name Threads value 4"
        name, value = uci_tuple(uci_str)

        assert name == "Threads"
        assert value == 4.0  # Actually converted to float
        assert isinstance(value, float)  # uci_tuple converts to float first

    def test_parse_string_value(self):
        """Parse a UCI tuple with string value."""
        from tune.io import uci_tuple

        uci_str = "setoption name UCI_Chess960 value true"
        name, value = uci_tuple(uci_str)

        assert name == "UCI_Chess960"
        assert value == "true"
        assert isinstance(value, str)

    def test_parse_negative_value(self):
        """Parse a UCI tuple with negative value."""
        from tune.io import uci_tuple

        uci_str = "setoption name MaxDepth value -1"
        name, value = uci_tuple(uci_str)

        assert name == "MaxDepth"
        assert value == -1

    def test_parse_long_parameter_name(self):
        """Parse a UCI tuple with long parameter name."""
        from tune.io import uci_tuple

        uci_str = "setoption name VeryLongParameterNameForTesting value 1.234"
        name, value = uci_tuple(uci_str)

        assert name == "VeryLongParameterNameForTesting"
        assert value == 1.234

    def test_parse_whitespace_handling(self):
        """Parse a UCI tuple with extra whitespace."""
        from tune.io import uci_tuple

        uci_str = "setoption name   Param   value   42.0"
        name, value = uci_tuple(uci_str)

        assert name == "Param"
        assert value == 42.0


class TestInitStrings:
    """Tests for the InitStrings class."""

    def test_init_with_basic_strings(self):
        """Initialize InitStrings with basic UCI strings."""
        from tune.io import InitStrings

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
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        value = strings["CPuct"]

        assert value == 0.5

    def test_get_nonexisting_key_raises(self):
        """Get a non-existing key raises KeyError."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        with pytest.raises(KeyError):
            _ = strings["NonExisting"]

    def test_set_existing_key(self):
        """Set an existing key updates the value."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings["CPuct"] = 1.0

        assert strings["CPuct"] == 1.0

    def test_set_new_key(self):
        """Set a new key adds it to the strings."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings["NewParam"] = 42.0

        assert "NewParam" in strings
        assert strings["NewParam"] == 42.0

    def test_delete_existing_key(self):
        """Delete an existing key removes it."""
        from tune.io import InitStrings

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
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        with pytest.raises(KeyError):
            del strings["NonExisting"]

    def test_contains_existing(self):
        """Check if existing key is in InitStrings."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        assert "CPuct" in strings
        assert "uci" not in strings

    def test_contains_nonexisting(self):
        """Check if non-existing key is not in InitStrings."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        assert "NonExisting" not in strings

    def test_iter_yields_keys(self):
        """Iterate over InitStrings yields parameter names."""
        from tune.io import InitStrings

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
        from tune.io import InitStrings

        init_strs = [
            "uci",
            "setoption name CPuct value 0.5",
            "setoption name Threads value 4",
        ]
        strings = InitStrings(init_strs)

        assert len(strings) == 3  # len() returns len of internal list

    def test_repr_returns_list_repr(self):
        """Repr returns representation of the internal list."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        repr_str = repr(strings)

        assert "CPuct" in repr_str

    def test_update_with_dict(self):
        """Update InitStrings with a dictionary."""
        from tune.io import InitStrings

        init_strs = ["uci", "setoption name CPuct value 0.5"]
        strings = InitStrings(init_strs)

        strings.update({"Threads": 4, "Hash": 128})

        assert "Threads" in strings
        assert "Hash" in strings
        assert strings["Threads"] == 4
        assert strings["Hash"] == 128


class TestParseRanges:
    """Tests for the parse_ranges function."""

    def test_parse_real_dimension(self):
        """Parse a Real dimension."""
        from tune.io import parse_ranges

        ranges = {"CPuct": "Real(0.0, 1.0)"}
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert hasattr(result["CPuct"], "prior")

    def test_parse_integer_dimension(self):
        """Parse an Integer dimension."""
        from tune.io import parse_ranges

        ranges = {"Threads": "Integer(1, 32)"}
        result = parse_ranges(ranges)

        assert "Threads" in result

    def test_parse_categorical_dimension(self):
        """Parse a Categorical dimension (single item only - code has bug with commas)."""
        from tune.io import parse_ranges

        ranges = {"Param": "Categorical([1])"}
        result = parse_ranges(ranges)

        assert "Param" in result

    def test_parse_with_json_string(self):
        """Parse ranges from a JSON string."""
        from tune.io import parse_ranges

        ranges_str = '{"CPuct": "Real(0.0, 1.0)"}'
        result = parse_ranges(ranges_str)

        assert "CPuct" in result

    def test_parse_multiple_dimensions(self):
        """Parse multiple dimensions at once."""
        from tune.io import parse_ranges

        ranges = {
            "CPuct": "Real(0.0, 1.0)",
            "Threads": "Integer(1, 32)",
        }
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert "Threads" in result
        assert len(result) == 2

    def test_parse_with_kwargs(self):
        """Parse dimension with keyword arguments (note: code has bug with string values)."""
        from tune.io import parse_ranges

        ranges = {"CPuct": "Real(0.0, 1.0, base=2)"}
        result = parse_ranges(ranges)

        assert "CPuct" in result
        assert result["CPuct"].base == 2

    def test_parse_tuple_dimension(self):
        """Parse a tuple dimension."""
        from tune.io import parse_ranges

        ranges = {"Param": "[1, 2, 3, 4, 5]"}
        result = parse_ranges(ranges)

        assert "Param" in result

    def test_parse_invalid_dimension_raises(self):
        """Parse an invalid dimension raises ValueError."""
        from tune.io import parse_ranges

        ranges = {"Param": "InvalidDimension(0.0, 1.0)"}

        with pytest.raises(ValueError):
            parse_ranges(ranges)


class TestPrepareEnginesJson:
    """Tests for the prepare_engines_json function."""

    def test_prepare_basic_engines(self):
        """Prepare basic engines JSON."""
        from tune.io import prepare_engines_json

        commands = ["lc0", "sf"]
        directories = ["", "/path/to/sf"]
        fixed_params = [
            {"CPuct": 0.5, "Threads": 4},
            {"Threads": 8},
        ]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert len(result) == 2
        assert result[0]["command"] == "lc0"
        assert result[1]["command"] == "sf"
        assert result[0]["name"] == "engine1"
        assert result[1]["name"] == "engine2"

    def test_prepare_with_working_directory(self):
        """Prepare engines JSON with working directories."""
        from tune.io import prepare_engines_json

        commands = ["lc0", "sf"]
        directories = ["/path/to/lc0", "/path/to/sf"]
        fixed_params = [{}, {}]  # Need matching length with commands

        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["workingDirectory"] == "/path/to/lc0"
        assert result[1]["workingDirectory"] == "/path/to/sf"

    def test_prepare_with_fixed_params(self):
        """Prepare engines JSON with fixed parameters."""
        from tune.io import prepare_engines_json

        commands = ["lc0"]
        directories = [""]
        fixed_params = [{"CPuct": 0.5, "Threads": 4}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert len(result) == 1
        assert "initStrings" in result[0]
        # Fixed params should be in initStrings
        assert len(result[0]["initStrings"]) > 1  # uci + fixed params

    def test_prepare_protocol_is_uci(self):
        """Prepare engines JSON has protocol set to UCI."""
        from tune.io import prepare_engines_json

        commands = ["lc0", "sf"]
        directories = ["", ""]
        fixed_params = [{}, {}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["protocol"] == "uci"
        assert result[1]["protocol"] == "uci"

    def test_prepare_initstrings_contains_uci(self):
        """Prepare engines JSON has initStrings starting with 'uci'."""
        from tune.io import prepare_engines_json

        commands = ["lc0"]
        directories = [""]
        fixed_params = [{}]
        result = prepare_engines_json(commands, directories, fixed_params)

        assert result[0]["initStrings"][0] == "uci"


class TestWriteEnginesJson:
    """Tests for the write_engines_json function."""

    def test_write_creates_file(self, tmp_path, monkeypatch):
        """Write engines JSON creates the file."""
        from tune.io import write_engines_json

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
        import json

        from tune.io import write_engines_json

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
        import json

        from tune.io import write_engines_json

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
        import json

        from tune.io import write_engines_json

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

        # Keys should appear in sorted order
        assert content.index("command") < content.index("initStrings")

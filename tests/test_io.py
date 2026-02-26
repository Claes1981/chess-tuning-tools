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

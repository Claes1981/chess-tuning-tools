"""Extended tests for utility functions in tune/utils.py."""

from decimal import Decimal

import numpy as np
import pytest
from scipy.special import erfinv

import tune.utils as utils_module


class TestConfidenceToMult:
    """Tests for confidence_to_mult function."""

    def test_returns_mult_for_confidence_level_0_95(self):
        """Returns multiplier for 95% confidence level."""
        result = utils_module.confidence_to_mult(0.95)
        assert isinstance(result, (int, float, np.floating))
        assert result > 0.0
        # 95% confidence should give approximately 1.96
        assert 1.9 < result < 2.0

    def test_returns_mult_for_confidence_level_0_99(self):
        """Returns multiplier for 99% confidence level."""
        result = utils_module.confidence_to_mult(0.99)
        assert isinstance(result, (int, float, np.floating))
        assert result > 0.0
        # 99% confidence should give approximately 2.58
        assert 2.5 < result < 2.7

    def test_returns_mult_for_confidence_level_0_90(self):
        """Returns multiplier for 90% confidence level."""
        result = utils_module.confidence_to_mult(0.90)
        assert isinstance(result, (int, float, np.floating))
        assert result > 0.0
        # 90% confidence should give approximately 1.645
        assert 1.6 < result < 1.7

    def test_returns_value_for_confidence_level_0_5(self):
        """Returns a value for 50% confidence level (not 1.0)."""
        result = utils_module.confidence_to_mult(0.5)
        assert isinstance(result, (int, float, np.floating))
        # erfinv(0.5) * sqrt(2) ≈ 0.674
        assert 0.6 < result < 0.7

    def test_returns_0_for_confidence_level_0(self):
        """Returns 0.0 for 0% confidence level."""
        result = utils_module.confidence_to_mult(0.0)
        assert isinstance(result, (int, float, np.floating))
        assert result == 0.0

    def test_returns_mult_for_confidence_level_1(self):
        """Returns a value for 100% confidence level."""
        result = utils_module.confidence_to_mult(1.0)
        assert isinstance(result, (int, float, np.floating))
        # erfinv(1.0) = inf, so result should be inf
        assert np.isinf(result)

    def test_raises_for_confidence_above_1(self):
        """Raises ValueError for confidence above 1.0."""
        with pytest.raises(ValueError):
            utils_module.confidence_to_mult(1.1)

    def test_raises_for_negative_confidence(self):
        """Raises ValueError for negative confidence."""
        with pytest.raises(ValueError):
            utils_module.confidence_to_mult(-0.1)

    def test_mult_increases_with_confidence(self):
        """Multiplier increases as confidence level increases."""
        mult_90 = utils_module.confidence_to_mult(0.90)
        mult_95 = utils_module.confidence_to_mult(0.95)
        mult_99 = utils_module.confidence_to_mult(0.99)
        assert mult_90 < mult_95 < mult_99

    def test_uses_erfinv_formula(self):
        """Uses erfinv(confidence) * sqrt(2) formula."""
        confidence = 0.95
        result = utils_module.confidence_to_mult(confidence)
        expected = erfinv(confidence) * np.sqrt(2)
        np.testing.assert_almost_equal(result, expected)


class TestTimeControl:
    """Tests for TimeControl class (namedtuple with from_string method)."""

    def test_is_namedtuple_with_time_and_increment(self):
        """TimeControl is a namedtuple with 'time' and 'increment' fields."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        assert hasattr(tc, "time")
        assert hasattr(tc, "increment")
        assert tc.time == Decimal("10")
        assert tc.increment == Decimal("0.1")

    def test_initializes_with_integer_time(self):
        """Initializes with integer time value (no increment)."""
        tc = utils_module.TimeControl.from_string("100")
        assert tc.time == Decimal("100")
        assert tc.increment == Decimal("0")

    def test_initializes_with_float_increment(self):
        """Initializes with small float increment."""
        tc = utils_module.TimeControl.from_string("5+0.05")
        assert tc.time == Decimal("5")
        assert tc.increment == Decimal("0.05")

    def test_initializes_with_large_time(self):
        """Initializes with large time value."""
        tc = utils_module.TimeControl.from_string("180+2")
        assert tc.time == Decimal("180")
        assert tc.increment == Decimal("2")

    def test_initializes_with_zero_increment(self):
        """Initializes with explicit zero increment."""
        tc = utils_module.TimeControl.from_string("10+0")
        assert tc.time == Decimal("10")
        assert tc.increment == Decimal("0")

    def test_str_representation_no_increment(self):
        """String representation shows only time when increment is 0."""
        tc = utils_module.TimeControl.from_string("100")
        assert str(tc) == "100"

    def test_str_representation_with_increment(self):
        """String representation shows time+increment format."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        assert str(tc) == "10+0.1"

    def test_str_representation_small_increment(self):
        """String representation shows small increment."""
        tc = utils_module.TimeControl.from_string("5+0.05")
        assert str(tc) == "5+0.05"

    def test_equality_comparison(self):
        """Equality comparison works correctly."""
        tc1 = utils_module.TimeControl.from_string("10+0.1")
        tc2 = utils_module.TimeControl.from_string("10+0.1")
        assert tc1 == tc2

    def test_inequality_comparison(self):
        """Inequality comparison works correctly."""
        tc1 = utils_module.TimeControl.from_string("10+0.1")
        tc2 = utils_module.TimeControl.from_string("5+0.05")
        assert tc1 != tc2

    def test_can_create_directly_with_time_and_increment(self):
        """Can create TimeControl directly with time and increment values."""
        tc = utils_module.TimeControl(
            time=Decimal("10"), increment=Decimal("0.1")
        )
        assert tc.time == Decimal("10")
        assert tc.increment == Decimal("0.1")


class TestTimeControlBag:
    """Tests for TimeControlBag class."""

    def test_initializes_with_empty_list(self):
        """Initializes with empty time control list."""
        bag = utils_module.TimeControlBag([])
        assert hasattr(bag, "tcs")
        assert hasattr(bag, "bag")
        assert len(bag.tcs) == 0

    def test_initializes_with_single_timecontrol(self):
        """Initializes with single time control."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        bag = utils_module.TimeControlBag([tc])
        assert len(bag.tcs) == 1
        assert bag.tcs[0] == tc

    def test_initializes_with_multiple_timecontrols(self):
        """Initializes with multiple time controls."""
        tcs = [
            utils_module.TimeControl.from_string("10+0.1"),
            utils_module.TimeControl.from_string("5+0.05"),
            utils_module.TimeControl.from_string("100"),
        ]
        bag = utils_module.TimeControlBag(tcs)
        assert len(bag.tcs) == 3

    def test_sets_default_uniform_probabilities(self):
        """Sets default uniform probabilities when p is None."""
        tcs = [
            utils_module.TimeControl.from_string("10+0.1"),
            utils_module.TimeControl.from_string("5+0.05"),
        ]
        bag = utils_module.TimeControlBag(tcs)
        assert hasattr(bag, "p")
        # Default should be uniform: [0.5, 0.5]
        np.testing.assert_array_almost_equal(bag.p, [0.5, 0.5])

    def test_sets_custom_probabilities(self):
        """Sets custom probabilities when p is provided."""
        tcs = [
            utils_module.TimeControl.from_string("10+0.1"),
            utils_module.TimeControl.from_string("5+0.05"),
        ]
        bag = utils_module.TimeControlBag(tcs, p=[0.7, 0.3])
        np.testing.assert_array_almost_equal(bag.p, [0.7, 0.3])

    def test_sets_bag_size(self):
        """Sets bag_size parameter."""
        tcs = [utils_module.TimeControl.from_string("10+0.1")]
        bag = utils_module.TimeControlBag(tcs, bag_size=20)
        assert bag.bag_size == 20

    def test_default_bag_size_is_10(self):
        """Default bag_size is 10."""
        tcs = [utils_module.TimeControl.from_string("10+0.1")]
        bag = utils_module.TimeControlBag(tcs)
        assert bag.bag_size == 10

    def test_next_tc_returns_timecontrol(self):
        """next_tc returns a TimeControl object."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        bag = utils_module.TimeControlBag([tc], bag_size=10)
        result = bag.next_tc()
        assert isinstance(result, utils_module.TimeControl)
        assert result == tc

    def test_next_tc_populates_bag_on_first_call(self):
        """next_tc populates the bag on first call."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        bag = utils_module.TimeControlBag([tc], bag_size=10)
        assert bag.bag == []  # Initially empty
        _ = bag.next_tc()
        assert len(bag.bag) < 10  # One was popped

    def test_next_tc_returns_different_tcs_with_multiple_options(self):
        """next_tc returns different time controls when multiple options exist."""
        tcs = [
            utils_module.TimeControl.from_string("10+0.1"),
            utils_module.TimeControl.from_string("5+0.05"),
        ]
        bag = utils_module.TimeControlBag(tcs, bag_size=20)
        results = [bag.next_tc() for _ in range(20)]
        # Should have a mix of both time controls
        has_10_0_1 = any(r == tcs[0] for r in results)
        has_5_0_05 = any(r == tcs[1] for r in results)
        assert has_10_0_1 and has_5_0_05

    def test_next_tc_regenerates_bag_when_empty(self):
        """next_tc regenerates the bag when it becomes empty."""
        tc = utils_module.TimeControl.from_string("10+0.1")
        bag = utils_module.TimeControlBag([tc], bag_size=10)
        # Drain the bag
        for _ in range(10):
            bag.next_tc()
        # Should still be able to get more
        result = bag.next_tc()
        assert result == tc


class TestParseTimecontrol:
    """Tests for parse_timecontrol function."""

    def test_parses_standard_timecontrol(self):
        """Parses standard time control format into tuple of Decimals."""
        result = utils_module.parse_timecontrol("10+0.1")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == Decimal("10")
        assert result[1] == Decimal("0.1")

    def test_parses_integer_timecontrol(self):
        """Parses integer time control into single-element tuple."""
        result = utils_module.parse_timecontrol("100")
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0] == Decimal("100")

    def test_parses_float_increment(self):
        """Parses time control with float increment."""
        result = utils_module.parse_timecontrol("5+0.05")
        assert result[0] == Decimal("5")
        assert result[1] == Decimal("0.05")

    def test_parses_zero_increment(self):
        """Parses time control with zero increment."""
        result = utils_module.parse_timecontrol("10+0")
        assert result[0] == Decimal("10")
        assert result[1] == Decimal("0")

    def test_returns_tuple_of_decimals(self):
        """Returns tuple of Decimal objects."""
        result = utils_module.parse_timecontrol("10+0.1")
        assert all(isinstance(x, Decimal) for x in result)

    def test_used_by_timecontrol_from_string(self):
        """parse_timecontrol is used internally by TimeControl.from_string."""
        # from_string uses parse_timecontrol to parse the input
        tc = utils_module.TimeControl.from_string("10+0.1")
        parsed = utils_module.parse_timecontrol("10+0.1")
        assert tc.time == parsed[0]
        assert tc.increment == parsed[1]

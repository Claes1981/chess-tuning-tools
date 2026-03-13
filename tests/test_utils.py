"""Test utility functions of the project."""

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import tune.utils as utils_module
from tune.utils import (
    confidence_to_mult,
    expected_ucb,
    latest_iterations,
    parse_timecontrol,
    TimeControl,
    TimeControlBag,
)


class TestLatestIterations:
    """Tests for latest_iterations function."""

    def test_filters_to_last_occurrence_of_each_value(self):
        """Returns only the last occurrence of each unique value."""
        iterations = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
        expected_indices = [0, 1, 3, 4]

        result = latest_iterations(iterations)
        assert len(result) == 1
        assert_allclose(result, (iterations[expected_indices],))

    def test_filters_additional_arrays(self):
        """Filters additional arrays using same indices as iteration array."""
        iterations = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
        array = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        expected_indices = [0, 1, 3, 4]

        result = latest_iterations(iterations, array)
        assert len(result) == 2
        assert_allclose(result[0], iterations[expected_indices])
        assert_allclose(result[1], array[expected_indices])

    def test_rejects_mismatched_lengths(self):
        """Raises ValueError when additional arrays have different lengths."""
        iterations = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
        array = np.array([0.0, 0.1])  # Shorter than iterations

        with pytest.raises(ValueError):
            latest_iterations(iterations, array)

    def test_handles_empty_input(self):
        """Returns empty tuple for empty input array."""
        iterations = np.array([])
        result = latest_iterations(iterations)
        assert len(result) == 1
        assert_allclose(result, (iterations,))


class TestExpectedUcb:
    """Tests for expected_ucb function."""

    def test_uses_float64_for_minimize(self, monkeypatch):
        class DummyRegressor:
            def __init__(self):
                self.seen_dtypes = []

            def predict(self, X, return_std=True):
                self.seen_dtypes.append(X.dtype)
                centered = X.astype(np.float32) - np.float32(0.2)
                mu = (centered**2).sum(axis=1).astype(np.float32)
                std = np.full_like(mu, 0.1, dtype=np.float32)
                return mu, std

        class DummySpace:
            def __init__(self):
                self.bounds = np.array(
                    [(np.float32(0.0), np.float32(1.0))], dtype=np.float32
                )

            def transform(self, X):
                return np.asarray(X, dtype=np.float32)

            def inverse_transform(self, X):
                return X.astype(np.float64)

            def rvs(self, n_random_starts, random_state=None):
                rng = np.random.default_rng(random_state)
                samples = rng.uniform(
                    0.0, 1.0, size=(n_random_starts, len(self.bounds))
                ).astype(np.float32)
                return samples

        reg = DummyRegressor()
        space = DummySpace()
        res = SimpleNamespace(
            models=[reg], space=space, x=np.array([0.5], dtype=np.float32)
        )

        captured = {}

        original_minimize = utils_module.minimize

        def spy_minimize(func, x0, bounds=None, **kwargs):
            captured["x0_dtype"] = np.asarray(x0).dtype
            captured["bounds"] = bounds
            return original_minimize(func, x0=x0, bounds=bounds, **kwargs)

        monkeypatch.setattr(utils_module, "minimize", spy_minimize)

        x_opt, fun = expected_ucb(res, n_random_starts=0)

        assert isinstance(x_opt, np.ndarray)
        assert x_opt.shape == (1,)
        assert x_opt.dtype == np.float64
        assert np.isfinite(fun)
        assert reg.seen_dtypes, "predict never called"
        assert all(dtype == np.float64 for dtype in reg.seen_dtypes)
        assert captured["x0_dtype"] == np.float64
        assert captured["bounds"] is not None
        assert (
            np.asarray(captured["bounds"], dtype=np.float64).dtype == np.float64
        )


class TestParseTimecontrol:
    """Tests for parse_timecontrol function."""

    def test_simple(self):
        """Test parsing simple time control."""
        result = parse_timecontrol("60")
        assert result == (Decimal("60"),)

    def test_with_increment(self):
        """Test parsing time control with increment."""
        result = parse_timecontrol("60+3")
        assert result == (Decimal("60"), Decimal("3"))

    def test_decimal(self):
        """Test parsing decimal time control."""
        result = parse_timecontrol("10.5+0.5")
        assert result == (Decimal("10.5"), Decimal("0.5"))


class TestTimeControl:
    """Tests for TimeControl class."""

    def test_from_string_simple(self):
        """Test TimeControl.from_string with simple time."""
        tc = TimeControl.from_string("60")
        assert tc.time == Decimal("60")
        assert tc.increment == Decimal("0")

    def test_from_string_with_increment(self):
        """Test TimeControl.from_string with increment."""
        tc = TimeControl.from_string("60+3")
        assert tc.time == Decimal("60")
        assert tc.increment == Decimal("3")

    def test_str_no_increment(self):
        """Test TimeControl.__str__ with no increment."""
        tc = TimeControl(time=Decimal("60"), increment=Decimal("0"))
        assert str(tc) == "60"

    def test_str_with_increment(self):
        """Test TimeControl.__str__ with increment."""
        tc = TimeControl(time=Decimal("60"), increment=Decimal("3"))
        assert str(tc) == "60+3"

    def test_str_small_increment(self):
        """Test TimeControl.__str__ with very small increment."""
        tc = TimeControl(time=Decimal("60"), increment=Decimal("0.0000000001"))
        assert str(tc) == "60"


class TestTimeControlBag:
    """Tests for TimeControlBag class."""

    def test_initialization(self):
        """Test TimeControlBag initialization."""
        tcs = [TimeControl.from_string("60"), TimeControl.from_string("120")]
        bag = TimeControlBag(tcs, bag_size=10)
        assert bag.bag_size == 10
        assert len(bag.tcs) == 2
        assert bag.bag == []

    def test_custom_probabilities(self):
        """Test TimeControlBag with custom probabilities."""
        tcs = [TimeControl.from_string("60"), TimeControl.from_string("120")]
        p = [0.7, 0.3]
        bag = TimeControlBag(tcs, bag_size=10, p=p)
        assert bag.p[0] == 0.7
        assert bag.p[1] == 0.3

    def test_next_tc(self):
        """Test TimeControlBag.next_tc returns time controls."""
        np.random.seed(42)
        tcs = [TimeControl.from_string("60"), TimeControl.from_string("120")]
        bag = TimeControlBag(tcs, bag_size=20)
        tc = bag.next_tc()
        assert tc in tcs

    def test_regenerates_bag(self):
        """Test TimeControlBag.next_tc regenerates bag when empty."""
        np.random.seed(42)
        tcs = [TimeControl.from_string("60"), TimeControl.from_string("120")]
        bag = TimeControlBag(tcs, bag_size=20)
        retrieved = []
        for _ in range(50):
            tc = bag.next_tc()
            retrieved.append(tc)
        assert len(retrieved) == 50
        assert all(tc in tcs for tc in retrieved)


class TestConfidenceToMult:
    """Tests for confidence_to_mult function."""

    def test_zero(self):
        """Test confidence_to_mult with 0 confidence."""
        result = confidence_to_mult(0)
        assert result == 0.0

    def test_one(self):
        """Test confidence_to_mult with 1 confidence."""
        result = confidence_to_mult(1)
        assert np.isinf(result)
        assert result > 0

    def test_0_95(self):
        """Test confidence_to_mult with 0.95 confidence."""
        result = confidence_to_mult(0.95)
        assert pytest.approx(result, 0.001) == 1.96

    def test_0_68(self):
        """Test confidence_to_mult with 0.68 confidence (1 sigma)."""
        result = confidence_to_mult(0.68)
        assert pytest.approx(result, 0.01) == 1.0

    def test_invalid_low(self):
        """Test confidence_to_mult raises ValueError for negative confidence."""
        with pytest.raises(
            ValueError, match="Confidence level must be in the range"
        ):
            confidence_to_mult(-0.1)

    def test_invalid_high(self):
        """Test confidence_to_mult raises ValueError for confidence > 1."""
        with pytest.raises(
            ValueError, match="Confidence level must be in the range"
        ):
            confidence_to_mult(1.1)

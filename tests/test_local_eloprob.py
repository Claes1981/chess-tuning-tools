"""Comprehensive tests for elo_to_prob and prob_to_elo functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tune.local import elo_to_prob, prob_to_elo


class TestEloToProb:
    """Tests for elo_to_prob function."""

    def test_converts_zero_elo_to_probability_0_5(self):
        """Zero Elo should convert to exactly 0.5 probability."""
        elo = np.array([0.0])
        result = elo_to_prob(elo)
        assert_allclose(result, np.array([0.5]))

    def test_converts_small_positive_elo_to_probability_greater_than_0_5(self):
        """Small positive Elo should convert to probability > 0.5."""
        elo = np.array([1.0])
        result = elo_to_prob(elo)
        assert result[0] > 0.5
        assert result[0] < 1.0

    def test_converts_small_negative_elo_to_probability_less_than_0_5(self):
        """Small negative Elo should convert to probability < 0.5."""
        elo = np.array([-1.0])
        result = elo_to_prob(elo)
        assert result[0] < 0.5
        assert result[0] > 0.0

    def test_converts_large_positive_elo_to_probability_near_or_at_1(self):
        """Large positive Elo should convert to probability near or at 1."""
        elo = np.array([100.0])
        result = elo_to_prob(elo)
        assert result[0] >= 0.99
        assert result[0] <= 1.0

    def test_converts_large_negative_elo_to_probability_near_or_at_0(self):
        """Large negative Elo should convert to probability near or at 0."""
        elo = np.array([-100.0])
        result = elo_to_prob(elo)
        assert result[0] <= 0.01
        assert result[0] >= 0.0

    def test_converts_multiple_elo_values(self):
        """Should correctly convert multiple Elo values at once."""
        elo = np.array([-2.0, 0.0, 2.0])
        result = elo_to_prob(elo)
        assert len(result) == 3
        assert result[0] < result[1] < result[2]
        assert_allclose(result[1], 0.5)

    def test_symmetry_around_zero(self):
        """Elo values should be symmetric around zero."""
        elo_pos = np.array([2.0])
        elo_neg = np.array([-2.0])
        prob_pos = elo_to_prob(elo_pos)
        prob_neg = elo_to_prob(elo_neg)
        assert_allclose(prob_pos, 1.0 - prob_neg)

    def test_monotonic_increase(self):
        """Probability should monotonically increase with Elo."""
        elo = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = elo_to_prob(elo)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_probability_bounds_inclusive(self):
        """All probabilities should be in [0, 1] interval (inclusive)."""
        elo = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        result = elo_to_prob(elo)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_custom_k_parameter(self):
        """Should respect custom k parameter for scaling."""
        elo = np.array([2.0])  # Use smaller elo to avoid saturation
        result_default = elo_to_prob(elo, k=4.0)
        result_custom = elo_to_prob(elo, k=2.0)
        # Different k should give different results
        assert result_default[0] != result_custom[0]

    def test_k_parameter_affects_scale(self):
        """Larger k should give probability closer to 0.5 for same elo."""
        elo = np.array([2.0])
        result_small_k = elo_to_prob(elo, k=1.0)
        result_large_k = elo_to_prob(elo, k=10.0)
        # With larger k, the probability should be closer to 0.5
        assert abs(result_large_k[0] - 0.5) < abs(result_small_k[0] - 0.5)

    def test_zero_array(self):
        """Should handle array of zeros."""
        elo = np.zeros(5)
        result = elo_to_prob(elo)
        assert_allclose(result, np.ones(5) * 0.5)

    def test_returns_ndarray(self):
        """Should return numpy ndarray."""
        elo = np.array([0.0, 2.0])
        result = elo_to_prob(elo)
        assert isinstance(result, np.ndarray)

    def test_preserves_dtype_float64(self):
        """Should preserve float64 dtype."""
        elo = np.array([0.0, 2.0], dtype=np.float64)
        result = elo_to_prob(elo)
        assert result.dtype == np.float64

    def test_raises_value_error_for_non_positive_k(self):
        """Should raise ValueError when k <= 0."""
        elo = np.array([2.0])
        with pytest.raises(ValueError, match="k must be positive"):
            elo_to_prob(elo, k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            elo_to_prob(elo, k=-1)

    def test_elo_100_with_k_4_gives_exactly_1(self):
        """Verify actual behavior: elo=100.0 with k=4.0 gives prob=1.0."""
        elo = np.array([100.0])
        result = elo_to_prob(elo, k=4.0)
        assert result[0] == 1.0


class TestProbToElo:
    """Tests for prob_to_elo function."""

    def test_converts_probability_0_5_to_zero_elo(self):
        """Probability 0.5 should convert to exactly 0 Elo."""
        prob = np.array([0.5])
        result = prob_to_elo(prob)
        assert_allclose(result, np.array([0.0]))

    def test_converts_probability_greater_than_0_5_to_positive_elo(self):
        """Probability > 0.5 should convert to positive Elo."""
        prob = np.array([0.75])
        result = prob_to_elo(prob)
        assert result[0] > 0.0

    def test_converts_probability_less_than_0_5_to_negative_elo(self):
        """Probability < 0.5 should convert to negative Elo."""
        prob = np.array([0.25])
        result = prob_to_elo(prob)
        assert result[0] < 0.0

    def test_converts_probability_near_1_to_positive_elo(self):
        """Probability near 1 should convert to positive Elo."""
        prob = np.array([0.99])
        result = prob_to_elo(prob)
        assert result[0] > 0.0

    def test_converts_probability_near_0_to_negative_elo(self):
        """Probability near 0 should convert to negative Elo."""
        prob = np.array([0.01])
        result = prob_to_elo(prob)
        assert result[0] < 0.0

    def test_converts_multiple_probabilities(self):
        """Should correctly convert multiple probabilities at once."""
        prob = np.array([0.25, 0.5, 0.75])
        result = prob_to_elo(prob)
        assert len(result) == 3
        assert result[0] < result[1] < result[2]
        assert_allclose(result[1], 0.0)

    def test_symmetry_around_0_5(self):
        """Probabilities should be symmetric around 0.5."""
        prob_low = np.array([0.25])
        prob_high = np.array([0.75])
        elo_low = prob_to_elo(prob_low)
        elo_high = prob_to_elo(prob_high)
        assert_allclose(elo_low, -elo_high)

    def test_monotonic_increase(self):
        """Elo should monotonically increase with probability."""
        prob = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = prob_to_elo(prob)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_inverse_of_eloprob_for_reasonable_values(self):
        """Should be inverse of elo_to_prob for non-saturated values."""
        elo_original = np.array([-2.0, 0.0, 2.0])
        prob = elo_to_prob(elo_original)
        elo_restored = prob_to_elo(prob)
        assert_allclose(elo_original, elo_restored)

    def test_custom_k_parameter(self):
        """Should respect custom k parameter for scaling."""
        prob = np.array([0.75])
        result_default = prob_to_elo(prob, k=4.0)
        result_custom = prob_to_elo(prob, k=2.0)
        # Different k should give different results
        assert result_default[0] != result_custom[0]

    def test_k_parameter_consistency(self):
        """k parameter should be consistent with elo_to_prob."""
        elo = np.array([2.0])  # Use smaller elo to avoid saturation
        prob = elo_to_prob(elo, k=4.0)
        elo_restored = prob_to_elo(prob, k=4.0)
        assert_allclose(elo, elo_restored)

    def test_probability_0_returns_negative_inf(self):
        """Probability of 0 should return -inf."""
        prob = np.zeros(5)
        result = prob_to_elo(prob)
        assert np.all(np.isinf(result))
        assert np.all(result < 0)

    def test_probability_1_returns_nan(self):
        """Probability of 1 should return NaN (log10 of infinity is undefined)."""
        prob = np.ones(5)
        result = prob_to_elo(prob)
        assert np.all(np.isnan(result))

    def test_returns_ndarray(self):
        """Should return numpy ndarray."""
        prob = np.array([0.5, 0.75])
        result = prob_to_elo(prob)
        assert isinstance(result, np.ndarray)

    def test_preserves_dtype_float64(self):
        """Should preserve float64 dtype."""
        prob = np.array([0.5, 0.75], dtype=np.float64)
        result = prob_to_elo(prob)
        assert result.dtype == np.float64

    def test_raises_value_error_for_non_positive_k(self):
        """Should raise ValueError when k <= 0."""
        prob = np.array([0.75])
        with pytest.raises(ValueError, match="k must be positive"):
            prob_to_elo(prob, k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            prob_to_elo(prob, k=-1)

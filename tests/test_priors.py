import numpy as np
from pytest import approx, raises

from tune.priors import create_priors, make_invgamma_prior, roundflat


class TestRoundFlat:
    """Tests for the roundflat function behavior."""

    def test_small_positive_input(self):
        """Test that roundflat returns approximately zero for small positive input."""
        result = roundflat(0.3)
        assert result == approx(0.0, abs=1e-6)

    def test_exactly_zero(self):
        """Test that roundflat returns negative infinity for exactly zero."""
        result = roundflat(0.0)
        assert result == -np.inf

    def test_negative_input(self):
        """Test that roundflat returns negative infinity for negative input."""
        result = roundflat(-1.0)
        assert result == -np.inf


class TestMakeInvGammaPrior:
    """Tests for the make_invgamma_prior function behavior and validation."""

    def test_default_parameters(self):
        """Test that default inverse gamma prior has expected parameters."""
        prior = make_invgamma_prior()
        assert prior.kwds["a"] == approx(8.919240823584246)
        assert prior.kwds["scale"] == approx(1.7290248731437994)

    def test_rejects_negative_lower_bound(self):
        """Test that ValueError is raised when lower_bound is negative."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(lower_bound=-1e-10)
        assert "bounds cannot be equal to or smaller than 0" in str(
            exc_info.value
        )

    def test_rejects_negative_upper_bound(self):
        """Test that ValueError is raised when upper_bound is negative."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(upper_bound=-1e-10)
        assert "bounds cannot be equal to or smaller than 0" in str(
            exc_info.value
        )

    def test_rejects_inconsistent_bounds(self):
        """Test that ValueError is raised when bounds are inconsistent."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(lower_bound=0.5, upper_bound=0.1)
        assert (
            "Lower bound needs to be strictly smaller than the upper bound"
            in str(exc_info.value)
        )


class TestCreatePriors:
    """Tests for the create_priors function behavior and validation."""

    def test_default_prior_count(self):
        """Test that default prior count is correct for given parameters."""
        priors = create_priors(n_parameters=3)
        assert len(priors) == 5

    def test_log_probabilities_at_value_two(self):
        """Test log probabilities of all priors at value 2.0."""
        expected_values = [
            -1.536140897416146,
            -23.620792572134874,
            -23.620792572134874,
            -23.620792572134874,
            -10262570.41553909,
        ]
        priors = create_priors(n_parameters=3)

        for i, expected in enumerate(expected_values):
            assert priors[i](2.0) == approx(expected)

    def test_rejects_zero_signal_scale(self):
        """Test that ValueError is raised when signal_scale is zero."""
        with raises(ValueError) as exc_info:
            create_priors(n_parameters=3, signal_scale=0.0)
        assert "The signal scale needs to be strictly positive" in str(
            exc_info.value
        )

    def test_rejects_zero_noise_scale(self):
        """Test that ValueError is raised when noise_scale is zero."""
        with raises(ValueError) as exc_info:
            create_priors(n_parameters=3, noise_scale=0.0)
        assert "The noise scale needs to be strictly positive" in str(
            exc_info.value
        )

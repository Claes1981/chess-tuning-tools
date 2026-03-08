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

    def test_rejects_negative_signal_scale(self):
        """Test that ValueError is raised when signal_scale is negative."""
        with raises(ValueError) as exc_info:
            create_priors(n_parameters=3, signal_scale=-1.0)
        assert "The signal scale needs to be strictly positive" in str(
            exc_info.value
        )

    def test_rejects_negative_noise_scale(self):
        """Test that ValueError is raised when noise_scale is negative."""
        with raises(ValueError) as exc_info:
            create_priors(n_parameters=3, noise_scale=-1.0)
        assert "The noise scale needs to be strictly positive" in str(
            exc_info.value
        )

    def test_single_parameter(self):
        """Test create_priors with single parameter."""
        priors = create_priors(n_parameters=1)
        assert len(priors) == 3

    def test_many_parameters(self):
        """Test create_priors with many parameters."""
        priors = create_priors(n_parameters=10)
        assert len(priors) == 12

    def test_custom_bounds(self):
        """Test create_priors with custom lengthscale bounds."""
        priors = create_priors(
            n_parameters=2,
            lengthscale_lower_bound=0.05,
            lengthscale_upper_bound=1.0,
        )
        assert len(priors) == 4


class TestRoundFlatExtended:
    """Extended tests for roundflat function."""

    def test_boundary_d_low(self):
        """Test roundflat at d_low boundary returns approximately -2."""
        result = roundflat(0.005)
        assert result == approx(-2.0, abs=0.5)

    def test_boundary_d_high(self):
        """Test roundflat at d_high boundary returns approximately -2."""
        result = roundflat(1.2)
        assert result == approx(-2.0, abs=0.5)

    def test_very_small_positive(self):
        """Test roundflat with very small positive value."""
        result = roundflat(1e-10)
        assert result < -100

    def test_very_large_value(self):
        """Test roundflat with very large value."""
        result = roundflat(1e10)
        assert result < -100

    def test_custom_boundaries(self):
        """Test roundflat with custom boundary parameters."""
        result = roundflat(0.5, a_low=1.0, a_high=10.0, d_low=0.1, d_high=2.0)
        assert result > -10


class TestMakeInvGammaPriorExtended:
    """Extended tests for make_invgamma_prior."""

    def test_rejects_zero_lower_bound(self):
        """Test that ValueError is raised when lower_bound is zero."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(lower_bound=0)
        assert "bounds cannot be equal to or smaller than 0" in str(
            exc_info.value
        )

    def test_rejects_zero_upper_bound(self):
        """Test that ValueError is raised when upper_bound is zero."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(upper_bound=0)
        assert "bounds cannot be equal to or smaller than 0" in str(
            exc_info.value
        )

    def test_rejects_equal_bounds(self):
        """Test that ValueError is raised when bounds are equal."""
        with raises(ValueError) as exc_info:
            make_invgamma_prior(lower_bound=0.5, upper_bound=0.5)
        assert "Lower bound needs to be strictly smaller" in str(exc_info.value)

    def test_custom_bounds(self):
        """Test make_invgamma_prior with custom bounds."""
        prior = make_invgamma_prior(lower_bound=0.01, upper_bound=2.0)
        assert prior is not None
        assert "a" in prior.kwds
        assert "scale" in prior.kwds

    def test_ppf_values(self):
        """Test that ppf at 0.01 and 0.99 match bounds."""
        lower, upper = 0.1, 0.5
        prior = make_invgamma_prior(lower_bound=lower, upper_bound=upper)
        ppf_01 = prior.ppf(0.01)
        ppf_99 = prior.ppf(0.99)
        assert ppf_01 == approx(lower, rel=0.1)
        assert ppf_99 == approx(upper, rel=0.1)

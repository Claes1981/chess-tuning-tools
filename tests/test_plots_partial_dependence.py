"""Tests for partial_dependence function in tune/plots.py."""

import os
import sys

import numpy as np

_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
from conftest import (
    MockGPModel,
    MockPolynomialFeatures,
    MockRegression,
    MockSpace,
)
from tune.plots import partial_dependence


class TestPartialDependence:
    """Tests for the partial_dependence function."""

    def test_1d_basic(self):
        """Test basic 1D partial dependence."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            n_samples=50,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20
        assert xi.shape == (20,)
        # yi is returned as a list for 1D case
        assert isinstance(yi, list)

    def test_1d_with_confidence_interval(self):
        """Test 1D partial dependence with confidence interval."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, yi_std = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            plot_confidence_interval_width=True,
            n_samples=50,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20
        assert len(yi_std) == 20

    def test_2d_basic(self):
        """Test basic 2D partial dependence."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, zi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=1,
            n_samples=50,
            n_points=15,
        )

        assert len(xi) == 15
        assert len(yi) == 15
        assert zi.shape == (15, 15)

    def test_2d_with_confidence_interval(self):
        """Test 2D partial dependence with confidence interval."""
        space = MockSpace(n_dims=3)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi, zi, zi_std = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=1,
            plot_confidence_interval_width=True,
            n_samples=50,
            n_points=15,
        )

        assert len(xi) == 15
        assert len(yi) == 15
        assert zi.shape == (15, 15)
        assert zi_std.shape == (15, 15)

    def test_higher_n_samples(self):
        """Test with higher number of samples."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            n_samples=100,
            n_points=20,
        )

        assert len(xi) == 20
        assert len(yi) == 20

    def test_higher_n_points(self):
        """Test with higher number of points."""
        space = MockSpace(n_dims=2)
        model = MockGPModel()
        regression = MockRegression()
        poly_features = MockPolynomialFeatures()

        xi, yi = partial_dependence(
            space=space,
            model=model,
            regression_object=regression,
            polynomial_features_object=poly_features,
            i=0,
            j=None,
            n_samples=50,
            n_points=50,
        )

        assert len(xi) == 50
        assert len(yi) == 50

import numpy as np
import pytest

from tune.local import initialize_data


class TestInitializeData:
    """Tests for initialize_data function."""

    def test_without_resume(self):
        """Test that initialize_data creates new empty structures when resume=False."""
        (
            X,
            y,
            noise,
            iteration,
            optima,
            performance,
            round,
            counts_array,
            point,
        ) = initialize_data(
            parameter_ranges=[(0.0, 1.0)],
            data_path=None,
            resume=False,
        )

        assert len(X) == 0
        assert len(y) == 0
        assert len(noise) == 0
        assert iteration == 0
        assert len(optima) == 0
        assert len(performance) == 0

        # Verify they are independent lists
        X.append(0)
        assert len(X) == 1
        assert len(y) == 0
        assert len(noise) == 0

    def test_ignores_file_when_not_resuming(self, tmp_path):
        """Test that initialize_data ignores existing file when resume=False."""
        testfile = tmp_path / "data.npz"
        X_in = np.array([[0.0], [0.5], [1.0]])
        y_in = np.array([1.0, -1.0, 0.0])
        noise_in = np.array([0.3, 0.2, 0.5])
        optima_in = np.array([[0.3]])
        performance_in = np.array([[2.0, 30.0, 20.0]])
        iteration_in = np.uint16(5)
        np.savez_compressed(
            testfile,
            X_in,
            y_in,
            noise_in,
            optima_in,
            performance_in,
            iteration_in,
        )

        X, _, _, _, _, _, _, _, _ = initialize_data(
            parameter_ranges=[(0.0, 1.0)],
            data_path=testfile,
            resume=False,
        )
        assert len(X) == 0

    def test_loads_saved_data(self, tmp_path):
        """Test that initialize_data correctly loads saved data when resume=True."""
        testfile = tmp_path / "data.npz"
        X_in = np.array([[0.0], [0.5], [1.0]])
        y_in = np.array([1.0, -1.0, 0.0])
        noise_in = np.array([0.3, 0.2, 0.5])
        optima_in = np.array([[0.3]])
        performance_in = np.array([[2.0, 30.0, 20.0]])
        iteration_in = np.uint16(5)
        np.savez_compressed(
            testfile,
            X_in,
            y_in,
            noise_in,
            optima_in,
            performance_in,
            iteration_in,
        )

        (
            X,
            y,
            noise,
            iteration,
            optima,
            performance,
            round,
            counts_array,
            point,
        ) = initialize_data(
            parameter_ranges=[(0.0, 1.0)],
            data_path=testfile,
            resume=True,
        )

        assert int(iteration) == 5
        assert np.allclose(X, X_in)
        assert np.allclose(y, y_in)
        assert np.allclose(noise, noise_in)
        assert np.allclose(optima, optima_in)
        assert np.allclose(performance, performance_in)

    def test_filters_by_parameter_range(self, tmp_path):
        """Test that initialize_data filters data based on reduced parameter ranges."""
        testfile = tmp_path / "data.npz"
        X_in = np.array([[0.0], [0.5], [1.0]])
        y_in = np.array([1.0, -1.0, 0.0])
        noise_in = np.array([0.3, 0.2, 0.5])
        optima_in = np.array([[0.3]])
        performance_in = np.array([[2.0, 30.0, 20.0]])
        iteration_in = np.uint16(5)
        np.savez_compressed(
            testfile,
            X_in,
            y_in,
            noise_in,
            optima_in,
            performance_in,
            iteration_in,
        )

        X, y, noise, iteration, _, _, _, _, _ = initialize_data(
            parameter_ranges=[(0.0, 0.5)],
            data_path=testfile,
            resume=True,
        )

        assert int(iteration) == 5
        assert np.allclose(X, np.array([[0.0], [0.5]]))
        assert np.allclose(y, np.array([1.0, -1.0]))
        assert np.allclose(noise, np.array([0.3, 0.2]))

    def test_raises_error_on_dimension_mismatch(self, tmp_path):
        """Test that ValueError is raised when saved data has different dimensions."""
        testfile = tmp_path / "data.npz"
        X_in = np.array([[0.0], [0.5], [1.0]])
        y_in = np.array([1.0, -1.0, 0.0])
        noise_in = np.array([0.3, 0.2, 0.5])
        optima_in = np.array([[0.3]])
        performance_in = np.array([[2.0, 30.0, 20.0]])
        iteration_in = np.uint16(5)
        np.savez_compressed(
            testfile,
            X_in,
            y_in,
            noise_in,
            optima_in,
            performance_in,
            iteration_in,
        )

        with pytest.raises(ValueError):
            _ = initialize_data(
                parameter_ranges=[(0.0, 1.0)] * 2,
                data_path=testfile,
                resume=True,
            )

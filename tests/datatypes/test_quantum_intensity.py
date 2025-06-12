"""Unit tests for the QuantumState class."""

import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qiskit_mps_initializer.datatypes import QuantumIntensity, QuantumState


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=10),
        elements=st.floats(
            min_value=1e-3, max_value=1e5, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_QuantumIntensity_using_nparrays(data: npt.NDArray[np.floating]) -> None:
    """Test the QuantumIntensity class using random numpy arrays."""

    # print(data)

    # Create an instance of PhasePreparedSignal
    intensity = QuantumIntensity.from_dense_data(data)

    # Check if the instance is created successfully
    assert isinstance(intensity, QuantumIntensity)

    # Validate the properties
    assert isinstance(intensity.state, QuantumState)
    assert intensity.alpha == np.sum(data)
    assert np.allclose(np.abs(intensity.wavefunction) ** 2 * intensity.alpha, data)
    assert intensity.size == len(data)

"""Unit tests for the QuantumState class."""

import numpy as np
import numpy.typing as npt
import qiskit
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qiskit_mps_initializer.datatypes import QuantumState


@given(
    arrays(
        dtype=np.complex128,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.complex_numbers(
            min_magnitude=0.01, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_QuantumState_using_nparrays(data: npt.NDArray[np.complex128]) -> None:
    """Test the QuantumState class using random numpy arrays."""

    # Create an instance of QuantumState
    state = QuantumState.from_dense_data(data=data, normalize=True)

    # Check if the instance is created successfully
    assert isinstance(state, QuantumState)

    # Validate the properties
    assert np.allclose(state.wavefunction, data / np.linalg.norm(data))
    assert state.num_qubits == int(np.log2(len(data)))
    assert state.size == len(data)


@given(
    arrays(
        dtype=np.complex128,
        shape=st.sampled_from([4, 8, 16]),
        elements=st.complex_numbers(
            min_magnitude=0.01,
            max_magnitude=1000,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
)
def test_QuantumState_mps_circuit(data: npt.NDArray[np.complex128]) -> None:
    """Test the MPS circuit generation of QuantumState."""

    # Create an instance of QuantumState
    state = QuantumState.from_dense_data(data=data, normalize=True)

    circuit = state.generate_mps_initializer_circuit(number_of_layers=1)
    assert isinstance(circuit, qiskit.QuantumCircuit)

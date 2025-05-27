import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qiskit_mps_initializer.datatypes.quantum_state import QuantumState
from qiskit_mps_initializer.utils.types import complex_array


@given(
    arrays(
        dtype=np.complex128,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.complex_numbers(
            min_magnitude=0.01, allow_nan=False, allow_infinity=False
        ),
    ),
    st.just(1),  # TODO: change this to a strategy for number_of_layers
)
def test_QuantumState_using_arrays(
    data: complex_array | list[float], number_of_layers: int
):
    # Create an instance of QuantumState
    quantum_state = QuantumState(data=data, number_of_layers=number_of_layers)

    # Check if the instance is created successfully
    assert isinstance(quantum_state, QuantumState)

    # Validate the properties
    assert quantum_state.wavefunction.size == len(data)
    assert quantum_state.num_qubits == int(np.log2(len(data)))
    assert quantum_state.size == len(data)

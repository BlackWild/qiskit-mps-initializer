import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qiskit_mps_initializer.datatypes.phase_prepared_signal import PhasePreparedSignal
from qiskit_mps_initializer.utils.types import real_array


@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=0.01, allow_nan=False, allow_infinity=False),
    )
)
def test_PhasePreparedSignal_using_lists(data: real_array):
    # Create an instance of PhasePreparedSignal
    signal = PhasePreparedSignal.from_dense_data(data, 1)

    # Validate the properties
    assert isinstance(signal, PhasePreparedSignal)
    assert signal.size == len(data)

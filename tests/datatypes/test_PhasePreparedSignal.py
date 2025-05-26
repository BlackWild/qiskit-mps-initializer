import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


# @given(st.lists(st.floats() | st.complex_numbers() | st.integers()), st.floats())
# def test_PhasePreparedSignal_using_lists(signal, alpha):
#     from qiskit_mps_initializer.datatypes.PhasePreparedSignal import PhasePreparedSignal

#     # Create an instance of PhasePreparedSignal
#     signal = PhasePreparedSignal(quantum_signal=signal, alpha)

#     # Check if the instance is created successfully
#     assert isinstance(signal, PhasePreparedSignal)

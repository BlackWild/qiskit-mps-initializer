from functools import cached_property
from typing import Self

import numpy as np
import pydantic
import qiskit

from qiskit_mps_initializer.datatypes.quantum_state import QuantumState
from qiskit_mps_initializer.utils.types import complex_array, real_array


class PhasePreparedSignal(pydantic.BaseModel):
    # Pydantic model configuration
    model_config = pydantic.ConfigDict(
        {
            "arbitrary_types_allowed": True,
        }
    )

    def __init__(self, quantum_state: QuantumState, alpha: float):
        super().__init__()
        self._corresponding_quantum_state = quantum_state
        self._original_alpha = alpha

    def by_multiplying_with(self, multiplier: float) -> "PhasePreparedSignal":
        return PhasePreparedSignal(
            quantum_state=self._corresponding_quantum_state,
            alpha=self._original_alpha * multiplier,
        )

    @classmethod
    def from_dense_data(cls, data: real_array, number_of_layers: int) -> Self:
        alpha, state = extract_alpha_and_phi_from_total_signals(data, number_of_layers)
        return cls(quantum_state=state, alpha=alpha)

    _corresponding_quantum_state: QuantumState
    _original_alpha: float

    _QUBIT_DIM: int = 2  # dimension of each qu-dit, we assume a qubit for now

    @pydantic.computed_field
    @cached_property
    def alpha(self) -> float:
        return self._original_alpha

    @pydantic.computed_field
    @cached_property
    def wavefunction(self) -> complex_array:
        return self._corresponding_quantum_state.wavefunction

    @pydantic.computed_field
    @cached_property
    def size(self) -> int:
        return self._corresponding_quantum_state.wavefunction.size

    @pydantic.computed_field
    @cached_property
    def num_qubits(self) -> int:
        return self._corresponding_quantum_state.num_qubits

    @pydantic.computed_field
    @cached_property
    def mps_initializer_circuit(self) -> qiskit.QuantumCircuit:
        return self._corresponding_quantum_state.mps_initializer_circuit

    # def get_alpha_and_quantum_state(self, multiplier: float = 1.0) -> tuple[float, QuantumState]:
    #     return self._original_alpha * multiplier, self._corresponding_quantum_state

    # multiplication with a scalar can be defined straightforwardly
    def __mul__(self, other: Self):
        if isinstance(other, int | float):
            return self.by_multiplying_with(other)
        else:
            raise ValueError("Multiplication is only defined for scalars.")


def extract_alpha_and_phi_from_total_signals(
    f: real_array, number_of_layers: int = 4
) -> tuple[float, QuantumState]:
    # check if all elements of f have the same sign
    if not all(
        [np.sign(f[0]) == np.sign(f[i]) or np.isclose(f[i], 0) for i in range(len(f))]  # type: ignore
    ):
        raise ValueError(
            "All elements of the signal vector must have the same sign. But got: "
            + str(f)
        )

    # normalization factor
    alpha = float(np.sum(f))

    # normalized signal
    normalized_signal = f / alpha

    # corresponding wavefunction
    phi = np.sqrt(normalized_signal)

    state = QuantumState(data=phi, number_of_layers=number_of_layers)

    return alpha, state

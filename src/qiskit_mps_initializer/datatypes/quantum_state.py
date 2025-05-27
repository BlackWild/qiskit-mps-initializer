from functools import cached_property

import numpy as np
import pydantic
import qiskit

from qiskit_mps_initializer.helpers.mps_technique import (
    multi_layered_circuit_for_non_approximated,
)
from qiskit_mps_initializer.utils.types import complex_array


class QuantumState(pydantic.BaseModel):
    # Pydantic model configuration
    model_config = pydantic.ConfigDict(
        {
            "arbitrary_types_allowed": True,
        }
    )

    def __init__(self, data: complex_array | list[float], number_of_layers: int):
        super().__init__()
        self._original_data = np.array(data)
        self._number_of_layers = number_of_layers

    _original_data: complex_array
    _number_of_layers: int

    _QUBIT_DIM: int = 2  # dimension of each qu-dit, we assume a qubit for now

    @pydantic.computed_field
    @cached_property
    def wavefunction(self) -> complex_array:
        return self._original_data / np.linalg.norm(self._original_data)

    @pydantic.computed_field
    @cached_property
    def size(self) -> int:
        return self.wavefunction.size

    @pydantic.computed_field
    @cached_property
    def num_qubits(self) -> int:
        return int(np.log2(self.size))

    @pydantic.computed_field
    @cached_property
    def mps_initializer_circuit(self) -> qiskit.QuantumCircuit:
        circuit = multi_layered_circuit_for_non_approximated(
            self.wavefunction, number_of_layers=self._number_of_layers
        )
        return circuit

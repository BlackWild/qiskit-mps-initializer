"""QuantumState."""

from typing import Self

import numpy as np
import numpy.typing as npt
import pydantic
import qiskit

from qiskit_mps_initializer.helpers.mps_technique import (
    multi_layered_circuit_for_non_approximated,
)
from qiskit_mps_initializer.utils.types import complex_array


class QuantumState(pydantic.BaseModel):
    """Represents a quantum state."""

    # Pydantic model configuration
    model_config = pydantic.ConfigDict(
        {
            "arbitrary_types_allowed": True,
        }
    )

    original_data: npt.NDArray[np.complex128]

    @classmethod
    def from_dense_data(cls, data: complex_array, normalize: bool = False) -> Self:
        """Initializes the QuantumState from the given dense data."""

        normalization_factor = np.linalg.norm(data)

        if not normalize and not np.isclose(normalization_factor, 1.0):
            raise ValueError(
                "The provided data is not normalized. Set `normalize=True` to normalize the wavefunction."
            )

        return cls(original_data=np.array(data, dtype=np.complex128))

    @pydantic.computed_field
    @property
    def _original_normalization_factor(self) -> np.floating:
        return np.linalg.norm(self.original_data)

    @pydantic.computed_field
    @property
    def wavefunction(self) -> npt.NDArray[np.complex128]:
        """Returns the normalized wavefunction of the quantum state."""
        return self.original_data / self._original_normalization_factor

    @pydantic.computed_field
    @property
    def size(self) -> int:
        """Returns the dimension of the quantum state."""
        return self.wavefunction.size

    @pydantic.computed_field
    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the quantum state."""
        return int(np.log2(self.size))

    def generate_mps_initializer_circuit(
        self, number_of_layers: int
    ) -> qiskit.QuantumCircuit:
        """Generates the MPS initializer circuit for the quantum state."""
        circuit = multi_layered_circuit_for_non_approximated(
            self.wavefunction, number_of_layers=number_of_layers
        )
        return circuit

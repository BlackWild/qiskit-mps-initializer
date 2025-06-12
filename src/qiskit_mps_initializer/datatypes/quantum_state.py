"""QuantumState."""

import numpy as np
import pydantic as pd
import pydantic_numpy.model as pdnm
import pydantic_numpy.typing as pdnt
import qiskit
import qiskit.circuit

from qiskit_mps_initializer.helpers.mps_technique import (
    multi_layered_circuit_for_non_approximated,
)
from qiskit_mps_initializer.utils.types import complex_array


class QuantumState(pdnm.NumpyModel):
    """Represents a quantum state."""

    original_data: pdnt.Np1DArrayComplex128
    """The original data of the quantum state."""

    @classmethod
    def from_dense_data(
        cls, data: complex_array, normalize: bool = False
    ) -> "QuantumState":
        """Initializes the QuantumState from the given dense data."""

        normalization_factor = np.linalg.norm(data)

        if not normalize and not np.isclose(normalization_factor, 1.0):
            raise ValueError(
                "The provided data is not normalized. Set `normalize=True` to normalize the wavefunction."
            )

        return cls(original_data=np.array(data, dtype=np.complex128))

    @pd.computed_field
    @property
    def _original_normalization_factor(self) -> float:
        return np.linalg.norm(self.original_data).astype(float)

    @pd.computed_field
    @property
    def wavefunction(self) -> pdnt.Np1DArrayComplex128:
        """The normalized wavefunction of the quantum state."""
        return self.original_data / self._original_normalization_factor

    @pd.computed_field
    @property
    def size(self) -> int:
        """The dimension of the quantum state."""
        return self.wavefunction.size

    @pd.computed_field
    @property
    def num_qubits(self) -> int:
        """The number of qubits required to represent the quantum state."""
        return np.log2(self.size).astype(int)

    def generate_mps_initializer_circuit(
        self, number_of_layers: int
    ) -> qiskit.circuit.QuantumCircuit:
        """Generates the MPS initializer circuit for the quantum state.

        Returns:
            QuantumCircuit: The MPS initializer circuit for the quantum state as a qiskit circuit.
        """
        circuit = multi_layered_circuit_for_non_approximated(
            self.wavefunction, number_of_layers=number_of_layers
        )
        return circuit

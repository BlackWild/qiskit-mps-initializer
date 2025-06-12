"""QuantumIntensity."""

import pydantic as pd
import pydantic_numpy.model as pdnm
import pydantic_numpy.typing as pdnt
import qiskit
import qiskit.circuit

from qiskit_mps_initializer.datatypes import QuantumState
from qiskit_mps_initializer.helpers.extractors import (
    extract_alpha_and_state_from_intensity_signal,
)
from qiskit_mps_initializer.utils.types import real_array


class QuantumIntensity(pdnm.NumpyModel):
    """Represents an intensity signal."""

    state: QuantumState
    """The quantum state corresponding to this intensity signal."""

    alpha: float
    """The alpha parameter of the intensity signal."""

    @classmethod
    def from_dense_data(cls, data: real_array) -> "QuantumIntensity":
        """Initializes a QuantumIntensity from the given dense data."""

        alpha, state_data = extract_alpha_and_state_from_intensity_signal(data)
        state = QuantumState.from_dense_data(data=state_data, normalize=False)

        return cls(state=state, alpha=alpha)

    @pd.computed_field
    @property
    def wavefunction(self) -> pdnt.Np1DArrayComplex128:
        """The normalized wavefunction of the quantum state."""
        return self.state.wavefunction

    @pd.computed_field
    @property
    def size(self) -> int:
        """The dimension of the quantum state."""
        return self.state.wavefunction.size

    @pd.computed_field
    @property
    def num_qubits(self) -> int:
        """The number of qubits required to represent the quantum state."""
        return self.state.num_qubits

    def generate_mps_initializer_circuit(
        self, number_of_layers: int
    ) -> qiskit.circuit.QuantumCircuit:
        """Generates the MPS initializer circuit for the quantum state."""
        return self.state.generate_mps_initializer_circuit(number_of_layers)

    # multiplication with a scalar can be defined straightforwardly
    def __mul__(self, other: int | float) -> "QuantumIntensity":
        """Defines the multiplication of the QuantumIntensity with a scalar."""
        if isinstance(other, int | float):  # type: ignore
            new_state = QuantumIntensity(state=self.state, alpha=self.alpha * other)
            return new_state
        else:
            raise ValueError("Multiplication is only defined for scalars.")

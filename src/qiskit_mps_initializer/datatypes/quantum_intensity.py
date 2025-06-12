"""QuantumIntensity."""

from typing import Self

import numpy as np
import numpy.typing as npt
import pydantic
import qiskit

from qiskit_mps_initializer.datatypes import QuantumState
from qiskit_mps_initializer.helpers.extractors import (
    extract_alpha_and_state_from_intensity_signal,
)
from qiskit_mps_initializer.utils.types import real_array


class QuantumIntensity(pydantic.BaseModel):
    """QuantumIntensity.

    Attributes:
        state: The quantum state.
        alpha: The alpha value.
    """

    state: QuantumState
    alpha: float

    # Pydantic model configuration
    model_config = pydantic.ConfigDict(
        {
            "arbitrary_types_allowed": True,
        }
    )

    @classmethod
    def from_dense_data(cls, data: real_array) -> Self:
        """Initializes the QuantumIntensity from the given dense data."""

        alpha, state_data = extract_alpha_and_state_from_intensity_signal(data)
        state = QuantumState.from_dense_data(data=state_data, normalize=False)

        return cls(state=state, alpha=alpha)

    @pydantic.computed_field
    @property
    def wavefunction(self) -> npt.NDArray[np.complex128]:
        """Returns the normalized wavefunction of the quantum state.

        Returns:
            (npt.NDArray[np.complex128]): The normalized wavefunction.
        """
        return self.state.wavefunction

    @pydantic.computed_field
    @property
    def size(self) -> int:
        """Returns the dimension of the quantum state."""
        return self.state.wavefunction.size

    @pydantic.computed_field
    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits required to represent the quantum state."""
        return self.state.num_qubits

    def generate_mps_initializer_circuit(
        self, number_of_layers: int
    ) -> qiskit.QuantumCircuit:
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

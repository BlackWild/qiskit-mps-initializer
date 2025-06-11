import qiskit
import qiskit.circuit

from qiskit_mps_initializer.datatypes.quantum_state import QuantumState


class MPSInitializer:
    def __init__(self, state: QuantumState, number_of_layers: int | None = None):
        """Initialize the MPSInitializer gate with the number of qubits and the MPS state."""

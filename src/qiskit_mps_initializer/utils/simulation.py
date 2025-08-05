"""Helper functions for running simulations in qiskit."""

import qiskit
import qiskit.quantum_info
import qiskit_aer


def simulate_statevector(
    circuit: qiskit.QuantumCircuit,
) -> qiskit.quantum_info.Statevector:
    """Simulates a quantum circuit and returns the resulting statevector.

    Args:
        circuit (qiskit.QuantumCircuit): The circuit to simulate.

    Returns:
        qiskit.quantum_info.Statevector: The resulting statevector.
    """

    copy_circuit = circuit.copy()
    copy_circuit.save_statevector()

    simulator = qiskit_aer.AerSimulator(method="statevector")
    transpiled_circuit = qiskit.transpile(copy_circuit, backend=simulator)
    job = simulator.run(transpiled_circuit, shots=1)

    result = job.result()
    statevector = result.get_statevector()

    return statevector


def simulate_quantum_info(
    circuit: qiskit.QuantumCircuit,
) -> qiskit.quantum_info.Statevector:
    """Simulates a quantum circuit and returns the resulting statevector.

    Args:
        circuit (qiskit.QuantumCircuit): The circuit to simulate.

    Returns:
        qiskit.quantum_info.Statevector: The resulting statevector.
    """

    unitary = qiskit.quantum_info.Operator.from_circuit(circuit).data

    statevector = qiskit.quantum_info.Statevector(unitary[:, 0])

    return statevector

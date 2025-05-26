import numpy as np
import qiskit
import qiskit.circuit
import quimb
import quimb.gates as gates
import quimb.tensor as qtn
import scipy

from qiskit_mps_initializer.utils.types import complex_array


def bond2_mps_approximation(psi: complex_array) -> qtn.MatrixProductState:
    if not np.isclose(np.linalg.norm(psi), 1.0):
        raise ValueError(
            "The state vector must be normalized. The norm was: "
            + str(np.linalg.norm(psi))
        )

    # create the bond-2 MPS approximation of the state vector
    # mps = qtn.MatrixProductState.from_dense(psi, max_bond=2)
    mps = qtn.MatrixProductState.from_dense(psi, max_bond=2, absorb="left")

    # ensure normalization
    mps.normalize()

    # ensure right-canonical form
    mps.right_canonicalize(inplace=True)

    # ensure left-physical-right order of the indices
    mps.permute_arrays(shape="lpr")

    # TODO: the following can probably be improved by constructing a final layer with only a few two-qubit gates instead of forcing all to be two-qubit

    # check if any of the bond sizes are 1
    bond_sizes = mps.bond_sizes()
    if any([bond_size != 2 for bond_size in bond_sizes]):
        raise ValueError(
            "The bond sizes of the MPS should be exactly 2. The bond sizes were: \n"
            + str(bond_sizes)
        )

    return mps


def G_matrices(mps: qtn.MatrixProductState) -> list[complex_array]:
    # TODO: things probably can be done more efficiently in terms of not transposing data around and working properly in the numpy realm

    G = []

    # G_first
    A0: quimb.tensor.Tensor = mps[0]  # type: ignore
    A0_vec = np.array([A0.data.flatten()])
    G0 = np.concatenate((A0_vec.T, scipy.linalg.null_space(A0_vec).conjugate()), axis=1)
    G.append(G0)

    # G_middle
    for i in range(1, mps.num_tensors - 1):
        Ai: quimb.tensor.Tensor = mps[i]  # type: ignore
        Ai_a_0 = Ai.data[0, :, :].flatten()
        Ai_a_1 = Ai.data[1, :, :].flatten()
        Gi_incomplete = np.array([Ai_a_0, Ai_a_1])
        Gi = np.concatenate(
            (Gi_incomplete.T, scipy.linalg.null_space(Gi_incomplete).conjugate()),
            axis=1,
        )
        Gi = Gi @ np.real(gates.SWAP)
        G.append(Gi)

    # G_last
    AN: quimb.tensor.Tensor = mps[-1]  # type: ignore
    G_last = AN.data.T
    G.append(G_last)

    # TODO: maybe also check the equivalence of the product of the G matrices with the original MPS

    return G


def one_layer_gates_for_bond2_approximated(
    G: list[complex_array],
) -> list[qiskit.circuit.library.UnitaryGate]:
    # this implicitly checks for the unitarity of the G matrices
    return [qiskit.circuit.library.UnitaryGate(Gi) for Gi in G]


def multi_layered_circuit_for_non_approximated(
    psi: complex_array, number_of_layers: int
) -> qiskit.QuantumCircuit:
    # check for normalization of psi
    if not np.isclose(np.linalg.norm(psi), 1.0):
        raise ValueError(
            "The state vector must be normalized. The norm was: "
            + str(np.linalg.norm(psi))
        )

    number_of_qubits = int(np.log2(len(psi)))
    if len(psi) != 2**number_of_qubits:
        raise ValueError("The state vector must have a size of 2^n.")

    # create a copy
    current_psi = np.copy(psi)
    current_psi = current_psi / np.linalg.norm(current_psi)

    # iteratively construct the layers
    layers = []
    for j in range(number_of_layers):
        mps = bond2_mps_approximation(current_psi)
        G = G_matrices(mps)

        current_layer_circuit = qiskit.QuantumCircuit(number_of_qubits)
        for i in range(len(G) - 1):
            current_layer_circuit.unitary(
                G[i], [number_of_qubits - 1 - i - 1, number_of_qubits - 1 - i]
            )
        current_layer_circuit.unitary(G[-1], [0])

        layers.append(current_layer_circuit)

    # the order of the construction of the layers is the reverse of the order of the application of them in the implementation
    layers.reverse()
    circuit = qiskit.QuantumCircuit(number_of_qubits)
    for layer in layers:
        circuit.compose(layer, inplace=True)

    print(
        "DEBUG LOG: MPS initializer generator was called. This log is for the purpose of reducing the number of calls to this function."
    )

    return circuit

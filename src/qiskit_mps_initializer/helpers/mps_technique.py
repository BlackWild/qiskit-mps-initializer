"""Helper functions for the MPS technique."""

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

    return mps


def G_matrices(mps: qtn.MatrixProductState) -> list[complex_array]:
    # TODO: things probably can be done more efficiently in terms of not transposing data around and working properly in the numpy realm

    # TODO: properly document this

    G = []

    # The following code calculates the first G matrix from the first A tensor.
    # The first A tensor is either of the shape (2, 1) or (2, 2).
    A0: qtn.Tensor = mps[0]  # type: ignore
    # In both cases, the following code creates the corresponding G matrix. Note that in case of a (2, 1) tensor, the flattened vector is of size 2 and the null space has 1 element, and in case of a (2, 2) tensor, the flattened vector is of size 4 and the null space has 3 elements.
    A0_vec = np.array([A0.data.flatten()])
    G0 = np.concatenate((A0_vec.T, scipy.linalg.null_space(A0_vec).conjugate()), axis=1)
    G.append(G0)

    # The following code calculates the middle G matrices from the A tensors.
    # The middle A tensors can be of the shape (2, 2, 2), (2, 2, 1), (1, 2, 2), or (1, 2, 1).
    for i in range(1, mps.num_tensors - 1):
        Ai: qtn.Tensor = mps[i]  # type: ignore
        Ai_a_0 = Ai.data[0, :, :].flatten()
        if Ai.data.shape[0] == 2:
            Ai_a_1 = Ai.data[1, :, :].flatten()
            Gi_incomplete = np.array([Ai_a_0, Ai_a_1])
        else:
            Gi_incomplete = np.array([Ai_a_0])

        Gi = np.concatenate(
            (Gi_incomplete.T, scipy.linalg.null_space(Gi_incomplete).conjugate()),
            axis=1,
        )

        if Ai.data.shape[2] == 2:
            Gi = Gi @ np.real(gates.SWAP)

        G.append(Gi)

    # The following code calculates the last G matrix from the last A tensor.
    # The last A tensor is of the shape (2, 2) or (1, 2).
    AN: quimb.tensor.Tensor = mps[-1]  # type: ignore
    if AN.data.shape[0] == 2:
        G_last = AN.data.T
        G.append(G_last)
    else:
        A_last_a_0 = AN.data[0, :].flatten()
        G_last_incomplete = np.array([A_last_a_0])
        G_last = np.concatenate(
            (
                G_last_incomplete.T,
                scipy.linalg.null_space(G_last_incomplete).conjugate(),
            ),
            axis=1,
        )
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
            if G[i].shape == (4, 4):
                current_layer_circuit.unitary(
                    G[i], [number_of_qubits - 1 - i - 1, number_of_qubits - 1 - i]
                )
            elif G[i].shape == (2, 2):
                current_layer_circuit.unitary(G[i], number_of_qubits - 1 - i)
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

"""Helper functions for the MPS technique."""

import numpy as np
import numpy.typing as npt
import pydantic_numpy.typing as pdnt
import qiskit
import qiskit.circuit
import qiskit.quantum_info
import quimb
import quimb.gates as gates
import quimb.tensor as qtn
import scipy


def bond2_mps_approximation(psi: npt.ArrayLike) -> qtn.MatrixProductState:
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


def G_matrices(mps: qtn.MatrixProductState) -> list[pdnt.Np2DArrayComplex64]:
    # make sure the maximum bond dimension is maximally 2 or there is only one tensor
    max_bond = mps.max_bond()
    if max_bond is None:  # there is no bond therefore there is only one site
        raise ValueError("This function does not support one-site MPS yet.")
    if max_bond > 2:
        raise ValueError("The maximum bond dimension of the MPS must be 2")

    # expand the bond dimension to 2 for easier handling of edge cases (when a bond dimension is 1)
    mps.expand_bond_dimension(2)

    # we start with an empty list of G matrices
    G = []

    # then we go for finding the matrices. if there is only one site, then we just have to find the final G matrix which is a one-qubit gate. otherwise, we first find the other g matrices which are two-qubit gates
    if mps.num_tensors > 1:
        tensor_data: pdnt.Np2DArrayComplex64 = mps[0].data.copy()  # type: ignore
        # the matrix reshaped as a 4x1 column vector
        A0_vec = tensor_data.reshape(4, 1)
        # the null space of the vector (we transpose the column vector to a row vector to get the null space as a 4x3 matrix)
        null_space = scipy.linalg.null_space(A0_vec.T)
        # the G matrix is the concatenation of the 4x1 vector and 4x3 null space matrix. note that the null space matrix is complex conjugated because A0_vec*null_space = 0 thus null_space actually already includes complex conjugated elements
        G0 = np.column_stack((A0_vec, null_space.conjugate()))

        G.append(G0)

        for i in range(1, mps.num_tensors - 1):
            Ai_a_0_data: pdnt.Np2DArrayComplex64 = mps[i].data[0, :, :].copy()  # type: ignore
            Ai_a_0 = Ai_a_0_data.reshape(4, 1)
            Ai_a_1_data: pdnt.Np2DArrayComplex64 = mps[i].data[1, :, :].copy()  # type: ignore
            Ai_a_1 = Ai_a_1_data.reshape(4, 1)

            if np.allclose(Ai_a_1, [0, 0, 0, 0]):
                Gi_incomplete = Ai_a_0
            else:
                Gi_incomplete = np.column_stack((Ai_a_0, Ai_a_1))

            null_space = scipy.linalg.null_space(Gi_incomplete.T)

            Gi = np.column_stack((Gi_incomplete, null_space.conjugate()))

            Gi = Gi @ np.real(gates.SWAP)

            G.append(Gi)

    # TODO: the following part still does not support only one site since the A tensor will just be a vector in this case with just one physical index. to check this out update the corresponding test in test_mps_technique.py to include one site only and fix this in code

    # the following code calculates the last G matrix from the last A tensor.
    # the last A tensor is of the shape (2, 2) or (1, 2).
    G_last_data: pdnt.Np2DArrayComplex64 = mps[-1].data.copy()  # type: ignore
    G_last = G_last_data.T
    # sometimes if it was (1, 2), the initial expansion to bond 2 creates a zero vector as the second column. the following code fixes this by substituting the zero vector with a vector orthogonal to the first column
    if np.allclose(G_last[:, 1], [0, 0]):
        new_vec_1 = scipy.linalg.null_space([G_last[:, 0].T.conjugate()])
        G_last = np.column_stack((G_last[:, 0], new_vec_1))

    G.append(G_last)

    return G


def one_layer_gates_for_bond2_approximated(
    G: list[pdnt.Np2DArrayComplex64],
) -> list[qiskit.circuit.library.UnitaryGate]:
    # this implicitly checks for the unitarity of the G matrices
    return [qiskit.circuit.library.UnitaryGate(Gi) for Gi in G]


def multi_layered_circuit_for_non_approximated(
    psi: pdnt.Np1DArrayComplex128, max_number_of_layers: int | None, atol: float = 1e-5
) -> tuple[qiskit.QuantumCircuit, bool]:
    # check for normalization of psi
    if not np.isclose(np.linalg.norm(psi), 1.0):
        raise ValueError(
            "The state vector must be normalized. The norm was: "
            + str(np.linalg.norm(psi))
        )

    number_of_qubits = int(np.log2(len(psi)))
    if len(psi) != 2**number_of_qubits:
        raise ValueError("The state vector must have of size of 2^n.")

    # create a copy
    current_psi = np.copy(psi)
    current_psi = current_psi / np.linalg.norm(current_psi)

    # zero state
    zero_state = np.zeros(len(psi), dtype=np.complex128)
    zero_state[0] = 1

    # iteratively construct the layers
    layers = []
    did_hit_atol = False
    number_of_layers = 0
    while True:
        if np.linalg.norm(current_psi - zero_state) < atol:
            did_hit_atol = True
            break
        if max_number_of_layers is not None and number_of_layers > max_number_of_layers:
            break

        mps = bond2_mps_approximation(current_psi)
        G = G_matrices(mps)

        current_layer_circuit = qiskit.QuantumCircuit(number_of_qubits)
        for i in range(len(G) - 1):
            if G[i].shape == (4, 4):
                current_layer_circuit.unitary(
                    G[i], [number_of_qubits - 1 - i - 1, number_of_qubits - 1 - i]
                )
            elif G[i].shape == (2, 2):
                current_layer_circuit.unitary(G[i], [number_of_qubits - 1 - i])
        current_layer_circuit.unitary(G[-1], [0])

        unitary: pdnt.Np2DArrayComplex64 = qiskit.quantum_info.Operator.from_circuit(
            current_layer_circuit
        ).data  # type: ignore

        current_psi = unitary.conjugate().T @ current_psi
        current_psi = current_psi / np.linalg.norm(current_psi)

        layers.append(current_layer_circuit)
        number_of_layers += 1

    # the order of the construction of the layers is the reverse of the order of the application of them in the implementation
    layers.reverse()
    circuit = qiskit.QuantumCircuit(number_of_qubits)
    for layer in layers:
        circuit.compose(layer, inplace=True)

    return circuit, did_hit_atol

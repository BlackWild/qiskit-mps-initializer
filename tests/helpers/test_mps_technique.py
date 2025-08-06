"""Test the MPS helper functions."""

import numpy as np
import numpy.typing as npt
import qiskit
import qiskit.quantum_info
import quimb.tensor as qtn
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qiskit_mps_initializer.helpers.mps_technique import (
    G_matrices,
    bond2_mps_approximation,
    multi_layered_circuit_for_non_approximated,
)
from qiskit_mps_initializer.utils import simulate_quantum_info


@given(number_of_sites=st.integers(min_value=2, max_value=8))
def test_bond2_mps_approximation(number_of_sites: int) -> None:
    """Tests that the bond-2 approximation of the MPS works as expected.

    To make ensure fair tests, we generate random mps states with max bond dimension 2.
    """
    random_mps_state = qtn.MPS_rand_state(
        L=number_of_sites, bond_dim=2, dtype="complex128", normalize=True
    )
    random_mps_data = random_mps_state.to_dense()

    tensor = bond2_mps_approximation(random_mps_data)

    approximated_data = tensor.to_dense()

    assert np.allclose(random_mps_data, approximated_data)


# TODO: test wether the indices are in correct order via internal quimb methods
# @given(number_of_sites=st.integers(min_value=2, max_value=8))
# def test_mps_index_permutation(number_of_sites: int) -> None:
#     random_mps_state = qtn.MPS_rand_state(
#         L=number_of_sites, bond_dim=2, dtype="complex128", normalize=True
#     )
#     random_mps_data = random_mps_state.to_dense()

#     tensor = bond2_mps_approximation(random_mps_data)

#     tensor.inner_inds


@given(number_of_sites=st.sampled_from([2, 3, 4, 5]))
def test_G_matrices_for_bond2(number_of_sites: int) -> None:
    # TODO: make this a hypothesis custom strategy because it created an insane amount of trouble to debug
    # Generate random mps state with bond dimension 2
    tensor = qtn.MPS_rand_state(
        L=number_of_sites, bond_dim=2, dtype="complex128", normalize=True
    )
    tensor.normalize()
    tensor.right_canonicalize(inplace=True)
    tensor.permute_arrays(shape="lpr")

    # Calculate the G matrices
    G = G_matrices(tensor)

    # Check that the number of G matrices is equal to the number of sites
    assert len(G) == number_of_sites

    # Check that the shapes of the G matrices are correct.
    # The first G matrix has shape (4, 4), and the last G matrix has shape (2, 2).
    for i in range(len(G) - 1):
        assert G[i].shape == (4, 4)
    assert G[-1].shape == (2, 2)

    # Check that the G matrices are unitary
    for Gi in G:
        assert np.allclose(np.matmul(Gi, Gi.conj().T), np.eye(Gi.shape[0]))

    # Generate the qiskit circuit of it and check if it initializes the objective state
    circuit = qiskit.QuantumCircuit(number_of_sites)
    for i in range(len(G) - 1):
        if G[i].shape == (4, 4):
            circuit.unitary(
                G[i], [number_of_sites - 1 - i - 1, number_of_sites - 1 - i]
            )
        elif G[i].shape == (2, 2):
            circuit.unitary(G[i], [number_of_sites - 1 - i])
    circuit.unitary(G[-1], [0])

    simulated_state = simulate_quantum_info(circuit)

    assert np.allclose(simulated_state.data, tensor.to_dense().flatten())


@given(
    arrays(
        dtype=np.complex128,
        shape=st.sampled_from([4, 8, 16]),
        elements=st.complex_numbers(
            min_magnitude=0.01, max_magnitude=1, allow_nan=False, allow_infinity=False
        ),
    ),
    st.floats(min_value=1e-4, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_multilayered_circuit_no_max_layer(
    psi: npt.NDArray[np.complex128], atol: float
) -> None:
    normalized_psi = psi / np.linalg.norm(psi)
    circuit, did_hit_atol = multi_layered_circuit_for_non_approximated(
        normalized_psi, None, atol
    )

    simulated_state = simulate_quantum_info(circuit)

    assert did_hit_atol, "Did not hit atol"
    assert np.allclose(simulated_state.data, normalized_psi, atol=atol), (
        "Did not create the correct state vector upto desired atol"
    )

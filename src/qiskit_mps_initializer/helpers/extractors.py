import numpy as np
import numpy.typing as npt

from qiskit_mps_initializer.utils.types import real_array


def extract_alpha_and_state_from_intensity_signal(
    f: real_array,
) -> tuple[float, npt.NDArray[np.complex128]]:
    """Extracts the alpha and the state from the intensity signal."""

    # check if all elements of f have the same sign
    if not all(
        [np.sign(f[0]) == np.sign(f[i]) or np.isclose(f[i], 0) for i in range(len(f))]
    ):
        raise ValueError("All elements of the signal vector must have the same sign.")

    # normalization factor
    alpha = np.sum(f)

    # normalized signal
    normalized_signal = f / alpha

    # corresponding wavefunction
    state_data = np.sqrt(normalized_signal)

    return alpha, state_data

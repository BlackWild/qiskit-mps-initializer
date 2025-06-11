import numpy as np
import numpy.typing as npt

real_array = npt.NDArray[np.floating] | list[float]
complex_array = npt.NDArray[np.complexfloating] | list[complex] | real_array

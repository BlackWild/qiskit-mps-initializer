from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy
import sympy

from qiskit_mps_initializer.utils.types import complex_array

# custom types

Sampling_Type = Literal["positive", "symmetric", "twos_complement"]
Symbolic_Function_Type = Callable[[sympy.Symbol], sympy.Expr]

# functions


def expression_to_array(
    expr: sympy.Expr,
    point: sympy.Symbol,
    sampling_period: float,
    number_of_indices: int,
    sampling_type: Sampling_Type,
) -> complex_array:
    # check if number_of_indices is a power of 2
    if not (number_of_indices & (number_of_indices - 1) == 0):
        raise ValueError("number_of_indices must be a power of 2")

    # create an array of points to
    match sampling_type:
        case "positive":
            gammas = np.arange(0, number_of_indices)
        case "symmetric":
            gammas = np.arange(-number_of_indices // 2, number_of_indices // 2)
        case "twos_complement":
            gammas = np.concatenate(
                (
                    np.arange(0, number_of_indices // 2),
                    np.arange(-number_of_indices // 2, 0),
                )
            )
        case _:
            raise ValueError(f"Invalid sampling type: {sampling_type}")

    points = gammas * sampling_period

    lambdified = sympy.lambdify(point, expr, "numpy")

    # evaluate the expression at each point
    values = lambdified(points)

    # return the values as a array
    return np.array(values)


def position_func_to_array(
    f: Symbolic_Function_Type,
    delta_x: float,
    number_of_samples: int,
    sampling_type: Sampling_Type,
) -> complex_array:
    # check if number_of_samples is a power of 2
    if not (number_of_samples & (number_of_samples - 1) == 0):
        raise ValueError("number_of_samples must be a power of 2")

    # prepare the symbol for points in position space
    x = sympy.symbols("x")

    # generate the result
    return expression_to_array(
        f(x),
        x,
        number_of_indices=number_of_samples,
        sampling_period=delta_x,
        sampling_type=sampling_type,
    )


def momentum_func_to_array(
    f: Symbolic_Function_Type,
    delta_x: float,
    number_of_samples: int,
    sampling_type: Sampling_Type,
) -> complex_array:
    # check if number_of_samples is a power of 2
    if not (number_of_samples & (number_of_samples - 1) == 0):
        raise ValueError("number_of_samples must be a power of 2")

    # prepare the symbol for points in position space
    delta_p = (
        scipy.constants.hbar * 2 * scipy.constants.pi / number_of_samples / delta_x
    )  # TODO: I might need to double-check this
    p = sympy.symbols("p")

    # generate the result
    return expression_to_array(
        f(p),
        p,
        number_of_indices=number_of_samples,
        sampling_period=delta_p,
        sampling_type=sampling_type,
    )


def non_normalized_wavefunction_to_statevector(
    f: Symbolic_Function_Type,
    delta_x: float,
    number_of_samples: int,
    sampling_type: Sampling_Type,
) -> complex_array:
    # check if number_of_samples is a power of 2
    if not (number_of_samples & (number_of_samples - 1) == 0):
        raise ValueError("number_of_samples must be a power of 2")

    # prepare the symbol for points in position space
    x = sympy.symbols("x")

    # generate the array
    array = expression_to_array(
        f(x),
        x,
        number_of_indices=number_of_samples,
        sampling_period=delta_x,
        sampling_type=sampling_type,
    )
    # normalize
    array /= np.linalg.norm(array)

    # construct the result
    return array


# define your signals as a function of x for the position representation
def custom_potential(x: sympy.Symbol) -> sympy.Expr:
    return x  # type: ignore


# and p for the momentum representation
def kinetic_energy(p: sympy.Symbol, m: float = 1.0) -> sympy.Expr:
    return p**2 / 2 / m  # type: ignore


def harmonic_oscillator_potential(
    x: sympy.Symbol, m: float = 1.0, omega: float = 1.0
) -> sympy.Expr:
    return m * omega**2 * x**2 / 2  # type: ignore


def psi_non_normalized_wavefunction(x: sympy.Symbol) -> sympy.Expr:
    return sympy.exp(-(x**2))  # type: ignore

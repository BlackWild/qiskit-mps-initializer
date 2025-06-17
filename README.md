# Qiskit MPS Initializer

![PyPI - License](https://img.shields.io/pypi/l/qiskit-mps-initializer)
![PyPI - Version](https://img.shields.io/pypi/v/qiskit-mps-initializer)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qiskit-mps-initializer)

This package provides extra tools on Qiskit enabling you to initialize wavefunctions on a quantum computer using techniques based on matrix product states (MPS).

## User guide

### Installation

The easiest way to get started is using `pip` which installs the package and its dependencies in your project

```bash
pip install qiskit-mps-initializer
```

You can alternatively use `uv`, `poetry` or any other python package manager to install this package.

### Usage

Once you have installed the package, you can import tools from it.

```python
from qiskit_mps_initializer.datatypes import QuantumState

# wavefunction as an array of numbers, could use np.array as well
psi = [1, 3, 1, 2, 7, 8, 0, 1]
# number of layers of two-qubit gates to use for the initializer
number_of_mps_layers = 2

# create the state object
state = QuantumState.from_dense_data(data=psi, normalize=True)

# generate the initializer circuit for this state
circuit = state.generate_mps_initializer_circuit(number_of_layers=number_of_mps_layers)

# the circuit object is a qiskit.QuantumCircuit object which
# means you can do whatever you could natively do in Qiskit
circuit.draw('mpl')
```

### Documentation

The project's documentation is located [here](https://blackwild.github.io/qiskit-mps-initializer/). Check it out for more information.

> [!CAUTION]
> This project is in alpha. This means you should expect drastic changes in the api in later releases.

## Project credits

### Tools

- Project and dependency manager: `uv`
- Linter: `ruff`
- Formatter & style: `ruff`
- Static typechecking: `pyright` (`ty` is currently in beta, `pyrefly` is also another candidate, both built using Rust)
- Unit testing: `pytest` (no Rust-based alternative)
  - Randomization: `hypothesis`
- Documentation: `mkdocs`
  - API auto-generation: `mkdocstrings-python`
  - Theme: `mkdocs-material`

### Dependencies

- Quantum circuits: `qiskit`
- Data modeling & validation: `pydatic`
  - Numpy support: `pydantic-numpy`
- Tensor networks: `quimb`

## Tools

- Project and dependency manager: `uv`
- Linter: `ruff`
- Formatter & style: `ruff`
- Static typecheck: `pyright` (`ty` is currently in beta, `pyrefly` is also another candidate, both built using Rust)
- Unit testing: `pytest` (no Rust-based alternative)
    - Randomization: `hypothesis`

## Dependencies

- Quantum circuits: `qiskit`
- Data modeling & validation: `pydatic`
- Tensor networks: `quimb`

## TODO

- License it?

## To fix

- remove manually added `# type: ignore` expressions

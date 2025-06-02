
## Dev guide

- Make sure you have `uv` installed. You can use the included dev container.
- Run `uv sync` in the directory to set up the `venv`.

Now you can start developing some code!

For runing the tests, you should run

```sh
uv run pytest
```

to run `pytest` inside the venv.

## Project credits

### Tools

- Project and dependency manager: `uv`
- Linter: `ruff`
- Formatter & style: `ruff`
- Static typecheck: `pyright` (`ty` is currently in beta, `pyrefly` is also another candidate, both built using Rust)
- Unit testing: `pytest` (no Rust-based alternative)
    - Randomization: `hypothesis`

### Dependencies

- Quantum circuits: `qiskit`
- Data modeling & validation: `pydatic`
- Tensor networks: `quimb`


## Notes

### Todo

- remove manually added `# type: ignore` expressions

### Other

- `scipy-stubs`'s python `3.10` reqiuirement is keeping this project's python requirement to go down to `3.9`.
- `typing.Self` was introduced in python `3.11`. Thus we are forced to use `3.11` at the moment.

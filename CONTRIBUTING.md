
This project is in alpha phase now. The documentation is not prepared yet. However, there are a few tips down here for anyone wanting to check out the repository.

## Developer guide

### Development environment

This project uses `uv` as package manager. You are expected to install `uv` and then will be able to easily interact with the project in an isolated environment.

The **Recommended** approach is to use Docker and run the development container defined in `.devcontainer` when working on the package.

Once you cloned the repo and prepared the dev environment, you should run

```bash
uv sync
```

in the root directory to have `uv` install all the dependency packages locally in the `.venv` folder. Now you can start developing some code!

### Unit testing

For running the tests, you should run

```bash
uv run pytest
```

in the root directory and this will the tests using `pytest` inside the venv.

Because the nature of this project is somehow that it involves lots of calculations on numbers, we use `hypothesis` to randomize the inputs of the tests.


## Development notes

### Todo

- remove manually added `# type: ignore` expressions

### Other

- `scipy-stubs`'s python `3.10` requirement is keeping this project's python requirement to go down to `3.9`.
- `typing.Self` was introduced in python `3.11`. Thus we are forced to use `3.11` at the moment.
- would it be possible to develop using higher python versions for static typing but release the package by transpling it for lower python versions?

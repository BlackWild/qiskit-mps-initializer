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

### Pydantic

- it seems defining custom `__init__()` functions for the pydantic models is not the best idea because it messes with the validation procedure of pydantic [Link](https://docs.pydantic.dev/latest/concepts/models/#model-signature). Instead, do not define custom constructors as `__init__()` but `@classmethod`s.

### Todo

- remove manually added `# type: ignore` expressions

### Other

- would it be possible to develop using higher python versions for static typing but release the package by transpling it for lower python versions?
- do not forget to add `__init__.py` files to all directories of the src folder... otherwise documentation generation won't work.
- unfortunately, as of today, `ruff` does not have a rule to enforce all the sections like `Arg:` in docstrings. you have to be careful to include all of them manually so that the doc generators extract type information from the code...
- defining `__all__` variable in `__init__.py` files enables better importability of objects in from the submodules i.e. `from ... import ...`
- be careful about order of the imports in the `__init.__py` files, avoiding circular imports
- `pydantic` does not support `numpy` arrays. I am using a class `pydantic_numpy.NumpyModel` instead of `pydantic.BaseModel` to enable the support.

#### Older notes

##### Python compatibility

- going down to `3.10`...
- `typing.Self` is not helpful anyways because `mkdocs` has troubles resolving `Self` to the actual class.

- ~~`scipy-stubs`'s python `3.10` requirement is keeping this project's python requirement to go down to `3.9`.~~
- ~~`typing.Self` was introduced in python `3.11`. Thus we are forced to use `3.11` at the moment.~~

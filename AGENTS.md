# Repository Guidelines

## 1. Project structure
- `tune/` – the package that contains the CLI, database workers, and algorithmic helpers.
- `tests/` – a mirror of `tune/` with pytest test modules.
- `docs/`, `examples/`, `README.rst` – user‑facing documentation.
- `.github/workflows/` – GitHub Actions CI.
- `Makefile` – command shortcuts for common tasks.
- `pyproject.toml` – uv build and dependency information.
- `noxfile.py` – run tests, lint, and type‑check in reproducible environments.
- `.pre-commit-config.yaml` – ruff hooks.

## 2. Installing / setting up the dev environment
```bash
# Pull the latest source and install dependencies
uv sync --group dev --extra dist
# Alternatively, use pip in a venv
# python -m venv .venv && source .venv/bin/activate && pip install -e .[dev,dist]
```
The `--extra dist` flag pulls optional packages for distributed execution (joblib, pandas, postgresql, sqlalchemy).

## 3. Running tests
- All tests
  ```bash
  uv run --group dev pytest
  ```
- Single test file
  ```bash
  uv run --group dev pytest tests/test_tune.py
  ```
- Single test function
  ```bash
  uv run --group dev pytest tests/test_tune.py::test_my_feature
  ```
- Use `-q` for quiet output or `-s` to see print statements.
- For CI‑aligned version:
  ```bash
  uv run --group dev nox -s tests
  ```

## 4. Linting, formatting, and type‑checking
| Command | Purpose |
|---------|---------|
| `uv run --group dev nox -s pre-commit` | Run ruff format & check on all files. |
| `uv run --group dev ruff format` | In‑place formatting only. |
| `uv run --group dev ruff check` | Find style and lint errors only. |
| `uv run --group dev nox -s ruff` | Same as above but via nox. |
| `uv run --group dev mypy .` | Static type checking. |

**Note:** All tools obey the ruff config in `pyproject.toml` (max‑line‑length 80, ignore E203,E501,F403,F405). `mypy.ini` ignores imports for third‑party libraries.

## 5. Building / publishing
```bash
uv build  # creates dist/*.tar.gz & dist/*.whl
uv publish  # pushes to PyPI if credentials are set via UV_PROJECT
```

## 6. Coding style guidelines
- **Indentation**: 4 spaces, never tabs. All files must pass `ruff format`.
- **Naming**: snake_case for functions, variables, modules; PascalCase for classes and data structures.
- **Imports**: absolute imports inside the package; group standard library, third‑party, local in that order, each separated by a blank line.
- **Type hints**: provide annotations for public functions and classes. Use `typing` primitives (`List`, `Dict`, `Tuple`, `Optional`), and `typing.Any` only when necessary.
- **Docstrings**: use Google style or reST; no empty docstrings. For public API functions, include brief description, parameters, and return values.
- **Error handling**: raise specific exceptions derived from `Exception`; avoid `sys.exit` in library code.
- **Logging**: use `logger = logging.getLogger(__name__)`; prefer structured logging via `logger.info("msg", extra={...})`.
- **Line length**: keep within 80 characters as enforced by ruff; use line continuation or `typing.get_type_hints` for complex type hints.
- **Magic numbers**: replace with named constants or `Enum` where appropriate.
- **Use of `# noqa`**: only for legitimate cases; include a short comment explaining why.

## 7. Testing guidelines
- Name tests `test_<module>_<feature>.py` in `tests/`.
- Keep tests deterministic; avoid external network calls.
- Use fixtures for reusable setup.
- Mark tests with `@pytest.mark.parametrize` for parameterized coverage.

## 8. Git workflow
- Commit messages in imperative, short form.
- Avoid committing `.venv` or other large binary directories.
- Run `uv run --group dev nox -s pre-commit` before pushing.
- Use `git add` followed by `git commit -m "Msg"`.

## 9. Pull request policy
- Reference related issue (e.g., `Fixes #123`).
- Include CI status badges in the PR description.
- Add a brief summary in the PR body and a list of changes.

## 10. Miscellaneous
- There are no cursor rules or Copilot instructions in this repo.
- The repository is primarily for research and distributed tuning of chess engines.
- If you need to contribute a new feature, first create a separate branch, run `uv run --group dev nox -s pre-commit`, then run all tests.
- For documentation, run `make docs` or `make servedocs`.
- The Makefile also includes targets: `clean`, `lint`, `test-all`, `coverage`. These wrap the same commands above.

# End of guidelines

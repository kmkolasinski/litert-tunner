# Contributing to litert-tunner

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository.
1. Ensure you have `uv` installed for Python package management.
1. Activate the virtual environment: `source .venv/bin/activate`
1. Run setup (or install dependencies via uv).
1. Setup pre-commit hooks: `make precommit` or `pre-commit install`.

## Coding Standards

- **Python ≥ 3.11**: Modern syntax (`|` unions, `match`).
- **Type hints** on all signatures. **Google-style docstrings** on public API.
- **Line length** 100 chars. **Linting**: `ruff check` + `ruff format`.
- **Imports**: Google-style module imports only. Never import symbols directly.
- **No magic numbers**: Use constants/enums.
- **RNG**: `np.random.default_rng(seed)` only. Never `np.random.seed`.

## Testing

- **Activate environment**: `source .venv/bin/activate`
- **Run tests**: `.venv/bin/python -m pytest tests/ -v`
- **Lint**: `make precommit`
- **Run all tests**: `make test`

## Making a Pull Request

1. Fork the repository and create your branch from `main`.
1. Implement your changes.
1. Ensure all tests pass (`make test`).
1. Ensure linting passes (`make precommit`).
1. Submit a Pull Request with a clear description of the changes.

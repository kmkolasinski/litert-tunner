help: ## Print this message and exit.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

# change shell from sh to bash, it enables source command in makefile
SHELL := /bin/bash


venv: ## Creates a virtual environment using uv
	uv venv

install: ## Installs the project in editable mode with dev dependencies
	uv pip install -e ".[dev]"

test: ## Runs the tests
	python -m pytest

precommit: ## Runs the pre-commit hooks
	pre-commit run --all-files

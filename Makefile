help: ## Print this message and exit.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

# change shell from sh to bash, it enables source command in makefile
SHELL := /bin/bash

# Prepend the virtual environment's bin directory to PATH
export PATH := $(CURDIR)/.venv/bin:$(PATH)

init: venv ## One-time dev setup: installs uv into the venv and pre-commit hooks
	pip install uv
	uv run --with pre-commit pre-commit install
	@echo "Done! Run 'make install' to install project dependencies."

venv: ## Creates a virtual environment
	python3 -m venv .venv

install: ## Installs the project in editable mode with dev dependencies
	uv pip install -e ".[dev]"

test: ## Runs the tests with coverage and parallel execution
	python -m pytest -n 4 --forked \
		--durations=20 \
		--cov=litert_tunner \
		--cov-branch \
		--cov-report=term \
		--cov-report=html:test-results/htmlcov \
		--no-cov-on-fail \
		--cov-fail-under=70

precommit: ## Runs the pre-commit hooks
	pre-commit run --all-files

clean: ## Removes test artifacts and cache directories
	rm -rf test-results/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

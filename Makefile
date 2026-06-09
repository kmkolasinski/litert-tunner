help: ## Print this message and exit.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

# change shell from sh to bash, it enables source command in makefile
SHELL := /bin/bash

# Prepend the virtual environment's bin directory to PATH
export PATH := $(CURDIR)/.venv/bin:$(PATH)

init: venv ## One-time dev setup: installs uv into the venv
	pip install uv
	@echo "Done! Run 'make install' to install project dependencies and setup pre-commit hooks."

venv: ## Creates a virtual environment
	python3 -m venv .venv

activate: ## Activates the virtual environment
	source .venv/bin/activate

install: ## Installs the project in editable mode with dev dependencies and sets up pre-commit hooks
	uv pip install -e ".[dev]"
	pre-commit install

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

release-patch: ## Bump patch version, commit, tag, and push
	bump-my-version bump patch
	git push origin main --follow-tags

release-minor: ## Bump minor version, commit, tag, and push
	bump-my-version bump minor
	git push origin main --follow-tags

release-major: ## Bump major version, commit, tag, and push
	bump-my-version bump major
	git push origin main --follow-tags

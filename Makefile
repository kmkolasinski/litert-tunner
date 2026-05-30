.PHONY: install test precommit

install:
	python -m pip install -e ".[dev]"

test:
	pytest

precommit:
	pre-commit run --all-files

---
trigger: model_decision
description: Testing New or Updated Code
---

After changes remember about automatic code check!

When running a test use current project env example command or you can specify exact test name for shorter waiting time.
source .venv/bin/activate && .venv/bin/python -m pytest tests/
Never timeout on tests, always wait to the end, some tests may take more time.

After writing code check for syntax errors with
npx -y pyright --pythonpath .venv/bin/python

Make sure pre-commits are passing with
source .venv/bin/activate && pre-commit migrate-config && make precommit
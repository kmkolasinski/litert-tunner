---
trigger: always_on
---

When running test use current project env example command:
source .venv/bin/activate && .venv/bin/python -m pytest tests/

After writing code check for syntax errors with
npx -y pyright --pythonpath .venv/bin/python

Make sure pre-commits are passing with 
source .venv/bin/activate && pre-commit migrate-config && make precommit
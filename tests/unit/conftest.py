# Minimal conftest for pure-logic unit tests (no DB/Redis/Kafka needed).
# This conftest intentionally overrides the parent conftest.py which requires
# sqlalchemy/fastapi backend modules not available in this environment.

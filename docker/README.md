# Docker

Dockerfiles for the Octopus stack. Build context is the **repository root**.

- **Dockerfile.fastapi** – API service
- **Dockerfile.celery** – Celery worker (and beat/flower when used)
- **Dockerfile.fingpt-inference** – Optional FinGPT inference server

Compose files (`docker-compose-core.yml`, etc.) at repo root reference these with `dockerfile: docker/Dockerfile.fastapi`.

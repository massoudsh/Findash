# Docker

Dockerfiles for the Octopus stack. Build context is the **repository root**.

- **Dockerfile.fastapi** – API service
- **Dockerfile.celery** – Celery worker (and beat/flower when used)
- **Dockerfile.fingpt-inference** – Optional FinGPT inference server

Compose files (`docker-compose-core.yml`, etc.) at repo root reference these with `dockerfile: docker/Dockerfile.fastapi`.

## Ports (core compose)

| Service | Inside container | On the host |
|---------|------------------|-------------|
| API (`api`) | `8000` | **8011** |
| Frontend | `3003` | **3003** |

The frontend image is built with `NEXT_PUBLIC_API_URL=http://localhost:8011` (browser → host). At runtime Compose also sets `BACKEND_INTERNAL_URL=http://api:8000` so NextAuth / server-side login can reach the API over the Docker network. Do not point `NEXT_PUBLIC_*` at `http://api:8000` — the browser cannot resolve that hostname.

## Persian PDF (issue #18)

`GET /api/reports/portfolio.pdf` (`src/services/pdf_reports.py`) registers Vazirmatn from:

- `/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf`
- `/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Bold.ttf` (optional; Regular is reused)

`Dockerfile.fastapi` does **not** copy or `apt-get install` those files. Without them the endpoint returns **HTTP 503**. Packages `reportlab`, `arabic_reshaper`, and `python-bidi` are in `requirements/requirements.txt`.

To enable PDF in the API image, bind-mount or copy those two TTFs into the paths above (as root in the image, before `USER trading`) and rebuild. Python deps alone are not enough: Helvetica has no Persian glyphs.

Session and BFF URL pitfalls: [docs/FRONTEND_SESSION.md](../docs/FRONTEND_SESSION.md).

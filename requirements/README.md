# Python requirements

- **requirements.txt** – Main app dependencies (FastAPI, DB, Redis, etc.). Use for production and CI.
- **requirements-dev.txt** – Dev tools (pytest, black, flake8, mypy). Install after main deps for local dev.
- **requirements-basic.txt** – Minimal set for quick runs.
- **requirements-llm.txt** – LLM/report-generation extras (e.g. transformers, Hugging Face).
- **requirements-quickstart.txt** – One-shot quickstart install list.

From project root: `pip install -r requirements/requirements.txt` then `pip install -r requirements/requirements-dev.txt` for dev.

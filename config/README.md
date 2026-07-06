# Config

- **env.example** – Template for environment variables. Copy to project root as `.env`:  
  `cp config/env.example .env`
- **api_keys_config.py** – Placeholder/reference for API keys and rate limits. Prefer setting keys via `.env` (e.g. `ALPHA_VANTAGE_API_KEY`). Do not commit real keys.
- **metrics_config.py** – Single source of truth for all metric categories (fundamental, social, technical, sentiment, macro, on-chain, options, risk, market_data). Every parameter (weights, thresholds, lookbacks) that affects metrics in any case is defined here and can change dynamically via:
  - **File:** `METRICS_CONFIG_FILE` pointing to a JSON file (see `metrics_config.example.json`).
  - **Env:** `METRICS_<CATEGORY>_ENABLED`, `METRICS_<CATEGORY>_WEIGHT`, `METRICS_<CATEGORY>_PARAMS` (JSON) to override per category.
  Use: `from config.metrics_config import get_metrics_config` then `cfg = get_metrics_config()`.

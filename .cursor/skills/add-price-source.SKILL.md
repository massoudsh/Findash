# Skill: Add a New Price or Data Source (M1 Data Collector)

Use this skill when adding a new scraper, API client, or feed that supplies **prices or market/data** to Findash. It keeps the Data Collector (M1) and orchestrator consistent.

## Steps

1. **Implement the fetcher**
   - **Scraper**: Add or extend a class under `src/data_processing/scraping/` (e.g. `news_scraper.py`, `dashboard_scraper.py`). Use `requests` with `timeout`, `BeautifulSoup` for HTML, and module-level `logger`. Return a list of dicts or a single dict with known keys (e.g. `symbol`, `price`, `timestamp`, `source`).
   - **API client**: Add or extend under `src/data_processing/` or `src/services/`. Read API key from env (e.g. `os.getenv("MY_API_KEY")`). Use rate limiting and retries; normalize symbols and timestamps before returning.

2. **Validate and normalize**
   - Normalize symbol tickers (e.g. uppercase, strip whitespace).
   - Use ISO timestamps or project-standard datetime format.
   - Validate numeric fields (price, volume) and handle missing/invalid values (log and skip or raise).

3. **Wire to ingestion**
   - If there is a Celery task for scheduled collection, add or update a task in `src/data_processing/collection_tasks.py` that calls your fetcher and writes to DB or cache.
   - If data is consumed by the orchestrator, ensure the output shape matches what `submit_task("M1_data_collector", task_type, data)` expects (e.g. `data` with `symbol`, `pipeline_id`, and any new keys).

4. **Orchestrator (optional)**
   - If this source should be triggered from the pipeline, extend `coordinate_pipeline()` in `src/core/intelligence_orchestrator.py` so that the M1 task payload includes the new source or a task_type that invokes it. Do not add a new agent; extend M1 capabilities.

5. **Config and docs**
   - Add any new env vars to `config/env.example` (no real keys). Update `config/api_keys_config.py` or project config if the project uses a central key store.
   - Optionally document the new source in `wiki-content/` or `docs/` (e.g. AI-Agents.md M1 section).

## Files to touch (typical)

- `src/data_processing/scraping/<new_scraper>.py` or `src/data_processing/ingestion/`
- `src/data_processing/collection_tasks.py` (if scheduled)
- `config/env.example` (new env vars)
- `src/core/intelligence_orchestrator.py` (only if pipeline must call this source)

# Skill: Add a New Report or Insight Type (M11 Visualizer)

Use this skill when adding a **new report, chart, or insight** (e.g. daily summary, strategy performance report, sentiment summary) so backend and frontend stay aligned.

## Steps

1. **Backend: data and logic**
   - Add or extend analytics under `src/analytics/` (e.g. new function or method that computes the report from DB/cache). For LLM-generated text or summaries, use `src/generative/` and the project’s LLM config.
   - Define the response shape: time range, series for charts (e.g. `[{ timestamp, value, label }]`), and any summary text. Use Pydantic schemas or dataclasses so the API response is consistent.

2. **Backend: API**
   - Add or extend an endpoint under `src/api/endpoints/` (e.g. a new router or route for reports/insights). Return JSON that matches the schema. Support query params for symbol, date range, and report type. Document in OpenAPI (FastAPI will pick up route and schemas).

3. **Orchestrator (optional)**
   - If the report is generated inside the pipeline, use or add a task_type for M11 in `src/core/intelligence_orchestrator.py` (e.g. `generate_report` or `generate_insight`) and pass the required payload (symbol, range, report_type). Implement the handler so it calls your new analytics/generative code.

4. **Frontend**
   - Add or extend a component under `frontend-nextjs/src/components/` (e.g. dashboard or charts). Fetch from the new API; use the project’s API client pattern (e.g. `src/lib/api.ts`). For charts, use the existing chart library (e.g. Recharts) with the same axis and tooltip conventions.
   - Add or update a page or section that exposes the report (e.g. Dashboard, Analytics, or a dedicated Reports page). Use loading and error states for the async request.

5. **Idempotency and performance**
   - Ensure the same inputs (symbol, range, type) produce the same report so caching/refresh work. For large ranges, support pagination or sampling in the backend and avoid unbounded queries.

## Files to touch (typical)

- `src/analytics/` or `src/generative/`
- `src/api/endpoints/` (new or existing report router)
- `src/core/intelligence_orchestrator.py` (if pipeline-driven)
- `frontend-nextjs/src/lib/api.ts` (new client method if needed)
- `frontend-nextjs/src/components/` (chart or report component and page)

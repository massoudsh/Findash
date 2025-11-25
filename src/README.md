# Core Python Modules (`src/`)

The `src/` directory contains the **core, reusable Python modules** that power the Octopus backend. Each sub-package has a clear responsibility and should expose a clean, well-documented interface.

| Module       | Path            | Responsibility                                                                 |
|-------------|-----------------|-------------------------------------------------------------------------------|
| API         | `src/api/`      | FastAPI **endpoints**, request/response **schemas**, and public HTTP interface. |
| Agents      | `src/agents/`   | **Intelligence orchestrator** and the M1–M11 **AI agents** (data, ML, risk, execution, backtesting, compliance, alt‑data). |
| DB          | `src/db/`       | **SQLAlchemy models**, database session utilities, and **migration glue** for the data layer. |
| Security    | `src/security/` | Authentication and authorization: **JWT handling**, password hashing, rate limiting, and related security helpers. |
| Services    | `src/services/` | Core **business logic** and reusable domain services that are not tied to HTTP or specific agents. |
| Monitoring  | `src/monitoring/` | **Metrics**, Prometheus instrumentation, health checks, and logging/observability helpers. |

## Usage Guidelines

- Put **shared, production-grade logic** here; keep ad‑hoc scripts and experiments outside `src/`.
- Keep modules **small and cohesive**; prefer clear function-based organization and explicit interfaces.
- Add a `README.md` to each submodule that documents its purpose, key entry points, and any external dependencies.

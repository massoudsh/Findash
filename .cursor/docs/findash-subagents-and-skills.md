# Using Cursor Rules, Subagents, and Skills with Findash Agents

Findash (Octopus) uses **11 orchestrated AI agents** (M1–M11). You can align Cursor’s **rules**, **subagents**, and **skills** with these modules so the AI assistant behaves like a specialist per area.

---

## 1. Rules (already set up)

**Location:** `.cursor/rules/*.mdc`

| Rule | When it applies | Purpose |
|------|------------------|---------|
| **findash-agents.mdc** | Editing orchestrator or any agent module under `src/` (data_processing, analytics, strategies, prediction, risk, backtesting, realtime, portfolio, etc.) | Keeps context: which agent (M1–M11) you’re in, code paths, and orchestrator contracts. |
| **findash-data-collector.mdc** | Editing `src/data_processing/**`, `src/infrastructure/market_data*.py`, `src/services/market_data*.py` | M1: scraping, API fetching, rate limits, validation, no hardcoded keys. |
| **findash-reports-insights.mdc** | Editing `src/analytics/**`, `src/generative/**`, frontend dashboard/charts | M11: reports, charts, insights, API response shape, frontend consistency. |
| **python-dev.mdc**, **ai-python.mdc**, **ai-fin.mdc** | (Always or by glob) | Project-wide Python/FastAPI and frontend conventions. |

Rules are applied automatically by **globs** when you open or edit matching files. You can also invoke a rule with `@rule-name` in Cursor chat or Cmd-K.

---

## 2. Subagents (how to configure in Cursor)

**Subagents** are separate agents with their own context and prompts. You can define them in Cursor so that, for a given task, the right “specialist” runs (e.g. one for scraping, one for reports).

Suggested mapping from **Findash agents** to **Cursor subagent roles**:

| Findash agent | Suggested subagent name | Suggested role / prompt summary |
|---------------|--------------------------|----------------------------------|
| **M1** Data Collector | Data Collector | “You are the Findash M1 Data Collector. You only edit code under `src/data_processing/`, `src/infrastructure/market_data*.py`, `src/services/market_data_service.py`. You add or fix scrapers and API clients: use requests/BeautifulSoup, timeouts, rate limiting, env for API keys, and validation. Follow .cursor/rules/findash-data-collector.mdc.” |
| **M9** Sentiment Analyzer | Sentiment & alternative data | “You are the Findash M9 Sentiment / alternative data agent. You only edit `src/analytics/sentiment_agent.py` and `src/alternative_data/`. You handle NLP, sentiment, news/social pipelines. Follow findash-agents.mdc and use project exceptions and async where applicable.” |
| **M4** Strategy Agent | Strategy & signals | “You are the Findash M4 Strategy agent. You only edit `src/strategies/` (strategy_agent, signal_fusion). You implement or fix signal generation, strategy logic, and backtesting integration. Follow findash-agents.mdc.” |
| **M11** Visualizer / reports | Reports & insights | “You are the Findash M11 Visualizer. You only edit `src/analytics/`, `src/generative/`, and frontend dashboard/charts. You add or fix reports, charts, and insight APIs. Follow findash-reports-insights.mdc.” |
| **Orchestrator** | Orchestrator | “You are editing the Findash Intelligence Orchestrator. You only change `src/core/intelligence_orchestrator.py`: agent registry, `submit_task` contracts, and `coordinate_pipeline` order. Follow findash-agents.mdc and the skill ‘orchestrator-agent’ when adding or changing agents.” |

**How to create a subagent in Cursor**

1. Open Cursor settings (or Agent / Subagents section where available).
2. Add a **custom subagent**.
3. Set **name** (e.g. “Data Collector”) and **prompt** using the text from the table (and point to the rule file).
4. Optionally restrict **tools** (e.g. only code edit + read, no run) or **paths** (e.g. `src/data_processing/`) if Cursor supports path scoping.

Then, when you say “add a new price scraper” or “fix the sentiment pipeline”, you can assign the task to the **Data Collector** or **Sentiment** subagent so the right rules and scope apply.

---

## 3. Skills (already set up)

**Location:** `.cursor/skills/*.SKILL.md`

Skills are **procedural “how-to”** instructions. They are applied when relevant (e.g. you ask to add a data source or a new agent), and can be invoked via the slash command menu or by the agent.

| Skill file | When to use |
|------------|-------------|
| **add-price-source.SKILL.md** | Adding a new price or data source: scraper, API client, or feed. Ensures M1 conventions, ingestion, and optional orchestrator wiring. |
| **add-report-insight.SKILL.md** | Adding a new report, chart, or insight type. Ensures backend API + analytics/generative logic and frontend stay aligned (M11). |
| **orchestrator-agent.SKILL.md** | Adding a new agent (M12+) or changing an existing agent’s capabilities or pipeline order in the Intelligence Orchestrator. |

**How to use**

- In chat: e.g. “Add a new price source for crypto XYZ” or “Add a daily PnL report” — the agent should discover and follow the corresponding skill.
- Manually: use Cursor’s slash command or skill menu and select the skill by name.

---

## 4. Quick reference: “I want to…”

| Goal | Use |
|------|-----|
| Scrape or add a new price/data source | Rule: **findash-data-collector**. Skill: **add-price-source**. Optional subagent: **Data Collector**. |
| Add or change reports/insights/charts | Rule: **findash-reports-insights**. Skill: **add-report-insight**. Optional subagent: **Reports & insights**. |
| Add or change an agent in the orchestrator | Rule: **findash-agents**. Skill: **orchestrator-agent**. Optional subagent: **Orchestrator**. |
| Work on sentiment / alternative data | Rule: **findash-agents** (+ data/analytics paths). Optional subagent: **Sentiment & alternative data**. |
| Work on strategy/signals | Rule: **findash-agents**. Optional subagent: **Strategy & signals**. |

This setup helps Cursor stay within the right agent boundaries and follow the same patterns as the Findash codebase (orchestrator, M1–M11, and pipeline contracts).

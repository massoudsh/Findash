# Skill: Add or Modify an Agent in the Intelligence Orchestrator

Use this skill when **registering a new agent (M12+)** or **changing an existing agent’s capabilities** in the Findash Intelligence Orchestrator so pipelines and task contracts stay consistent.

## Steps

1. **Agent registry**
   - Open `src/core/intelligence_orchestrator.py`.
   - In `_initialize_agents()`, add or update an entry in `self.agents`:
     - **New agent**: Use a unique key (e.g. `M12_my_agent`), and set `name`, `status` (`"active"`), `capabilities` (list of strings), and `priority` (1=high, 4=low).
     - **Existing agent**: Update `capabilities` or `name` only if the change is intentional; avoid breaking existing `task_type` contracts.

2. **Task types and contracts**
   - `submit_task(agent_name, task_type, data, priority)` is the main API. Decide the `task_type` string (e.g. `fetch_market_data`, `generate_report`) and the expected `data` keys (e.g. `symbol`, `pipeline_id`, `date_range`). Document or add a comment so other code and pipelines use the same contract.
   - Ensure the agent’s **implementation** (in its module under `src/`) actually handles this `task_type` and returns a result that `get_task_result()` callers expect.

3. **Pipeline integration**
   - If the agent must run in the standard analysis pipeline, edit `coordinate_pipeline()` in the same file. Add a `submit_task()` call for your agent at the right stage (order matters: e.g. data first, then ML, then strategy, then reports). Use the same `data` keys (e.g. `symbol`, `pipeline_id`) so downstream code can correlate results.
   - If the pipeline should be conditional (e.g. only for `analysis_type == "full"`), mirror the pattern used for M10 (backtester) and M11 (visualizer).

4. **Cursor rule**
   - Update `.cursor/rules/findash-agents.mdc`: add the new agent to the Agent → Code Map table with its ID, name, code paths, and capabilities. If the new agent has its own domain (e.g. a new “data source” type), consider adding a dedicated rule file with `globs` for its paths.

5. **Docs**
   - Update `wiki-content/AI-Agents.md` (and any architecture docs) with the new agent’s purpose, capabilities, and where it runs in the pipeline.

## Do not

- Remove or rename an existing agent ID (e.g. M1–M11) without updating all callers and pipelines.
- Change `task_type` or `data` shape for an existing agent without updating every place that submits or consumes that task.
- Add a new agent without implementing the corresponding logic in `src/` and (if used in pipeline) without adding the `submit_task` call in `coordinate_pipeline()`.

## Files to touch (typical)

- `src/core/intelligence_orchestrator.py` (registry + pipeline)
- `.cursor/rules/findash-agents.mdc` (agent map)
- `wiki-content/AI-Agents.md` (documentation)
- The agent’s implementation module under `src/` (so it handles the new or updated task_type)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vineland-runner` — CLI tool for evaluating LLM agents against a psychometric assessment instrument adapted from Vineland Adaptive Behavior Scales (VABS-3). Target: NeurIPS 2026 Education Track (8 models × 24 items × 5 replications = 960 runs).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

API keys via environment variables (see `configs/agents.yaml` for `api_key_env` field per agent):
```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...
```

## Commands

```bash
# Validate items YAML
vineland-runner validate items/items.yaml

# Run (or resume) a pilot
vineland-runner run --config configs/pilot.yaml

# Compute polytomous scores
vineland-runner score runs/pilot_2026_04

# Print markdown summary
vineland-runner summary runs/pilot_2026_04

# Extract run statistics
vineland-runner stats runs/pilot_2026_04/runs.jsonl

# Run all tests
pytest

# Run a single test file
pytest tests/test_grading.py

# Run a single test by name
pytest tests/test_runner_integration.py::test_resumability
```

## Architecture

The pipeline is: **items.yaml → runner → runs.jsonl → scoring → scores.jsonl/csv → summary.md**

### Core modules (`vineland_runner/`)

- **`types.py`** — Pydantic models for everything: `Item`, `Agent`, `PilotConfig`, `RunRecord`, `ScoreRow`, `APIResponse`. `SuccessCriterion` is a discriminated union (`exact_match` | `llm_judge`).
- **`runner.py`** — Async pilot loop. `run_pilot()` fans out `(agent, item, replication)` tasks under a semaphore (`max_concurrency`). Each task writes to `runs.jsonl` atomically before proceeding. Resumability is achieved by loading completed `run_id`s at startup and skipping them.
- **`grading.py`** — Two grader strategies: `ExactMatchGrader` (regex or string containment) and `LLMJudgeGrader` (calls a second agent, parses its output with a regex). `make_grader()` dispatches on `item.success_criterion.type`.
- **`client.py`** — `LLMClient` wraps `httpx.AsyncClient` for OpenAI-compatible `/chat/completions`. Handles retries (tenacity), seed, reasoning tokens, and the `seed_unsupported` flag.
- **`scoring.py`** — `compute_scores()` groups `runs.jsonl` by `(agent_id, item_id)`, computes success rate `s`, and maps to polytomous score `y` (0/1/2) using thresholds τ₁=0.20, τ₂=0.75. `build_summary()` produces the markdown report.
- **`items.py`** — Loads items from YAML, optionally validates against `items/schema.json`, then Pydantic-validates. `select_items()` supports `"all"`, `"auto"` (middle tiers per subdomain), or an explicit list of IDs.
- **`config.py`** — Loads `pilot.yaml` and `agents.yaml`; resolves API keys from environment.
- **`storage.py`** — JSONL append (`append_record`), iteration (`iter_records`), and deduplication set (`load_completed_ids`).

### `vineland_api/` (separate package)

FastAPI service for leaderboard/ranking results. Has its own `pyproject.toml` and `Dockerfile`. Uses a GitHub-backed store (`github_store.py`) and a background worker (`worker.py`). Independent from `vineland_runner`.

### Data files

- `items/items.yaml` — Assessment items (72 total; partial set currently). `items/schema.json` defines the JSON Schema for validation.
- `configs/pilot.yaml` — Pilot configuration (agents, item selection, replications, output dir).
- `configs/agents.yaml` — Agent definitions (base_url, model_id, api_key_env, etc.).
- `runs/*/runs.jsonl` — Raw run records, one per `(agent, item, replication)`. The `run_id` field (`{agent_id}__{item_id}__rep{n}`) is the deduplication key.

### Testing

Tests use `respx` to mock `httpx` at the transport level — no real API calls. `conftest.py` is in `tests/`. `asyncio_mode = "auto"` in `pyproject.toml` means all `async def test_*` functions run without `@pytest.mark.asyncio`.

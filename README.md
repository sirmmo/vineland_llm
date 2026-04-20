# vineland-runner

Python CLI tool for evaluating LLM agents against a psychometric assessment instrument adapted from the [Vineland Adaptive Behavior Scales (VABS-3)](https://www.pearsonassessments.com/store/usassessments/en/Store/Professional-Assessments/Behavior/Adaptive/Vineland-Adaptive-Behavior-Scales-%7C-Third-Edition/p/100001622.html) methodology.

Targeting: NeurIPS 2026 Education Track pilot — 8 models × 24 items × 5 replications = 960 runs.

---

## Installation

```bash
# Create and activate a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install (editable)
pip install -e ".[dev]"
```

---

## BYOK Setup (Bring Your Own Key)

The tool uses any OpenAI-compatible API. Set environment variables before running:

```bash
# OpenRouter (covers Claude, DeepSeek, Mistral, Gemma, Llama via one key)
export OPENROUTER_API_KEY=sk-or-...

# OpenAI (for GPT-5, o3)
export OPENAI_API_KEY=sk-...

# Local vLLM or other OpenAI-compatible endpoint: configure base_url in agents.yaml
```

Each agent in `configs/agents.yaml` has an `api_key_env` field pointing to the relevant variable.

---

## Usage

### Validate items

```bash
vineland-runner validate items/items.yaml
```

### Run a pilot

```bash
vineland-runner run --config configs/pilot.yaml
```

The run is **resumable**: if interrupted, re-running the same command picks up from where it left off (based on `runs/pilot_2026_04/runs.jsonl`).

### Compute scores

```bash
vineland-runner score runs/pilot_2026_04
```

Writes `scores.jsonl` and `scores.csv` to the run directory.

### Print summary

```bash
vineland-runner summary runs/pilot_2026_04
```

Writes `summary.md` with per-agent token totals and per-subdomain mean scores.

---

## Dry run against mocked API (for development)

Use `respx` in a script or pytest fixture to mock `https://mock.api/v1`:

```python
import httpx, respx, asyncio
from vineland_runner.client import LLMClient
from vineland_runner.runner import run_pilot
from vineland_runner.types import Agent, Item, PilotConfig

agent = Agent(id="mock", display_name="Mock", base_url="https://mock.api/v1",
              model_id="mock-model", api_key_env="MOCK_KEY")
items = [Item(id="AF-TOL-1", domain="AF", subdomain="TOL", tier=1,
              prompt_template='Q: {question}', prompt_variables={"question": "2+2?"},
              success_criterion={"type": "exact_match", "expected_regex": "calculator"})]
config = PilotConfig(name="dry-run", agents=["mock"], items="all",
                     n_replications=3, output_dir="runs/dry-run", judge_agent=None)

import os; os.environ["MOCK_KEY"] = "sk-fake"

with respx.mock(base_url="https://mock.api") as router:
    router.post("/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": '{"tool": "calculator"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        })
    )
    async with httpx.AsyncClient() as http:
        asyncio.run(run_pilot(config, items, {"mock": agent}, LLMClient(http)))
```

---

## JSONL Schema

`runs/*/runs.jsonl` — one line per `(agent, item, replication)`:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | `{agent_id}__{item_id}__rep{n}` — unique, used for deduplication |
| `agent_id` | str | Agent identifier from agents.yaml |
| `item_id` | str | Item identifier (e.g. `CM-REC-1`) |
| `replication` | int | 0-indexed replication number |
| `prompt` | str | Rendered prompt sent to the model |
| `response` | str\|null | Model output (null if API error) |
| `prompt_tokens` | int | Token count from API usage field |
| `completion_tokens` | int | Token count from API usage field |
| `reasoning_tokens` | int | Reasoning tokens (0 for non-reasoning models) |
| `latency_s` | float | Wall-clock seconds for the API call |
| `success` | bool\|null | True/False if graded, null if API error |
| `grading_detail` | dict | Grader-specific debug info (pattern, judge output, etc.) |
| `error` | str\|null | Exception traceback if the run failed |
| `timestamp` | str | ISO-8601 UTC timestamp |
| `seed_unsupported` | bool | True if the model ignored the seed parameter |

`runs/*/scores.jsonl` — one line per `(agent, item)`:

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | str | Agent identifier |
| `item_id` | str | Item identifier |
| `domain` | str | Assessment domain (CM/DLS/SOC/AF) |
| `subdomain` | str | Subdomain code |
| `tier` | int | Capability tier 1-5 |
| `n_reps` | int | Number of valid (non-null) replications |
| `n_successes` | int | Count of successful runs |
| `s` | float | Success rate = n_successes / n_reps |
| `y` | int | Polytomous score: 0 (s<τ₁), 1 (τ₁≤s<τ₂), 2 (s≥τ₂) |

---

## Assessment Framework

- **4 domains**: Communication (CM), Daily Living Skills (DLS), Socialization (SOC), Agentic Functioning (AF)
- **13 subdomains**, **72 items** total
- **Tier bands 1–5** (1 = easiest, 5 = hardest)
- **Polytomous scoring 0/1/2**: default τ₁=0.20, τ₂=0.75

Items are in `items/items.yaml`. Currently 5 representative stubs are included.
Full 72-item set will be added from the research paper appendix.

---

## Running tests

```bash
pytest
```

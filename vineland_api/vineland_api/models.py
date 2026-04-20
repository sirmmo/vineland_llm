"""API request/response schemas."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, SecretStr


# ── Submit ────────────────────────────────────────────────────────────────────

class AgentSubmission(BaseModel):
    """Everything needed to register and test a new model."""
    id: str = Field(description="Short unique slug, e.g. 'my-llm-7b'")
    display_name: str = Field(description="Human-readable name for the leaderboard")
    base_url: str = Field(description="OpenAI-compatible base URL (no trailing slash)")
    model_id: str = Field(description="Model identifier passed to the API")
    api_key: SecretStr = Field(description="API key — used only during the run, never stored")
    max_tokens: int = Field(default=2048, ge=64, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    reasoning: bool = Field(default=False, description="Set true for chain-of-thought/reasoning models")
    notes: str = Field(default="", description="Free-text notes shown on leaderboard")

    model_config = {"json_schema_extra": {"example": {
        "id": "my-llm-7b",
        "display_name": "My LLM 7B",
        "base_url": "https://openrouter.ai/api/v1",
        "model_id": "myorg/my-llm-7b-instruct",
        "api_key": "sk-or-...",
        "max_tokens": 2048,
        "temperature": 0.7,
        "reasoning": False,
        "notes": "Fine-tuned on instruction data",
    }}}


class SubmitResponse(BaseModel):
    job_id: str
    agent_id: str
    status: str
    message: str


# ── Jobs ──────────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    agent_id: str
    status: str                      # queued | running | done | failed
    n_runs_total: int = 0
    n_runs_done: int = 0
    error: Optional[str] = None
    scores_url: Optional[str] = None  # GitHub URL once uploaded


# ── Ranking ───────────────────────────────────────────────────────────────────

class DomainScore(BaseModel):
    domain: str
    mean_y: float
    n_items: int


class LeaderboardEntry(BaseModel):
    rank: int
    agent_id: str
    display_name: str
    mean_y: float                    # mean polytomous score across all items (0-2)
    mean_success_rate: float
    n_items_evaluated: int
    domain_scores: list[DomainScore]
    notes: str = ""


class RankingResponse(BaseModel):
    leaderboard: list[LeaderboardEntry]
    n_items_total: int
    scoring_thresholds: dict[str, float] = {"tau1": 0.20, "tau2": 0.75}


# ── Items ─────────────────────────────────────────────────────────────────────

class ItemInfo(BaseModel):
    id: str
    domain: str
    subdomain: str
    tier: int
    prompt_preview: str              # first 200 chars of rendered prompt (no variables filled)
    criterion_type: str
    n_agents_evaluated: int = 0
    mean_success_rate: Optional[float] = None


class ItemsResponse(BaseModel):
    items: list[ItemInfo]
    domains: dict[str, list[str]]   # domain → list of subdomains
    total: int

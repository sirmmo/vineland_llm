"""Pydantic data models for all data shapes in the runner."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ExactMatchCriterion(BaseModel):
    type: Literal["exact_match"]
    expected_answer: Optional[str] = None
    expected_regex: Optional[str] = None

    @model_validator(mode="after")
    def check_at_least_one(self) -> "ExactMatchCriterion":
        if self.expected_answer is None and self.expected_regex is None:
            raise ValueError("exact_match must specify expected_answer or expected_regex")
        return self


class LLMJudgeCriterion(BaseModel):
    type: Literal["llm_judge"]
    judge_prompt: str
    judge_parse: str  # regex pattern; match → success


class JudgeYesNoCriterion(BaseModel):
    """Judge extracts first YES/NO token. success = (verdict == 'YES')."""
    type: Literal["judge_yesno"]
    judge_prompt: str
    judge_parse: Optional[str] = None  # optional override; default = first YES|NO


class JudgePassFailCriterion(BaseModel):
    """Judge extracts LAST PASS|PARTIAL|FAIL token. success = (verdict == 'PASS')."""
    type: Literal["judge_passfail"]
    judge_prompt: str
    judge_parse: Optional[str] = None  # optional override; default = last PASS|PARTIAL|FAIL


SuccessCriterion = Union[
    ExactMatchCriterion,
    LLMJudgeCriterion,
    JudgeYesNoCriterion,
    JudgePassFailCriterion,
]


class Item(BaseModel):
    id: str
    domain: str
    subdomain: str
    tier: int = Field(ge=1, le=8)
    declared_tier: Optional[int] = Field(default=None, ge=1, le=8)
    observed_tier: Optional[int] = Field(default=None, ge=1, le=8)
    prompt_template: str
    prompt_variables: dict[str, str] = Field(default_factory=dict)
    success_criterion: SuccessCriterion = Field(discriminator="type")

    @model_validator(mode="before")
    @classmethod
    def _fill_tier_from_declared(cls, data: Any) -> Any:
        """Accept items that specify only declared_tier (phase-2 style)."""
        if not isinstance(data, dict):
            return data
        if "tier" not in data and data.get("declared_tier") is not None:
            data = {**data, "tier": data["declared_tier"]}
        return data

    def rendered_prompt(self) -> str:
        result = self.prompt_template
        for key, value in self.prompt_variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result


class Agent(BaseModel):
    id: str
    display_name: str
    base_url: str
    model_id: str
    api_key_env: str
    max_tokens: int = 2048
    temperature: float = 0.7
    reasoning: bool = False
    wait_between_requests_s: float = 0.0
    notes: str = ""


class PilotConfig(BaseModel):
    name: str
    agents: list[str]
    items: Union[Literal["auto", "all"], list[str]]
    items_per_subdomain: int = 2
    n_replications: int = 5
    output_dir: str
    judge_agent: Optional[str] = None
    max_concurrency: int = 4
    null_success_as_zero: bool = False


class APIResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)
    seed_unsupported: bool = False


class RunRecord(BaseModel):
    run_id: str
    agent_id: str
    item_id: str
    replication: int
    prompt: str
    response: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    success: Optional[bool] = None
    verdict: Optional[str] = None  # "PASS"|"PARTIAL"|"FAIL"|"YES"|"NO"|None
    grading_detail: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = ""
    seed_unsupported: bool = False


class ScoreRow(BaseModel):
    agent_id: str
    item_id: str
    domain: str
    subdomain: str
    tier: int
    n_reps: int
    n_successes: int
    s: float          # raw success rate (backward-compat)
    y: int            # polytomous score 0/1/2 (computed on s_partial_weighted)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_latency_s: float = 0.0
    mean_latency_s: float = 0.0
    # Verdict distribution (Vineland-Extended)
    n_partial: int = 0
    n_fail: int = 0
    n_yes: int = 0
    n_no: int = 0
    n_null: int = 0
    n_empty_response: int = 0
    s_partial_weighted: float = 0.0
    grader_type: str = ""

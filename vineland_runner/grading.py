"""Graders: exact_match and llm_judge."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .types import Agent, ExactMatchCriterion, Item, LLMJudgeCriterion


@dataclass
class GradeResult:
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)


class Grader(Protocol):
    async def grade(
        self,
        item: Item,
        response: str,
        judge_agent: Agent | None = None,
        judge_api_key: str | None = None,
    ) -> GradeResult: ...


class ExactMatchGrader:
    async def grade(
        self,
        item: Item,
        response: str,
        judge_agent: Agent | None = None,
        judge_api_key: str | None = None,
    ) -> GradeResult:
        criterion: ExactMatchCriterion = item.success_criterion  # type: ignore[assignment]

        if criterion.expected_regex is not None:
            match = re.search(criterion.expected_regex, response, re.IGNORECASE | re.DOTALL)
            matched_text = match.group(0) if match else None
            return GradeResult(
                success=match is not None,
                detail={"pattern": criterion.expected_regex, "matched_text": matched_text},
            )

        # Fallback: exact string containment
        success = (criterion.expected_answer or "") in response
        return GradeResult(
            success=success,
            detail={"expected": criterion.expected_answer, "found": success},
        )


class LLMJudgeGrader:
    def __init__(self, client: Any):
        self._client = client  # LLMClient

    async def grade(
        self,
        item: Item,
        response: str,
        judge_agent: Agent | None = None,
        judge_api_key: str | None = None,
    ) -> GradeResult:
        criterion: LLMJudgeCriterion = item.success_criterion  # type: ignore[assignment]

        if judge_agent is None or judge_api_key is None:
            raise ValueError(f"llm_judge item {item.id} requires a judge_agent")

        judge_prompt = criterion.judge_prompt.format(response=response)
        judge_response = await self._client.complete(
            judge_agent,
            judge_api_key,
            [{"role": "user", "content": judge_prompt}],
        )

        judge_output = judge_response.content
        match = re.search(criterion.judge_parse, judge_output, re.IGNORECASE)

        return GradeResult(
            success=match is not None,
            detail={
                "judge_output": judge_output,
                "judge_tokens": judge_response.prompt_tokens + judge_response.completion_tokens,
                "pattern": criterion.judge_parse,
            },
        )


def make_grader(item: Item, client: Any) -> Grader:
    if item.success_criterion.type == "exact_match":
        return ExactMatchGrader()
    return LLMJudgeGrader(client)

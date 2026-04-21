"""Graders: exact_match, llm_judge (legacy), judge_yesno, judge_passfail."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .scoring import _PASSFAIL_RE, _YESNO_RE
from .types import (
    Agent,
    ExactMatchCriterion,
    Item,
    JudgePassFailCriterion,
    JudgeYesNoCriterion,
    LLMJudgeCriterion,
)


@dataclass
class GradeResult:
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)
    verdict: Optional[str] = None  # "PASS"|"PARTIAL"|"FAIL"|"YES"|"NO"|None


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
    """Legacy llm_judge grader: uses the item-provided regex."""
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


class JudgeYesNoGrader:
    """Judge returns free-form text ending in YES/NO. Extract first match."""
    def __init__(self, client: Any):
        self._client = client

    async def grade(
        self,
        item: Item,
        response: str,
        judge_agent: Agent | None = None,
        judge_api_key: str | None = None,
    ) -> GradeResult:
        criterion: JudgeYesNoCriterion = item.success_criterion  # type: ignore[assignment]
        if judge_agent is None or judge_api_key is None:
            raise ValueError(f"judge_yesno item {item.id} requires a judge_agent")

        judge_prompt = criterion.judge_prompt.format(response=response)
        judge_response = await self._client.complete(
            judge_agent,
            judge_api_key,
            [{"role": "user", "content": judge_prompt}],
        )

        judge_output = judge_response.content
        pattern = criterion.judge_parse
        verdict: Optional[str] = None
        if pattern:
            m = re.search(pattern, judge_output, re.IGNORECASE)
            if m:
                verdict = (m.group(1) if m.groups() else m.group(0)).upper()
        else:
            m = _YESNO_RE.search(judge_output)
            if m:
                verdict = m.group(1).upper()

        return GradeResult(
            success=(verdict == "YES"),
            verdict=verdict,
            detail={
                "judge_output": judge_output,
                "judge_tokens": judge_response.prompt_tokens + judge_response.completion_tokens,
            },
        )


class JudgePassFailGrader:
    """Judge returns a rubric analysis ending with VERDICT: PASS|PARTIAL|FAIL.
    Extract the LAST such token (rubric may quote earlier ones)."""
    def __init__(self, client: Any):
        self._client = client

    async def grade(
        self,
        item: Item,
        response: str,
        judge_agent: Agent | None = None,
        judge_api_key: str | None = None,
    ) -> GradeResult:
        criterion: JudgePassFailCriterion = item.success_criterion  # type: ignore[assignment]
        if judge_agent is None or judge_api_key is None:
            raise ValueError(f"judge_passfail item {item.id} requires a judge_agent")

        judge_prompt = criterion.judge_prompt.format(response=response)
        judge_response = await self._client.complete(
            judge_agent,
            judge_api_key,
            [{"role": "user", "content": judge_prompt}],
        )

        judge_output = judge_response.content
        verdict: Optional[str] = None
        pattern = criterion.judge_parse
        if pattern:
            matches = re.findall(pattern, judge_output, re.IGNORECASE)
            if matches:
                raw = matches[-1]
                verdict = (raw if isinstance(raw, str) else raw[0]).upper()
        else:
            matches = _PASSFAIL_RE.findall(judge_output)
            if matches:
                verdict = matches[-1].upper()

        return GradeResult(
            success=(verdict == "PASS"),
            verdict=verdict,
            detail={
                "judge_output": judge_output,
                "judge_tokens": judge_response.prompt_tokens + judge_response.completion_tokens,
            },
        )


def make_grader(item: Item, client: Any) -> Grader:
    ctype = item.success_criterion.type
    if ctype == "exact_match":
        return ExactMatchGrader()
    if ctype == "judge_yesno":
        return JudgeYesNoGrader(client)
    if ctype == "judge_passfail":
        return JudgePassFailGrader(client)
    # llm_judge (legacy)
    return LLMJudgeGrader(client)

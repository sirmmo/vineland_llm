"""Tests for ExactMatchGrader and LLMJudgeGrader."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from vineland_runner.grading import ExactMatchGrader, LLMJudgeGrader, GradeResult
from vineland_runner.types import (
    Agent,
    APIResponse,
    ExactMatchCriterion,
    Item,
    LLMJudgeCriterion,
)


def _make_item(criterion_data: dict) -> Item:
    from pydantic import TypeAdapter
    from vineland_runner.types import SuccessCriterion
    return Item(
        id="CM-REC-1",
        domain="CM",
        subdomain="REC",
        tier=1,
        prompt_template="Test prompt",
        prompt_variables={},
        success_criterion=criterion_data,
    )


# --- ExactMatchGrader ---

@pytest.mark.asyncio
async def test_exact_match_regex_success():
    item = _make_item({"type": "exact_match", "expected_regex": '"tool":\\s*"calculator"'})
    grader = ExactMatchGrader()
    result = await grader.grade(item, '{"tool": "calculator", "args": {"expression": "17*340/100"}}')
    assert result.success is True
    assert result.detail["matched_text"] is not None


@pytest.mark.asyncio
async def test_exact_match_regex_failure():
    item = _make_item({"type": "exact_match", "expected_regex": '"tool":\\s*"calculator"'})
    grader = ExactMatchGrader()
    result = await grader.grade(item, '{"tool": "search", "args": {"query": "17% of 340"}}')
    assert result.success is False
    assert result.detail["matched_text"] is None


@pytest.mark.asyncio
async def test_exact_match_string_success():
    item = _make_item({"type": "exact_match", "expected_answer": "Paris"})
    grader = ExactMatchGrader()
    result = await grader.grade(item, "The capital of France is Paris.")
    assert result.success is True


@pytest.mark.asyncio
async def test_exact_match_string_failure():
    item = _make_item({"type": "exact_match", "expected_answer": "Paris"})
    grader = ExactMatchGrader()
    result = await grader.grade(item, "The capital of France is Lyon.")
    assert result.success is False


# --- LLMJudgeGrader ---

@pytest.mark.asyncio
async def test_llm_judge_success():
    item = _make_item({
        "type": "llm_judge",
        "judge_prompt": "Assess this: {response}. Say YES or NO.",
        "judge_parse": "^YES",
    })

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=APIResponse(
        content="YES — the response is correct.",
        prompt_tokens=10,
        completion_tokens=8,
    ))

    mock_agent = Agent(
        id="judge",
        display_name="Judge",
        base_url="https://api.example.com/v1",
        model_id="judge-model",
        api_key_env="TEST_KEY",
    )

    grader = LLMJudgeGrader(mock_client)
    result = await grader.grade(item, "Exercise improves mood.", mock_agent, "test-key")

    assert result.success is True
    assert "judge_output" in result.detail
    assert result.detail["judge_tokens"] == 18


@pytest.mark.asyncio
async def test_llm_judge_failure():
    item = _make_item({
        "type": "llm_judge",
        "judge_prompt": "Assess: {response}. YES or NO.",
        "judge_parse": "^YES",
    })

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=APIResponse(
        content="NO — the response misses the point.",
        prompt_tokens=10,
        completion_tokens=8,
    ))

    mock_agent = Agent(
        id="judge",
        display_name="Judge",
        base_url="https://api.example.com/v1",
        model_id="judge-model",
        api_key_env="TEST_KEY",
    )

    grader = LLMJudgeGrader(mock_client)
    result = await grader.grade(item, "Exercise is bad.", mock_agent, "test-key")

    assert result.success is False


@pytest.mark.asyncio
async def test_llm_judge_requires_agent():
    item = _make_item({
        "type": "llm_judge",
        "judge_prompt": "Assess: {response}. YES or NO.",
        "judge_parse": "^YES",
    })
    mock_client = MagicMock()
    grader = LLMJudgeGrader(mock_client)

    with pytest.raises(ValueError, match="judge_agent"):
        await grader.grade(item, "some response", judge_agent=None, judge_api_key=None)

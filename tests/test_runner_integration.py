"""Integration tests: mock API, full run loop, resumability."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
respx = pytest.importorskip("respx")
import httpx

from vineland_runner.client import LLMClient
from vineland_runner.runner import run_pilot
from vineland_runner.storage import iter_records, load_completed_ids
from vineland_runner.types import Agent, APIResponse, Item, PilotConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_agent(agent_id: str = "test-agent") -> Agent:
    return Agent(
        id=agent_id,
        display_name="Test Agent",
        base_url="https://mock.api/v1",
        model_id="test-model",
        api_key_env="TEST_API_KEY",
        max_tokens=256,
        temperature=0.7,
    )


def _make_item_exact(item_id: str, domain: str = "AF", subdomain: str = "TOL") -> Item:
    return Item(
        id=item_id,
        domain=domain,
        subdomain=subdomain,
        tier=1,
        prompt_template='Choose a tool: {question}',
        prompt_variables={"question": "What is 2+2?"},
        success_criterion={"type": "exact_match", "expected_regex": "calculator"},
    )


def _make_item_judge(item_id: str) -> Item:
    return Item(
        id=item_id,
        domain="CM",
        subdomain="REC",
        tier=1,
        prompt_template="Answer: {q}",
        prompt_variables={"q": "What improves mood?"},
        success_criterion={
            "type": "llm_judge",
            "judge_prompt": "Does this answer mention exercise? {response} YES/NO",
            "judge_parse": "^YES",
        },
    )


def _openai_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "completion_tokens_details": {}},
    }


# ── API mock helper ────────────────────────────────────────────────────────────

def _mock_route(router: respx.MockRouter, url: str, content: str):
    router.post(url).mock(return_value=httpx.Response(
        200,
        json=_openai_response(content),
    ))


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_run_produces_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """1 agent × 2 items × 3 reps = 6 JSONL entries."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test")

    agent = _make_agent()
    items = [_make_item_exact("AF-TOL-1"), _make_item_exact("AF-TOL-2")]
    config = PilotConfig(
        name="test",
        agents=["test-agent"],
        items="all",
        n_replications=3,
        output_dir=str(tmp_path / "runs"),
        judge_agent=None,
        max_concurrency=2,
    )

    with respx.mock(base_url="https://mock.api") as router:
        router.post("/v1/chat/completions").mock(return_value=httpx.Response(
            200, json=_openai_response('{"tool": "calculator"}')
        ))

        async with httpx.AsyncClient(timeout=30) as http:
            client = LLMClient(http)
            await run_pilot(config, items, {"test-agent": agent}, client)

    runs_path = tmp_path / "runs" / "runs.jsonl"
    records = list(iter_records(runs_path))

    assert len(records) == 6
    run_ids = {r.run_id for r in records}
    assert len(run_ids) == 6  # all unique

    # Grading: response contains "calculator" → success=True
    assert all(r.success is True for r in records)


@pytest.mark.asyncio
async def test_resumability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Simulate kill mid-run: pre-populate 2 of 6 records, verify only 4 new ones written."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test")

    agent = _make_agent()
    items = [_make_item_exact("AF-TOL-1"), _make_item_exact("AF-TOL-2")]
    config = PilotConfig(
        name="test",
        agents=["test-agent"],
        items="all",
        n_replications=3,
        output_dir=str(tmp_path / "runs"),
        judge_agent=None,
        max_concurrency=2,
    )

    runs_path = tmp_path / "runs" / "runs.jsonl"
    runs_path.parent.mkdir(parents=True)

    # Pre-populate 2 completed runs (simulating a killed run)
    pre_done = [
        {
            "run_id": "test-agent__AF-TOL-1__rep0",
            "agent_id": "test-agent",
            "item_id": "AF-TOL-1",
            "replication": 0,
            "prompt": "test",
            "response": '{"tool": "calculator"}',
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "reasoning_tokens": 0,
            "latency_s": 0.1,
            "success": True,
            "grading_detail": {},
            "error": None,
            "timestamp": "2026-04-17T00:00:00+00:00",
            "seed_unsupported": False,
        },
        {
            "run_id": "test-agent__AF-TOL-1__rep1",
            "agent_id": "test-agent",
            "item_id": "AF-TOL-1",
            "replication": 1,
            "prompt": "test",
            "response": '{"tool": "calculator"}',
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "reasoning_tokens": 0,
            "latency_s": 0.1,
            "success": True,
            "grading_detail": {},
            "error": None,
            "timestamp": "2026-04-17T00:00:00+00:00",
            "seed_unsupported": False,
        },
    ]
    with open(runs_path, "w") as f:
        for rec in pre_done:
            f.write(json.dumps(rec) + "\n")

    call_count = 0

    with respx.mock(base_url="https://mock.api") as router:
        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=_openai_response('{"tool": "calculator"}'))

        router.post("/v1/chat/completions").mock(side_effect=handler)

        async with httpx.AsyncClient(timeout=30) as http:
            client = LLMClient(http)
            await run_pilot(config, items, {"test-agent": agent}, client)

    # Should have made exactly 4 new API calls (6 total - 2 already done)
    assert call_count == 4

    records = list(iter_records(runs_path))
    assert len(records) == 6  # 2 pre-existing + 4 new

    run_ids = {r.run_id for r in records}
    assert "test-agent__AF-TOL-1__rep0" in run_ids
    assert "test-agent__AF-TOL-1__rep1" in run_ids


@pytest.mark.asyncio
async def test_api_error_logged_as_null_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A 500 after all retries logs success=null with error field, run continues."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test")

    agent = _make_agent()
    items = [_make_item_exact("AF-TOL-1")]
    config = PilotConfig(
        name="test",
        agents=["test-agent"],
        items="all",
        n_replications=1,
        output_dir=str(tmp_path / "runs"),
        judge_agent=None,
        max_concurrency=1,
    )

    with respx.mock(base_url="https://mock.api") as router:
        router.post("/v1/chat/completions").mock(return_value=httpx.Response(500, text="Server Error"))

        async with httpx.AsyncClient(timeout=30) as http:
            client = LLMClient(http)
            await run_pilot(config, items, {"test-agent": agent}, client)

    records = list(iter_records(tmp_path / "runs" / "runs.jsonl"))
    assert len(records) == 1
    assert records[0].success is None
    assert records[0].error is not None


@pytest.mark.asyncio
async def test_judge_grading_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """llm_judge items: verify judge is called and grading uses its response."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test")

    agent = _make_agent()
    judge_agent = _make_agent("judge-agent")
    judge_agent = judge_agent.model_copy(update={"api_key_env": "TEST_API_KEY"})

    items = [_make_item_judge("CM-REC-1")]
    config = PilotConfig(
        name="test",
        agents=["test-agent"],
        items="all",
        n_replications=2,
        output_dir=str(tmp_path / "runs"),
        judge_agent="judge-agent",
        max_concurrency=1,
    )
    agents = {"test-agent": agent, "judge-agent": judge_agent}

    call_contents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # First call: model response; second call: judge
        content = "Exercise improves mood." if not call_contents else "YES — the response mentions exercise."
        call_contents.append(content)
        return httpx.Response(200, json=_openai_response(content))

    with respx.mock(base_url="https://mock.api") as router:
        router.post("/v1/chat/completions").mock(side_effect=handler)

        async with httpx.AsyncClient(timeout=30) as http:
            client = LLMClient(http)
            await run_pilot(config, items, agents, client)

    records = list(iter_records(tmp_path / "runs" / "runs.jsonl"))
    assert len(records) == 2
    for rec in records:
        assert rec.success is True
        assert "judge_output" in rec.grading_detail

"""Tests for polytomous scoring logic."""
import json
from pathlib import Path

import pytest

from vineland_runner.scoring import compute_scores
from vineland_runner.types import RunRecord


def _write_runs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _run(
    agent_id: str,
    item_id: str,
    rep: int,
    success: bool | None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    latency_s: float = 0.5,
) -> dict:
    return {
        "run_id": f"{agent_id}__{item_id}__rep{rep}",
        "agent_id": agent_id,
        "item_id": item_id,
        "replication": rep,
        "prompt": "test",
        "response": "test",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": 0,
        "latency_s": latency_s,
        "success": success,
        "grading_detail": {},
        "error": None,
        "timestamp": "2026-04-17T00:00:00+00:00",
        "seed_unsupported": False,
    }


def test_score_all_success(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [_run("a1", "item1", i, True) for i in range(5)])
    scores = compute_scores(runs)
    assert len(scores) == 1
    assert scores[0].y == 2
    assert scores[0].s == 1.0
    assert scores[0].n_successes == 5


def test_score_all_failure(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [_run("a1", "item1", i, False) for i in range(5)])
    scores = compute_scores(runs)
    assert scores[0].y == 0
    assert scores[0].s == 0.0


def test_score_partial_low(tmp_path: Path):
    # 1/5 successes = 0.20 → exactly at tau1 boundary → y=1 (not 0; 0.20 < tau1 is False)
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [_run("a1", "item1", i, i == 0) for i in range(5)])
    scores = compute_scores(runs)
    # s=0.20, tau1=0.20, s < tau1 is False, s < tau2 is True → y=1
    assert scores[0].s == 0.2
    assert scores[0].y == 1


def test_score_partial_high(tmp_path: Path):
    # 4/5 = 0.80 >= tau2 → y=2
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [_run("a1", "item1", i, i < 4) for i in range(5)])
    scores = compute_scores(runs)
    assert scores[0].y == 2
    assert scores[0].n_successes == 4


def test_score_custom_thresholds(tmp_path: Path):
    # 2/5 = 0.40; with tau1=0.50 → y=0
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [_run("a1", "item1", i, i < 2) for i in range(5)])
    scores = compute_scores(runs, tau1=0.50, tau2=0.90)
    assert scores[0].y == 0


def test_score_null_excluded_by_default(tmp_path: Path):
    # 3 successes, 2 nulls → only 3 valid → s=1.0 → y=2
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [
        _run("a1", "item1", 0, True),
        _run("a1", "item1", 1, True),
        _run("a1", "item1", 2, True),
        _run("a1", "item1", 3, None),
        _run("a1", "item1", 4, None),
    ])
    scores = compute_scores(runs)
    assert scores[0].n_reps == 3
    assert scores[0].y == 2


def test_score_null_as_zero(tmp_path: Path):
    # 3 success + 2 null as zero → s = 3/5 = 0.60 → y=1
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [
        _run("a1", "item1", 0, True),
        _run("a1", "item1", 1, True),
        _run("a1", "item1", 2, True),
        _run("a1", "item1", 3, None),
        _run("a1", "item1", 4, None),
    ])
    scores = compute_scores(runs, null_as_zero=True)
    assert scores[0].n_reps == 5
    assert scores[0].s == pytest.approx(0.60)
    assert scores[0].y == 1


def test_score_token_and_latency_aggregation(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    _write_runs(runs, [
        _run("a1", "item1", i, True, prompt_tokens=100, completion_tokens=50, latency_s=1.0)
        for i in range(4)
    ])
    scores = compute_scores(runs)
    assert scores[0].total_prompt_tokens == 400
    assert scores[0].total_completion_tokens == 200
    assert scores[0].total_latency_s == pytest.approx(4.0)
    assert scores[0].mean_latency_s == pytest.approx(1.0)


def test_score_multiple_agents_items(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    records = []
    for agent in ["a1", "a2"]:
        for item in ["i1", "i2"]:
            for rep in range(5):
                records.append(_run(agent, item, rep, True))
    _write_runs(runs, records)
    scores = compute_scores(runs)
    assert len(scores) == 4
    assert all(s.y == 2 for s in scores)

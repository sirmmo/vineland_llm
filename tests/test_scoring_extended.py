"""Tests for the Vineland-Extended scoring extensions (partial credit,
regrade, profiles, diagnostics, rankings)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vineland_runner.diagnostics import compute_item_diagnostics
from vineland_runner.profile import compute_profiles
from vineland_runner.ranking import compute_rankings
from vineland_runner.scoring import (
    compute_scores,
    load_items_meta,
    regrade_runs,
)
from vineland_runner.types import ScoreRow


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _rec(
    agent_id: str = "a1",
    item_id: str = "AF-PLN-1",
    rep: int = 0,
    success: bool | None = True,
    verdict: str | None = None,
    response: str | None = "resp",
    error: str | None = None,
    grading_detail: dict | None = None,
) -> dict:
    return {
        "run_id": f"{agent_id}__{item_id}__rep{rep}",
        "agent_id": agent_id,
        "item_id": item_id,
        "replication": rep,
        "prompt": "p",
        "response": response,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 0,
        "latency_s": 0.5,
        "success": success,
        "verdict": verdict,
        "grading_detail": grading_detail or {},
        "error": error,
        "timestamp": "2026-04-17T00:00:00+00:00",
        "seed_unsupported": False,
    }


# ── 1. PARTIAL credit test ────────────────────────────────────────────────────

def test_partial_credit_weighting(tmp_path: Path):
    """3 PASS + 1 PARTIAL + 1 FAIL → s=0.60, s_partial_weighted=0.70.

    y is computed on s_partial_weighted with τ₁=0.20, τ₂=0.75. 0.70 < 0.75 → y=1.
    """
    runs = tmp_path / "runs.jsonl"
    _write_jsonl(runs, [
        _rec(rep=0, success=True, verdict="PASS"),
        _rec(rep=1, success=True, verdict="PASS"),
        _rec(rep=2, success=True, verdict="PASS"),
        _rec(rep=3, success=False, verdict="PARTIAL"),
        _rec(rep=4, success=False, verdict="FAIL"),
    ])
    [row] = compute_scores(
        runs,
        items_grader_type={"AF-PLN-1": "judge_passfail"},
        partial_credit=0.5,
    )
    assert row.s == pytest.approx(0.60)
    assert row.s_partial_weighted == pytest.approx(0.70)
    assert row.y == 1
    assert row.n_partial == 1
    assert row.n_fail == 1


def test_partial_credit_pushes_to_pass(tmp_path: Path):
    """4 PASS + 1 PARTIAL → s_partial=(4+0.5)/5=0.90 → y=2 (≥τ₂=0.75)."""
    runs = tmp_path / "runs.jsonl"
    _write_jsonl(runs, [
        _rec(rep=i, success=True, verdict="PASS") for i in range(4)
    ] + [_rec(rep=4, success=False, verdict="PARTIAL")])
    [row] = compute_scores(
        runs,
        items_grader_type={"AF-PLN-1": "judge_passfail"},
    )
    assert row.s_partial_weighted == pytest.approx(0.90)
    assert row.y == 2


# ── 2. Null / error exclusion ─────────────────────────────────────────────────

def test_null_error_excluded_single_valid_survives(tmp_path: Path):
    """3 reps = 1 success + 1 error + 1 null → n_reps=1 (low reliability)."""
    runs = tmp_path / "runs.jsonl"
    _write_jsonl(runs, [
        _rec(rep=0, success=True, verdict="PASS"),
        _rec(rep=1, success=None, error="HTTPStatusError: 500"),
        _rec(rep=2, success=None, response=""),  # null, empty response
    ])
    [row] = compute_scores(runs, items_grader_type={"AF-PLN-1": "judge_passfail"})
    assert row.n_reps == 1
    assert row.n_successes == 1
    assert row.s == 1.0
    assert row.n_null == 2  # both success=None count here
    assert row.n_empty_response == 1


# ── 3. Regrade: PASS token variants ────────────────────────────────────────────

def test_regrade_extracts_last_pass_verdict(tmp_path: Path):
    """Original bug: judge_output containing **PASS** / VERDICT: PASS were missed."""
    runs_in = tmp_path / "runs_in.jsonl"
    runs_out = tmp_path / "runs_out.jsonl"
    _write_jsonl(runs_in, [
        _rec(rep=0, success=False, grading_detail={"judge_output": "... **PASS**"}),
        _rec(rep=1, success=False, grading_detail={"judge_output": "VERDICT: PASS"}),
        _rec(rep=2, success=False, grading_detail={"judge_output": "PASS"}),
        _rec(rep=3, success=False, grading_detail={"judge_output": "PARTIAL noise blah"}),
        _rec(rep=4, success=False, grading_detail={"judge_output": "Discussion: PASS initially looked ok but FAIL"}),
    ])
    counts = regrade_runs(runs_in, {"AF-PLN-1": "judge_passfail"}, runs_out)
    assert counts["total"] == 5
    assert counts["updated"] == 5
    assert counts["parse_failed"] == 0

    regraded = [json.loads(line) for line in runs_out.read_text().splitlines() if line.strip()]
    assert regraded[0]["verdict"] == "PASS"
    assert regraded[0]["success"] is True
    assert regraded[1]["verdict"] == "PASS"
    assert regraded[1]["success"] is True
    assert regraded[2]["verdict"] == "PASS"
    assert regraded[2]["success"] is True
    assert regraded[3]["verdict"] == "PARTIAL"
    assert regraded[3]["success"] is False
    # Last-match rule: "FAIL" beats earlier "PASS"
    assert regraded[4]["verdict"] == "FAIL"
    assert regraded[4]["success"] is False


def test_regrade_yesno(tmp_path: Path):
    runs_in = tmp_path / "runs_in.jsonl"
    runs_out = tmp_path / "runs_out.jsonl"
    _write_jsonl(runs_in, [
        _rec(rep=0, success=False, grading_detail={"judge_output": "YES — improves mood."}),
        _rec(rep=1, success=True, grading_detail={"judge_output": "NO evidence."}),
    ])
    regrade_runs(runs_in, {"AF-PLN-1": "judge_yesno"}, runs_out)
    rows = [json.loads(l) for l in runs_out.read_text().splitlines() if l.strip()]
    assert rows[0]["verdict"] == "YES"
    assert rows[0]["success"] is True
    assert rows[1]["verdict"] == "NO"
    assert rows[1]["success"] is False


# ── 4. Empty response detection ───────────────────────────────────────────────

def test_empty_response_flagged(tmp_path: Path):
    runs = tmp_path / "runs.jsonl"
    _write_jsonl(runs, [
        _rec(rep=0, success=None, response="", error=None),  # empty response
        _rec(rep=1, success=None, response=None, error=None),  # null response
        _rec(rep=2, success=True, verdict="PASS"),
    ])
    [row] = compute_scores(runs, items_grader_type={"AF-PLN-1": "judge_passfail"})
    assert row.n_empty_response == 2
    assert row.n_null == 2


# ── 5. Profile computation ────────────────────────────────────────────────────

def _score(agent_id: str, item_id: str, domain: str, subdomain: str, y: int,
           s: float = 1.0, n_reps: int = 5) -> ScoreRow:
    return ScoreRow(
        agent_id=agent_id, item_id=item_id, domain=domain, subdomain=subdomain,
        tier=3, n_reps=n_reps, n_successes=int(s * n_reps),
        s=s, y=y, s_partial_weighted=s,
    )


def test_profile_missing_dim_is_none():
    # Agent a1 has only AF items → theta_CM should be None
    scores = [_score("a1", "AF-TOL-1", "AF", "TOL", y=2)]
    [p] = compute_profiles(scores)
    assert p.theta_CM is None
    assert p.theta_AF_TOL == 2.0
    assert p.n_items_AF_TOL == 1
    # Only one item → stdev undefined, se is None
    assert p.se_theta_AF_TOL is None


def test_profile_multi_item_se():
    scores = [
        _score("a1", "CM-1", "CM", "EXP", y=2),
        _score("a1", "CM-2", "CM", "EXP", y=0),
        _score("a1", "CM-3", "CM", "REC", y=1),
    ]
    [p] = compute_profiles(scores)
    assert p.theta_CM == 1.0
    assert p.n_items_CM == 3
    assert p.se_theta_CM is not None and p.se_theta_CM > 0


def test_profile_af_overall_aggregates_all_af_items():
    scores = [
        _score("a1", "AF-TOL-1", "AF", "TOL", y=2),
        _score("a1", "AF-PLN-1", "AF", "PLN", y=0),
    ]
    [p] = compute_profiles(scores)
    assert p.theta_AF_overall == 1.0


# ── 6. Diagnostic flags ───────────────────────────────────────────────────────

def test_diagnostic_floor_and_zero_variance():
    # All 5 agents at s=0.0 on this item
    scores = [
        _score(f"agent{i}", "CM-EXP-4", "CM", "EXP", y=0, s=0.0) for i in range(5)
    ]
    [diag] = compute_item_diagnostics(scores)
    assert diag.is_floor is True
    assert diag.is_zero_variance is True
    assert diag.is_problematic is True
    assert any("floor" in r for r in diag.flag_reasons)


def test_diagnostic_bimodal():
    # Half the agents at 1.0, half at 0.0 → bimodal
    rows = (
        [_score(f"hi{i}", "DLS-DOM-2", "DLS", "DOM", y=2, s=1.0) for i in range(4)]
        + [_score(f"lo{i}", "DLS-DOM-2", "DLS", "DOM", y=0, s=0.0) for i in range(4)]
    )
    [diag] = compute_item_diagnostics(rows)
    assert diag.is_bimodal is True
    assert diag.is_problematic is True


def test_diagnostic_ceiling():
    scores = [_score(f"a{i}", "CM-EXP-1", "CM", "EXP", y=2, s=1.0) for i in range(5)]
    [diag] = compute_item_diagnostics(scores)
    assert diag.is_ceiling is True
    # ceiling items are also zero-variance here — both flags valid
    assert diag.is_zero_variance is True


# ── 7. Ranking min_n_items exclusion ─────────────────────────────────────────

def test_ranking_excludes_below_min_n_items():
    # a1 has 3 CM items, a2 has 2 CM items
    scores = [
        _score("a1", "CM-1", "CM", "EXP", y=2),
        _score("a1", "CM-2", "CM", "EXP", y=2),
        _score("a1", "CM-3", "CM", "REC", y=2),
        _score("a2", "CM-1", "CM", "EXP", y=1),
        _score("a2", "CM-2", "CM", "EXP", y=1),
    ]
    profiles = compute_profiles(scores)
    rankings = compute_rankings(profiles, min_n_items=3)
    cm = next(r for r in rankings if r.dimension == "theta_CM")
    agent_ids = [r["agent_id"] for r in cm.ranked_agents]
    assert "a1" in agent_ids
    assert "a2" not in agent_ids  # excluded (n_items_CM=2 < 3)


# ── 8. load_items_meta handles judge_passfail items ──────────────────────────

def test_load_items_meta_accepts_judge_passfail(tmp_path: Path):
    """Must bypass Item.model_validate so judge_passfail items yield meta."""
    items_yaml = tmp_path / "items.yaml"
    items_yaml.write_text(
        "items:\n"
        "  - id: AF-PLN-2\n"
        "    domain: AF\n"
        "    subdomain: PLN\n"
        "    tier: 3\n"
        "    prompt_template: 'Plan: {task}'\n"
        "    prompt_variables:\n"
        "      task: 'cook'\n"
        "    success_criterion:\n"
        "      type: judge_passfail\n"
        "      judge_prompt: 'Evaluate...'\n"
        "      judge_parse: 'VERDICT'\n"
    )
    meta = load_items_meta(items_yaml)
    assert meta["AF-PLN-2"]["grader_type"] == "judge_passfail"
    assert meta["AF-PLN-2"]["domain"] == "AF"

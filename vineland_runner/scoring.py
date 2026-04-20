"""Polytomous 0/1/2 scoring + PARTIAL-credit extensions for Vineland-Extended."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .storage import iter_records
from .types import RunRecord, ScoreRow


# ── verdict extraction patterns (post-hoc regrade) ────────────────────────────
_PASSFAIL_RE = re.compile(r"(?m)(?:^|\W)(PASS|PARTIAL|FAIL)(?:\W|$)", re.IGNORECASE)
_YESNO_RE = re.compile(r"(?m)(?:^|\W)(YES|NO)(?:\W|$)", re.IGNORECASE)


# ── items meta loader ─────────────────────────────────────────────────────────

def load_items_meta(items_yaml_path: Path) -> dict[str, dict]:
    """
    Parse items.yaml and return {item_id: {domain, subdomain, declared_tier,
    observed_tier, grader_type}}.

    Does NOT go through Item.model_validate — that way items with grader types
    the live runner doesn't support (e.g. judge_passfail) still yield metadata.
    """
    with open(items_yaml_path) as f:
        data = yaml.safe_load(f) or {}

    out: dict[str, dict] = {}
    for raw in data.get("items", []):
        item_id = raw.get("id")
        if not item_id:
            continue
        crit = raw.get("success_criterion", {}) or {}
        grader_type = crit.get("type", "")
        # Normalize: the spec defines exact_match / judge_yesno / judge_passfail.
        # Legacy "llm_judge" items are treated as judge_yesno by default —
        # this matches the dominant historical parse pattern (^YES).
        if grader_type == "llm_judge":
            grader_type = "judge_yesno"
        out[item_id] = {
            "domain": raw.get("domain", ""),
            "subdomain": raw.get("subdomain", ""),
            "declared_tier": raw.get("declared_tier", raw.get("tier", 0)) or 0,
            "observed_tier": raw.get("observed_tier"),
            "grader_type": grader_type,
        }
    return out


# ── per-run verdict classification ────────────────────────────────────────────

def _extract_verdict(
    judge_output: str,
    grader_type: str,
) -> Optional[str]:
    """Extract a verdict token from judge output. Returns None if no match."""
    if not judge_output:
        return None
    if grader_type == "judge_passfail":
        matches = _PASSFAIL_RE.findall(judge_output)
        return matches[-1].upper() if matches else None  # LAST match
    if grader_type == "judge_yesno":
        m = _YESNO_RE.search(judge_output)
        return m.group(1).upper() if m else None  # FIRST match
    return None


def _is_empty_response(r: RunRecord) -> bool:
    return (
        r.success is None
        and r.error is None
        and (r.response is None or (isinstance(r.response, str) and r.response.strip() == ""))
    )


# ── core scoring ──────────────────────────────────────────────────────────────

def compute_scores(
    runs_jsonl_path: Path,
    items_meta: dict[str, tuple[str, str, int]] | None = None,
    tau1: float = 0.20,
    tau2: float = 0.75,
    null_as_zero: bool = False,
    items_grader_type: dict[str, str] | None = None,
    partial_credit: float = 0.5,
) -> list[ScoreRow]:
    """
    Group runs by (agent_id, item_id) and produce one ScoreRow per cell.

    Backward-compat: with no items_grader_type, items with no PARTIAL verdicts
    collapse to s_partial_weighted == s, so `y` is unchanged from legacy behavior.

    y is computed on s_partial_weighted (≡ s when no PARTIAL).
    """
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for rec in iter_records(runs_jsonl_path):
        groups[(rec.agent_id, rec.item_id)].append(rec)

    scores: list[ScoreRow] = []
    for (agent_id, item_id), recs in groups.items():
        grader_type = (items_grader_type or {}).get(item_id, "")

        n_partial = n_fail = n_yes = n_no = 0
        n_null = sum(1 for r in recs if r.success is None)
        n_empty_response = sum(1 for r in recs if _is_empty_response(r))

        for r in recs:
            if r.verdict == "PARTIAL":
                n_partial += 1
            elif r.verdict == "FAIL":
                n_fail += 1
            elif r.verdict == "YES":
                n_yes += 1
            elif r.verdict == "NO":
                n_no += 1

        if null_as_zero:
            evaluated = [bool(r.success) for r in recs]
            # PARTIAL that has success=None becomes 0 here (spec says null_as_zero
            # is the legacy strict mode — partial-weighting is a separate axis).
            partials_in_evaluated = n_partial
        else:
            evaluated = [bool(r.success) for r in recs if r.success is not None]
            partials_in_evaluated = sum(
                1 for r in recs if r.success is not None and r.verdict == "PARTIAL"
            )

        n_reps_valid = len(evaluated)
        if n_reps_valid == 0:
            continue

        n_successes = sum(evaluated)
        s = n_successes / n_reps_valid
        s_partial_weighted = (n_successes + partial_credit * partials_in_evaluated) / n_reps_valid

        if s_partial_weighted < tau1:
            y = 0
        elif s_partial_weighted < tau2:
            y = 1
        else:
            y = 2

        domain, subdomain, tier = "", "", 0
        if items_meta and item_id in items_meta:
            domain, subdomain, tier = items_meta[item_id]

        total_prompt = sum(r.prompt_tokens for r in recs)
        total_completion = sum(r.completion_tokens for r in recs)
        total_reasoning = sum(r.reasoning_tokens for r in recs)
        total_latency = sum(r.latency_s for r in recs)

        scores.append(ScoreRow(
            agent_id=agent_id,
            item_id=item_id,
            domain=domain,
            subdomain=subdomain,
            tier=tier,
            n_reps=n_reps_valid,
            n_successes=n_successes,
            s=round(s, 4),
            y=y,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_reasoning_tokens=total_reasoning,
            total_latency_s=round(total_latency, 3),
            mean_latency_s=round(total_latency / len(recs), 3),
            n_partial=n_partial,
            n_fail=n_fail,
            n_yes=n_yes,
            n_no=n_no,
            n_null=n_null,
            n_empty_response=n_empty_response,
            s_partial_weighted=round(s_partial_weighted, 4),
            grader_type=grader_type,
        ))

    return scores


def write_scores(
    scores: list[ScoreRow],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "scores.jsonl"
    csv_path = output_dir / "scores.csv"

    with open(jsonl_path, "w") as f:
        for row in scores:
            f.write(row.model_dump_json() + "\n")

    if scores:
        fieldnames = list(scores[0].model_fields.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in scores:
                writer.writerow(row.model_dump())

    return jsonl_path, csv_path


# ── regrade (post-hoc verdict re-extraction) ──────────────────────────────────

def regrade_runs(
    runs_jsonl_path: Path,
    items_grader_type: dict[str, str],
    output_jsonl_path: Path,
) -> dict[str, int]:
    """
    Re-parse judge_output from existing runs, set verdict + success accordingly.

    Writes a new JSONL (does not modify input).

    - judge_passfail: extract LAST PASS|PARTIAL|FAIL. success = (verdict == "PASS").
    - judge_yesno:    extract FIRST YES|NO.             success = (verdict == "YES").
    - exact_match:    preserve success, verdict=None.
    - Empty/missing judge_output → verdict=None, success=False.
    - Runs with error set → success=None preserved (null/error runs untouched).
    """
    counts = {"total": 0, "updated": 0, "parse_failed": 0}
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(runs_jsonl_path) as src, open(output_jsonl_path, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                rec = RunRecord.model_validate_json(line)
            except Exception:
                dst.write(line + "\n")
                continue

            counts["total"] += 1
            grader_type = items_grader_type.get(rec.item_id, "")

            # Error runs pass through untouched
            if rec.error is not None:
                dst.write(rec.model_dump_json() + "\n")
                continue

            if grader_type == "exact_match":
                # preserve as-is
                dst.write(rec.model_dump_json() + "\n")
                continue

            judge_output = ""
            if isinstance(rec.grading_detail, dict):
                judge_output = str(rec.grading_detail.get("judge_output", "") or "")

            verdict = _extract_verdict(judge_output, grader_type)

            if verdict is None:
                # Empty / unparseable output → mark as failed but not null
                # (unless response itself is empty, which is a runner-side issue).
                if _is_empty_response(rec):
                    rec.verdict = None
                    rec.success = None
                else:
                    rec.verdict = None
                    rec.success = False
                counts["parse_failed"] += 1
            else:
                rec.verdict = verdict
                if grader_type == "judge_passfail":
                    rec.success = (verdict == "PASS")
                elif grader_type == "judge_yesno":
                    rec.success = (verdict == "YES")
                counts["updated"] += 1

            dst.write(rec.model_dump_json() + "\n")

    return counts


# ── summary (markdown) ────────────────────────────────────────────────────────

def build_summary(
    scores: list[ScoreRow],
    runs_jsonl_path: Path,
) -> str:
    lines: list[str] = ["# Pilot Run Summary\n"]

    # Token totals per agent
    token_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "reasoning": 0})
    for rec in iter_records(runs_jsonl_path):
        t = token_totals[rec.agent_id]
        t["prompt"] += rec.prompt_tokens
        t["completion"] += rec.completion_tokens
        t["reasoning"] += rec.reasoning_tokens

    lines.append("## Token Usage per Agent\n")
    lines.append("| Agent | Prompt Tokens | Completion Tokens | Reasoning Tokens |")
    lines.append("|-------|--------------|-------------------|-----------------|")
    for agent_id, t in sorted(token_totals.items()):
        lines.append(f"| {agent_id} | {t['prompt']:,} | {t['completion']:,} | {t['reasoning']:,} |")

    # Per-agent, per-subdomain mean polytomous score
    lines.append("\n## Mean Polytomous Score by Agent × Subdomain\n")
    by_agent_subdomain: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in scores:
        by_agent_subdomain[(row.agent_id, row.subdomain)].append(row.y)

    agents = sorted({r.agent_id for r in scores})
    subdomains = sorted({r.subdomain for r in scores})

    header = "| Agent | " + " | ".join(subdomains) + " |"
    sep = "|-------|" + "|".join(["---"] * len(subdomains)) + "|"
    lines.append(header)
    lines.append(sep)
    for agent in agents:
        cells = []
        for sd in subdomains:
            vals = by_agent_subdomain.get((agent, sd), [])
            cells.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        lines.append(f"| {agent} | " + " | ".join(cells) + " |")

    # Response Integrity per agent (NEW)
    lines.append("\n## Response Integrity by Agent\n")
    lines.append("Critical for detecting silent-failure regressions (e.g., empty responses).\n")
    lines.append("| Agent | Total Runs | Success | Fail | Partial | Null | Empty Response | Empty Rate |")
    lines.append("|-------|-----------:|--------:|-----:|--------:|-----:|---------------:|-----------:|")
    integ: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "success": 0, "fail": 0, "partial": 0, "null": 0, "empty": 0,
    })
    for rec in iter_records(runs_jsonl_path):
        b = integ[rec.agent_id]
        b["total"] += 1
        if rec.success is None:
            b["null"] += 1
        elif rec.success is True:
            b["success"] += 1
        else:
            b["fail"] += 1
        if rec.verdict == "PARTIAL":
            b["partial"] += 1
        if _is_empty_response(rec):
            b["empty"] += 1
    for agent_id, b in sorted(integ.items()):
        rate = (b["empty"] / b["total"]) if b["total"] else 0.0
        lines.append(
            f"| {agent_id} | {b['total']} | {b['success']} | {b['fail']} | {b['partial']} | "
            f"{b['null']} | {b['empty']} | {rate:.2%} |"
        )

    # Grader Type Distribution (NEW) — sanity check
    lines.append("\n## Grader Type Distribution\n")
    gtypes: dict[str, int] = defaultdict(int)
    seen_items: set[str] = set()
    for row in scores:
        if row.item_id in seen_items:
            continue
        seen_items.add(row.item_id)
        gtypes[row.grader_type or "(unset)"] += 1
    lines.append("| Grader Type | Items |")
    lines.append("|-------------|------:|")
    for gt, n in sorted(gtypes.items()):
        lines.append(f"| {gt} | {n} |")

    return "\n".join(lines) + "\n"


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_full_pipeline(
    runs_jsonl_path: Path,
    items_meta_yaml_path: Path,
    output_dir: Path,
    tau1: float = 0.20,
    tau2: float = 0.75,
    partial_credit: float = 0.5,
    regrade: bool = True,
) -> dict[str, Path]:
    """Full pipeline: meta → regrade → score → profile → diagnostics → rank → summary."""
    from .diagnostics import (
        build_diagnostics_summary, compute_item_diagnostics, write_diagnostics,
    )
    from .profile import compute_profiles, write_profiles
    from .ranking import compute_rankings, write_rankings

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    meta = load_items_meta(items_meta_yaml_path)
    items_grader_type = {iid: m["grader_type"] for iid, m in meta.items()}
    items_meta_triple = {
        iid: (m["domain"], m["subdomain"], m["declared_tier"])
        for iid, m in meta.items()
    }

    active_runs = runs_jsonl_path
    if regrade:
        regraded = output_dir / "runs_regraded.jsonl"
        regrade_runs(runs_jsonl_path, items_grader_type, regraded)
        active_runs = regraded
        paths["regraded_jsonl"] = regraded

    scores = compute_scores(
        active_runs,
        items_meta=items_meta_triple,
        tau1=tau1,
        tau2=tau2,
        items_grader_type=items_grader_type,
        partial_credit=partial_credit,
    )
    scores_jsonl, scores_csv = write_scores(scores, output_dir)
    paths["scores_jsonl"] = scores_jsonl
    paths["scores_csv"] = scores_csv

    profiles = compute_profiles(scores)
    prof_csv, prof_json, prof_radar = write_profiles(profiles, output_dir)
    paths["profiles_csv"] = prof_csv
    paths["profiles_json"] = prof_json
    paths["profiles_radar_json"] = prof_radar

    diagnostics = compute_item_diagnostics(scores)
    diag_all, diag_prob = write_diagnostics(diagnostics, output_dir)
    paths["item_diagnostics_csv"] = diag_all
    paths["item_diagnostics_problematic_csv"] = diag_prob

    rankings = compute_rankings(profiles)
    rank_csv, rank_json = write_rankings(rankings, output_dir)
    paths["rankings_csv"] = rank_csv
    paths["rankings_json"] = rank_json

    summary = (
        build_summary(scores, active_runs)
        + "\n"
        + build_diagnostics_summary(diagnostics)
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary)
    paths["summary_md"] = summary_path

    return paths


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

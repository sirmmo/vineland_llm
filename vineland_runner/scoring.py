"""Polytomous 0/1/2 scoring from JSONL run log."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .storage import iter_records
from .types import ScoreRow


def compute_scores(
    runs_jsonl_path: Path,
    items_meta: dict[str, tuple[str, str, int]] | None = None,
    tau1: float = 0.20,
    tau2: float = 0.75,
    null_as_zero: bool = False,
) -> list[ScoreRow]:
    """
    Group runs by (agent_id, item_id), compute success rate, return polytomous scores.

    items_meta: optional {item_id: (domain, subdomain, tier)} for richer output.
    null_as_zero: if True, treat runs with success=None as failures instead of skipping.
    """
    from .types import RunRecord
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)

    for rec in iter_records(runs_jsonl_path):
        groups[(rec.agent_id, rec.item_id)].append(rec)

    scores: list[ScoreRow] = []
    for (agent_id, item_id), recs in groups.items():
        if null_as_zero:
            evaluated = [bool(r.success) for r in recs]
        else:
            evaluated = [bool(r.success) for r in recs if r.success is not None]

        n_reps = len(evaluated)
        if n_reps == 0:
            continue

        n_successes = sum(evaluated)
        s = n_successes / n_reps

        if s < tau1:
            y = 0
        elif s < tau2:
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
            n_reps=n_reps,
            n_successes=n_successes,
            s=round(s, 4),
            y=y,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_reasoning_tokens=total_reasoning,
            total_latency_s=round(total_latency, 3),
            mean_latency_s=round(total_latency / len(recs), 3),
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


def build_summary(
    scores: list[ScoreRow],
    runs_jsonl_path: Path,
) -> str:
    from collections import defaultdict

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

    # Per-agent, per-subdomain mean score
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

    return "\n".join(lines) + "\n"

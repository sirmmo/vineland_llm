"""Statistics extractor for a single runs.jsonl file."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .storage import iter_records
from .types import RunRecord


# ── helpers ───────────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(n: float, decimals: int = 2) -> str:
    return f"{n:.{decimals}f}"


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "—"


# ── core computation ───────────────────────────────────────────────────────────

def compute_stats(runs_jsonl_path: Path) -> dict[str, Any]:
    all_records: list[RunRecord] = list(iter_records(runs_jsonl_path))
    if not all_records:
        return {"error": "No records found", "path": str(runs_jsonl_path)}

    total = len(all_records)
    n_success = sum(1 for r in all_records if r.success is True)
    n_failure = sum(1 for r in all_records if r.success is False)
    n_error   = sum(1 for r in all_records if r.success is None)

    # ── per-agent ──────────────────────────────────────────────────────────────
    agent_recs: dict[str, list[RunRecord]] = defaultdict(list)
    for r in all_records:
        agent_recs[r.agent_id].append(r)

    per_agent: list[dict[str, Any]] = []
    for agent_id, recs in sorted(agent_recs.items()):
        latencies = sorted(r.latency_s for r in recs if r.latency_s > 0)
        ok   = [r for r in recs if r.success is True]
        fail = [r for r in recs if r.success is False]
        err  = [r for r in recs if r.success is None]
        per_agent.append({
            "agent_id": agent_id,
            "n_runs": len(recs),
            "n_success": len(ok),
            "n_failure": len(fail),
            "n_error": len(err),
            "success_rate": len(ok) / (len(ok) + len(fail)) if (ok or fail) else None,
            "total_prompt_tokens": sum(r.prompt_tokens for r in recs),
            "total_completion_tokens": sum(r.completion_tokens for r in recs),
            "total_reasoning_tokens": sum(r.reasoning_tokens for r in recs),
            "mean_latency_s": _mean(latencies),
            "p50_latency_s": _percentile(latencies, 50),
            "p95_latency_s": _percentile(latencies, 95),
            "max_latency_s": max(latencies) if latencies else 0.0,
        })

    # ── per-item ───────────────────────────────────────────────────────────────
    item_recs: dict[str, list[RunRecord]] = defaultdict(list)
    for r in all_records:
        item_recs[r.item_id].append(r)

    per_item: list[dict[str, Any]] = []
    for item_id, recs in sorted(item_recs.items()):
        ok   = [r for r in recs if r.success is True]
        fail = [r for r in recs if r.success is False]
        err  = [r for r in recs if r.success is None]
        # per-agent success rates for this item (to compute spread)
        by_agent: dict[str, list[bool]] = defaultdict(list)
        for r in recs:
            if r.success is not None:
                by_agent[r.agent_id].append(r.success)
        agent_rates = [sum(v) / len(v) for v in by_agent.values() if v]
        per_item.append({
            "item_id": item_id,
            "n_runs": len(recs),
            "n_agents": len(by_agent),
            "n_success": len(ok),
            "n_error": len(err),
            "overall_success_rate": len(ok) / (len(ok) + len(fail)) if (ok or fail) else None,
            "min_agent_rate": min(agent_rates) if agent_rates else None,
            "max_agent_rate": max(agent_rates) if agent_rates else None,
            "std_agent_rate": (
                math.sqrt(_mean([(x - _mean(agent_rates)) ** 2 for x in agent_rates]))
                if len(agent_rates) > 1 else 0.0
            ),
        })

    # ── overall token totals ───────────────────────────────────────────────────
    return {
        "path": str(runs_jsonl_path),
        "overall": {
            "total_runs": total,
            "n_success": n_success,
            "n_failure": n_failure,
            "n_error": n_error,
            "success_rate": n_success / (n_success + n_failure) if (n_success + n_failure) else None,
            "error_rate": n_error / total,
            "total_prompt_tokens": sum(r.prompt_tokens for r in all_records),
            "total_completion_tokens": sum(r.completion_tokens for r in all_records),
            "total_reasoning_tokens": sum(r.reasoning_tokens for r in all_records),
            "total_latency_s": sum(r.latency_s for r in all_records),
            "mean_latency_s": _mean([r.latency_s for r in all_records if r.latency_s > 0]),
        },
        "per_agent": per_agent,
        "per_item": per_item,
    }


# ── text rendering ─────────────────────────────────────────────────────────────

def render_stats(stats: dict[str, Any]) -> str:
    if "error" in stats:
        return f"Error: {stats['error']}\n"

    lines: list[str] = []
    ov = stats["overall"]

    lines += [
        f"runs.jsonl: {stats['path']}",
        "",
        "── Overall ─────────────────────────────────────────────────────────────────",
        f"  Total runs      : {ov['total_runs']}",
        f"  Success         : {ov['n_success']}  ({_pct(ov['n_success'], ov['total_runs'])})",
        f"  Failure         : {ov['n_failure']}  ({_pct(ov['n_failure'], ov['total_runs'])})",
        f"  Error (null)    : {ov['n_error']}  ({_pct(ov['n_error'], ov['total_runs'])})",
        f"  Success rate    : {_fmt(ov['success_rate'] * 100) + '%' if ov['success_rate'] is not None else '—'}",
        f"  Total latency   : {_fmt(ov['total_latency_s'])}s  (mean {_fmt(ov['mean_latency_s'])}s/run)",
        f"  Prompt tokens   : {ov['total_prompt_tokens']:,}",
        f"  Completion tok. : {ov['total_completion_tokens']:,}",
        f"  Reasoning tok.  : {ov['total_reasoning_tokens']:,}",
        "",
    ]

    # ── per-agent table ────────────────────────────────────────────────────────
    lines.append("── Per-agent ───────────────────────────────────────────────────────────────")
    col = "{:<32} {:>6} {:>7} {:>7} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8}"
    lines.append(col.format(
        "Agent", "Runs", "Succ", "Fail", "Error",
        "Succ%", "P.Tok", "Mean.lat", "P50.lat", "P95.lat"
    ))
    lines.append("─" * 110)
    for a in stats["per_agent"]:
        sr = f"{a['success_rate']*100:.1f}%" if a["success_rate"] is not None else "—"
        lines.append(col.format(
            a["agent_id"][:32],
            a["n_runs"], a["n_success"], a["n_failure"], a["n_error"],
            sr,
            f"{a['total_prompt_tokens']:,}",
            f"{a['mean_latency_s']:.2f}s",
            f"{a['p50_latency_s']:.2f}s",
            f"{a['p95_latency_s']:.2f}s",
        ))
    lines.append("")

    # ── per-item table ─────────────────────────────────────────────────────────
    lines.append("── Per-item ────────────────────────────────────────────────────────────────")
    col2 = "{:<16} {:>6} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8}"
    lines.append(col2.format(
        "Item", "Runs", "Agents", "Succ", "Errors",
        "Overall%", "Min%", "Max%"
    ))
    lines.append("─" * 75)
    for it in stats["per_item"]:
        sr   = f"{it['overall_success_rate']*100:.1f}%" if it["overall_success_rate"] is not None else "—"
        mn   = f"{it['min_agent_rate']*100:.1f}%" if it["min_agent_rate"] is not None else "—"
        mx   = f"{it['max_agent_rate']*100:.1f}%" if it["max_agent_rate"] is not None else "—"
        lines.append(col2.format(
            it["item_id"][:16],
            it["n_runs"], it["n_agents"], it["n_success"], it["n_error"],
            sr, mn, mx,
        ))
    lines.append("")

    return "\n".join(lines)

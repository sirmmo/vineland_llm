"""Compute leaderboard from a flat list of ScoreRow dicts."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import DomainScore, LeaderboardEntry, RankingResponse


def compute_ranking(
    score_rows: list[dict[str, Any]],
    agent_meta: list[dict[str, Any]],
) -> RankingResponse:
    meta_by_id = {a["id"]: a for a in agent_meta}

    # group by agent
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in score_rows:
        by_agent[row["agent_id"]].append(row)

    # total unique items across all agents
    all_item_ids = {r["item_id"] for r in score_rows}

    entries: list[LeaderboardEntry] = []
    for agent_id, rows in by_agent.items():
        mean_y = sum(r["y"] for r in rows) / len(rows)
        mean_sr = sum(r["s"] for r in rows) / len(rows)

        # per-domain breakdown
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get("domain"):
                by_domain[r["domain"]].append(r)

        domain_scores = [
            DomainScore(
                domain=domain,
                mean_y=round(sum(r["y"] for r in d_rows) / len(d_rows), 4),
                n_items=len(d_rows),
            )
            for domain, d_rows in sorted(by_domain.items())
        ]

        meta = meta_by_id.get(agent_id, {})
        entries.append(LeaderboardEntry(
            rank=0,  # assigned after sort
            agent_id=agent_id,
            display_name=meta.get("display_name", agent_id),
            mean_y=round(mean_y, 4),
            mean_success_rate=round(mean_sr, 4),
            n_items_evaluated=len(rows),
            domain_scores=domain_scores,
            notes=meta.get("notes", ""),
        ))

    # sort by mean_y desc, break ties by success rate
    entries.sort(key=lambda e: (-e.mean_y, -e.mean_success_rate))
    for i, e in enumerate(entries, 1):
        e.rank = i

    return RankingResponse(
        leaderboard=entries,
        n_items_total=len(all_item_ids),
    )

"""Per-dimension agent leaderboards built from AgentProfile data."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .profile import AgentProfile


# Ordered list of all ranked dimensions.
_DIMENSIONS: list[tuple[str, str]] = [
    ("theta_CM", "Communication"),
    ("theta_DLS", "Daily Living"),
    ("theta_SOC", "Socialization"),
    ("theta_AF_overall", "Agentic (overall)"),
    ("theta_AF_WMM", "World-Model Maint."),
    ("theta_AF_TOL", "Tool Use"),
    ("theta_AF_CTX", "Context Mgmt"),
    ("theta_AF_AFD", "Affordance Disc."),
    ("theta_AF_PLN", "Planning"),
    ("theta_AF_REC", "Recovery"),
    ("theta_AF_MET", "Metacognition"),
]


@dataclass
class DimensionRanking:
    dimension: str
    label: str
    ranked_agents: list[dict[str, Any]] = field(default_factory=list)
    n_agents: int = 0


def _n_items_attr(dim_key: str) -> Optional[str]:
    """Return the AgentProfile attribute holding n_items for this dim (or None for AF_overall)."""
    if dim_key == "theta_AF_overall":
        return None
    return f"n_items_{dim_key[len('theta_'):]}"


def _se_attr(dim_key: str) -> Optional[str]:
    if dim_key == "theta_AF_overall":
        return None
    return f"se_{dim_key}"


def compute_rankings(
    profiles: list[AgentProfile],
    min_n_items: int = 3,
) -> list[DimensionRanking]:
    rankings: list[DimensionRanking] = []
    for dim_key, label in _DIMENSIONS:
        items: list[dict[str, Any]] = []
        n_attr = _n_items_attr(dim_key)
        se_attr = _se_attr(dim_key)

        for p in profiles:
            theta = getattr(p, dim_key, None)
            if theta is None:
                continue
            n_items = getattr(p, n_attr) if n_attr else None
            # AF_overall has no explicit n_items — skip the min check there.
            if n_items is not None and n_items < min_n_items:
                continue
            se = getattr(p, se_attr) if se_attr else None
            items.append({
                "agent_id": p.agent_id,
                "theta": theta,
                "se": se,
                "n_items": n_items if n_items is not None else 0,
            })

        items.sort(key=lambda x: (-x["theta"], x["se"] if x["se"] is not None else 1.0))
        for i, row in enumerate(items, 1):
            row["rank"] = i

        ordered = [
            {"rank": r["rank"], "agent_id": r["agent_id"], "theta": r["theta"],
             "se": r["se"], "n_items": r["n_items"]}
            for r in items
        ]

        rankings.append(DimensionRanking(
            dimension=dim_key,
            label=label,
            ranked_agents=ordered,
            n_agents=len(ordered),
        ))

    return rankings


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_rankings(
    rankings: list[DimensionRanking],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rankings.csv"
    json_path = output_dir / "rankings.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "rank", "agent_id", "theta", "se", "n_items"])
        for r in rankings:
            for row in r.ranked_agents:
                writer.writerow([r.dimension, row["rank"], row["agent_id"],
                                 row["theta"], row["se"], row["n_items"]])

    payload = {
        "version": "1.0",
        "generated_at": _iso_now(),
        "rankings": [
            {
                "dimension": r.dimension,
                "label": r.label,
                "n_agents": r.n_agents,
                "agents": r.ranked_agents,
            }
            for r in rankings
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path

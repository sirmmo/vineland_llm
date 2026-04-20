"""Agent multidimensional profiles for the Vineland-Extended paper.

Theta proxy = mean polytomous y across items in a (sub)domain, in [0, 2].
Standard error = std(y) / sqrt(n_items).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import stdev
from typing import Optional

from .types import ScoreRow


# The three non-AF domains live at the main level.
_MAIN_DOMAINS = ("CM", "DLS", "SOC")
# The seven AF subdomains — must match spec section 2.3 ordering.
_AF_SUBDOMAINS = ("WMM", "TOL", "CTX", "AFD", "PLN", "REC", "MET")


@dataclass
class AgentProfile:
    agent_id: str
    # Non-AF domain thetas
    theta_CM: Optional[float] = None
    theta_DLS: Optional[float] = None
    theta_SOC: Optional[float] = None
    # AF subdomain thetas
    theta_AF_WMM: Optional[float] = None
    theta_AF_TOL: Optional[float] = None
    theta_AF_CTX: Optional[float] = None
    theta_AF_AFD: Optional[float] = None
    theta_AF_PLN: Optional[float] = None
    theta_AF_REC: Optional[float] = None
    theta_AF_MET: Optional[float] = None
    # Aggregated AF
    theta_AF_overall: Optional[float] = None
    # Counts
    n_items_CM: int = 0
    n_items_DLS: int = 0
    n_items_SOC: int = 0
    n_items_AF_WMM: int = 0
    n_items_AF_TOL: int = 0
    n_items_AF_CTX: int = 0
    n_items_AF_AFD: int = 0
    n_items_AF_PLN: int = 0
    n_items_AF_REC: int = 0
    n_items_AF_MET: int = 0
    # Standard errors
    se_theta_CM: Optional[float] = None
    se_theta_DLS: Optional[float] = None
    se_theta_SOC: Optional[float] = None
    se_theta_AF_WMM: Optional[float] = None
    se_theta_AF_TOL: Optional[float] = None
    se_theta_AF_CTX: Optional[float] = None
    se_theta_AF_AFD: Optional[float] = None
    se_theta_AF_PLN: Optional[float] = None
    se_theta_AF_REC: Optional[float] = None
    se_theta_AF_MET: Optional[float] = None
    # Covariates
    mean_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0


def _theta_and_se(ys: list[int]) -> tuple[Optional[float], Optional[float]]:
    if not ys:
        return None, None
    theta = sum(ys) / len(ys)
    se = (stdev(ys) / sqrt(len(ys))) if len(ys) > 1 else None
    return round(theta, 4), (round(se, 4) if se is not None else None)


def compute_profiles(
    scores: list[ScoreRow],
    min_items_per_dim: int = 1,
) -> list[AgentProfile]:
    """Aggregate ScoreRow list into per-agent profiles."""
    # Group y values by (agent, dim_key)
    by_agent_dim: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    covariates: dict[str, dict[str, float]] = defaultdict(
        lambda: {"latency_sum": 0.0, "n": 0, "prompt": 0, "completion": 0, "reasoning": 0}
    )

    for row in scores:
        if row.domain in _MAIN_DOMAINS:
            key = f"theta_{row.domain}"
            by_agent_dim[row.agent_id][key].append(row.y)
        elif row.domain == "AF":
            sub = row.subdomain.upper()
            if sub in _AF_SUBDOMAINS:
                key = f"theta_AF_{sub}"
                by_agent_dim[row.agent_id][key].append(row.y)
            # Every AF item also contributes to theta_AF_overall
            by_agent_dim[row.agent_id]["theta_AF_overall"].append(row.y)

        c = covariates[row.agent_id]
        c["latency_sum"] += row.total_latency_s
        c["n"] += row.n_reps
        c["prompt"] += row.total_prompt_tokens
        c["completion"] += row.total_completion_tokens
        c["reasoning"] += row.total_reasoning_tokens

    profiles: list[AgentProfile] = []
    all_keys = (
        [f"theta_{d}" for d in _MAIN_DOMAINS]
        + [f"theta_AF_{s}" for s in _AF_SUBDOMAINS]
        + ["theta_AF_overall"]
    )

    for agent_id in sorted(by_agent_dim.keys()):
        p = AgentProfile(agent_id=agent_id)
        for key in all_keys:
            ys = by_agent_dim[agent_id].get(key, [])
            n = len(ys)
            theta, se = _theta_and_se(ys) if n >= min_items_per_dim else (None, None)

            if key == "theta_AF_overall":
                p.theta_AF_overall = theta
                # Spec omits n_items/se fields for AF_overall.
                continue

            setattr(p, key, theta)
            setattr(p, f"n_items_{key[len('theta_'):]}", n)
            setattr(p, f"se_{key}", se)

        c = covariates[agent_id]
        p.mean_latency_s = round(c["latency_sum"] / c["n"], 3) if c["n"] else 0.0
        p.total_prompt_tokens = int(c["prompt"])
        p.total_completion_tokens = int(c["completion"])
        p.total_reasoning_tokens = int(c["reasoning"])
        profiles.append(p)

    return profiles


# ── serialization ─────────────────────────────────────────────────────────────

_MAIN_DIMS_META = [
    {"key": "theta_CM", "label": "Communication", "domain": "CM"},
    {"key": "theta_DLS", "label": "Daily Living", "domain": "DLS"},
    {"key": "theta_SOC", "label": "Socialization", "domain": "SOC"},
    {"key": "theta_AF_overall", "label": "Agentic", "domain": "AF"},
]
_AF_DIMS_META = [
    {"key": "theta_AF_WMM", "label": "World-Model Maint.", "subdomain": "WMM"},
    {"key": "theta_AF_TOL", "label": "Tool Use", "subdomain": "TOL"},
    {"key": "theta_AF_CTX", "label": "Context Mgmt", "subdomain": "CTX"},
    {"key": "theta_AF_AFD", "label": "Affordance Disc.", "subdomain": "AFD"},
    {"key": "theta_AF_PLN", "label": "Planning", "subdomain": "PLN"},
    {"key": "theta_AF_REC", "label": "Recovery", "subdomain": "REC"},
    {"key": "theta_AF_MET", "label": "Metacognition", "subdomain": "MET"},
]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_profiles(
    profiles: list[AgentProfile],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "profiles.csv"
    json_path = output_dir / "profiles.json"
    radar_path = output_dir / "profiles_radar.json"

    # Flat CSV
    if profiles:
        fields_list = list(asdict(profiles[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_list)
            writer.writeheader()
            for p in profiles:
                writer.writerow(asdict(p))
    else:
        csv_path.write_text("")

    # Nested JSON
    with open(json_path, "w") as f:
        json.dump([asdict(p) for p in profiles], f, indent=2)

    # Radar JSON (web-API-ready)
    radar = {
        "version": "1.0",
        "generated_at": _iso_now(),
        "dimensions": {
            "main": _MAIN_DIMS_META,
            "af_subdomains": _AF_DIMS_META,
        },
        "scale": {"min": 0.0, "max": 2.0, "note": "mean polytomous y"},
        "agents": [],
    }
    for p in profiles:
        main_block = {d["key"]: getattr(p, d["key"]) for d in _MAIN_DIMS_META}
        af_block = {d["key"]: getattr(p, d["key"]) for d in _AF_DIMS_META}
        se_block = {f"se_{k}": getattr(p, f"se_{k}", None) for k in [d["key"] for d in _AF_DIMS_META]}
        se_block.update({
            "se_theta_CM": p.se_theta_CM,
            "se_theta_DLS": p.se_theta_DLS,
            "se_theta_SOC": p.se_theta_SOC,
        })
        n_items_block = {
            f"n_items_{k[len('theta_'):]}": getattr(p, f"n_items_{k[len('theta_'):]}", 0)
            for k in [d["key"] for d in _AF_DIMS_META] + ["theta_CM", "theta_DLS", "theta_SOC"]
        }
        radar["agents"].append({
            "agent_id": p.agent_id,
            "main": main_block,
            "af_subdomains": af_block,
            "se": se_block,
            "n_items": n_items_block,
            "covariates": {
                "mean_latency_s": p.mean_latency_s,
                "total_prompt_tokens": p.total_prompt_tokens,
                "total_completion_tokens": p.total_completion_tokens,
                "total_reasoning_tokens": p.total_reasoning_tokens,
            },
        })
    with open(radar_path, "w") as f:
        json.dump(radar, f, indent=2)

    return csv_path, json_path, radar_path

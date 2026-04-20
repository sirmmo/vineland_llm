"""Item-level quality flags: ceiling / floor / zero-variance / bimodal.

Flags items that are problematic for IRT calibration or indicate rubric bugs.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Optional

from .types import ScoreRow


@dataclass
class ItemDiagnostic:
    item_id: str
    domain: str
    subdomain: str
    declared_tier: int
    n_agents: int
    mean_s: float
    mean_s_partial: float
    var_s: float
    min_s: float
    max_s: float
    range_s: float
    mean_y: float
    observed_tier: Optional[int]
    is_ceiling: bool
    is_floor: bool
    is_zero_variance: bool
    is_bimodal: bool
    is_problematic: bool
    flag_reasons: list[str] = field(default_factory=list)


def _infer_observed_tier(mean_s: float) -> int:
    """Spec §3.2 buckets: 1=trivial → 6=floor."""
    if mean_s >= 0.95:
        return 1
    if mean_s >= 0.75:
        return 2
    if mean_s >= 0.50:
        return 3
    if mean_s >= 0.25:
        return 4
    if mean_s >= 0.05:
        return 5
    return 6


def _check_bimodal(
    s_values: list[float],
    bimodality_threshold: float,
) -> bool:
    """Simple heuristic (NOT a full bimodality test).

    Fraction of s in extremes [0, 0.2] ∪ [0.8, 1.0] vs middle (0.2, 0.8).
    If extreme_fraction > threshold AND middle_fraction < (1 - threshold),
    flag as bimodal. Also requires at least one observation at each extreme
    (otherwise a floor or ceiling item would incorrectly trigger).
    """
    if not s_values:
        return False
    n = len(s_values)
    low = sum(1 for s in s_values if s <= 0.2)
    high = sum(1 for s in s_values if s >= 0.8)
    middle = n - low - high
    extreme = low + high
    return (
        low > 0
        and high > 0
        and (extreme / n) > bimodality_threshold
        and (middle / n) < (1 - bimodality_threshold)
    )


def compute_item_diagnostics(
    scores: list[ScoreRow],
    ceiling_threshold: float = 0.95,
    floor_threshold: float = 0.05,
    zero_var_threshold: float = 0.01,
    bimodality_threshold: float = 0.60,
) -> list[ItemDiagnostic]:
    by_item: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in scores:
        by_item[row.item_id].append(row)

    diagnostics: list[ItemDiagnostic] = []
    for item_id, rows in by_item.items():
        s_values = [r.s for r in rows]
        s_partial_values = [r.s_partial_weighted for r in rows]
        y_values = [r.y for r in rows]
        n = len(rows)

        mean_s = sum(s_values) / n
        mean_s_partial = sum(s_partial_values) / n
        var_s = pvariance(s_values) if n > 1 else 0.0
        min_s = min(s_values)
        max_s = max(s_values)
        mean_y = sum(y_values) / n

        is_ceiling = mean_s > ceiling_threshold
        is_floor = mean_s < floor_threshold
        is_zero_variance = var_s < zero_var_threshold
        is_bimodal = _check_bimodal(s_values, bimodality_threshold)

        reasons: list[str] = []
        if is_ceiling:
            reasons.append(f"ceiling (mean_s={mean_s:.3f} > {ceiling_threshold})")
        if is_floor:
            reasons.append(f"floor (mean_s={mean_s:.3f} < {floor_threshold})")
        if is_zero_variance:
            reasons.append(f"zero-variance (var_s={var_s:.4f} < {zero_var_threshold})")
        if is_bimodal:
            reasons.append("bimodal distribution across agents")

        is_problematic = any([is_ceiling, is_floor, is_zero_variance, is_bimodal])

        first = rows[0]
        diagnostics.append(ItemDiagnostic(
            item_id=item_id,
            domain=first.domain,
            subdomain=first.subdomain,
            declared_tier=first.tier,
            n_agents=n,
            mean_s=round(mean_s, 4),
            mean_s_partial=round(mean_s_partial, 4),
            var_s=round(var_s, 4),
            min_s=round(min_s, 4),
            max_s=round(max_s, 4),
            range_s=round(max_s - min_s, 4),
            mean_y=round(mean_y, 4),
            observed_tier=_infer_observed_tier(mean_s),
            is_ceiling=is_ceiling,
            is_floor=is_floor,
            is_zero_variance=is_zero_variance,
            is_bimodal=is_bimodal,
            is_problematic=is_problematic,
            flag_reasons=reasons,
        ))

    diagnostics.sort(key=lambda d: (d.domain, d.subdomain, d.item_id))
    return diagnostics


def write_diagnostics(
    diagnostics: list[ItemDiagnostic],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "item_diagnostics.csv"
    prob_csv = output_dir / "item_diagnostics_problematic.csv"

    def _write(path: Path, rows: list[ItemDiagnostic]) -> None:
        if not rows:
            path.write_text("")
            return
        fields_list = list(asdict(rows[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_list)
            writer.writeheader()
            for r in rows:
                d = asdict(r)
                # Join list field for CSV readability
                d["flag_reasons"] = "; ".join(d.get("flag_reasons") or [])
                writer.writerow(d)

    _write(all_csv, diagnostics)
    _write(prob_csv, [d for d in diagnostics if d.is_problematic])
    return all_csv, prob_csv


def build_diagnostics_summary(diagnostics: list[ItemDiagnostic]) -> str:
    lines: list[str] = ["## Item Diagnostics\n"]
    total = len(diagnostics)
    lines.append(f"Total items analyzed: **{total}**\n")

    counts = {
        "ceiling": sum(1 for d in diagnostics if d.is_ceiling),
        "floor": sum(1 for d in diagnostics if d.is_floor),
        "zero_variance": sum(1 for d in diagnostics if d.is_zero_variance),
        "bimodal": sum(1 for d in diagnostics if d.is_bimodal),
        "problematic (any)": sum(1 for d in diagnostics if d.is_problematic),
    }
    lines.append("| Flag | Count |")
    lines.append("|------|------:|")
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    # Tier coherence
    lines.append("\n### Tier Coherence (declared vs observed)\n")
    coh: dict[tuple[int, Optional[int]], int] = defaultdict(int)
    for d in diagnostics:
        coh[(d.declared_tier, d.observed_tier)] += 1
    lines.append("| Declared Tier | Observed Tier | Count |")
    lines.append("|--------------:|--------------:|------:|")
    for (decl, obs), n in sorted(coh.items()):
        lines.append(f"| {decl} | {obs if obs is not None else '-'} | {n} |")

    # Problematic items list
    probs = [d for d in diagnostics if d.is_problematic]
    if probs:
        lines.append("\n### Problematic Items\n")
        lines.append("| Item | Domain | Subdomain | Declared Tier | Mean s | Reasons |")
        lines.append("|------|--------|-----------|--------------:|-------:|---------|")
        for d in probs:
            lines.append(
                f"| {d.item_id} | {d.domain} | {d.subdomain} | {d.declared_tier} | "
                f"{d.mean_s:.3f} | {'; '.join(d.flag_reasons)} |"
            )

    return "\n".join(lines) + "\n"

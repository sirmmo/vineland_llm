"""CLI entry point: vineland-runner <command> [args]."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    from .config import load_agents, load_pilot_config
    from .items import load_items, select_items
    from .runner import run_pilot

    config_path = Path(args.config)
    config = load_pilot_config(config_path)

    agents_path = Path(args.agents) if args.agents else config_path.parent / "agents.yaml"
    agents = load_agents(agents_path)

    items_path = Path(args.items) if args.items else Path("items")
    schema_path = Path(args.schema) if args.schema else Path("items/schema.json")
    all_items = load_items(items_path, schema_path if schema_path.exists() else None)
    selected = select_items(all_items, config.items, config.items_per_subdomain)

    print(f"Pilot: {config.name}", file=sys.stderr)
    print(f"Agents: {len([a for a in config.agents if a in agents])}", file=sys.stderr)
    print(f"Items: {len(selected)}", file=sys.stderr)
    print(f"Replications: {config.n_replications}", file=sys.stderr)
    print(f"Total runs: {len(selected) * len(config.agents) * config.n_replications}", file=sys.stderr)

    # Copy config to output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "pilot.yaml"
    if not dest.exists():
        shutil.copy(config_path, dest)

    asyncio.run(run_pilot(config, selected, agents))


def cmd_score(args: argparse.Namespace) -> None:
    from .items import load_items
    from .scoring import build_summary, compute_scores, write_scores

    run_dir = Path(args.run_dir)
    runs_path = run_dir / "runs.jsonl"
    if not runs_path.exists():
        print(f"Error: {runs_path} not found", file=sys.stderr)
        sys.exit(1)

    # Try to load item metadata for richer output
    items_meta: dict[str, tuple[str, str, int]] = {}
    items_path = Path(args.items) if args.items else Path("items")
    if items_path.exists():
        try:
            items = load_items(items_path)
            items_meta = {i.id: (i.domain, i.subdomain, i.tier) for i in items}
        except Exception:
            pass

    tau1 = float(args.tau1) if args.tau1 else 0.20
    tau2 = float(args.tau2) if args.tau2 else 0.75

    scores = compute_scores(runs_path, items_meta, tau1=tau1, tau2=tau2)
    jsonl_path, csv_path = write_scores(scores, run_dir)

    print(f"Scored {len(scores)} (agent, item) pairs")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")


def cmd_summary(args: argparse.Namespace) -> None:
    from .items import load_items
    from .scoring import build_summary, compute_scores

    run_dir = Path(args.run_dir)
    runs_path = run_dir / "runs.jsonl"
    scores_path = run_dir / "scores.jsonl"

    if not runs_path.exists():
        print(f"Error: {runs_path} not found", file=sys.stderr)
        sys.exit(1)

    items_meta: dict[str, tuple[str, str, int]] = {}
    items_path = Path(args.items) if args.items else Path("items")
    if items_path.exists():
        try:
            items = load_items(items_path)
            items_meta = {i.id: (i.domain, i.subdomain, i.tier) for i in items}
        except Exception:
            pass

    if scores_path.exists():
        from .types import ScoreRow
        from .storage import iter_records
        scores = []
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        scores.append(ScoreRow.model_validate_json(line))
                    except Exception:
                        pass
    else:
        scores = compute_scores(runs_path, items_meta)

    summary = build_summary(scores, runs_path)
    print(summary)

    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary)
    print(f"Summary written to {summary_path}", file=sys.stderr)


def cmd_stats(args: argparse.Namespace) -> None:
    from .stats import compute_stats, render_stats

    runs_path = Path(args.runs_jsonl)
    if not runs_path.exists():
        print(f"Error: {runs_path} not found", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(runs_path)
    print(render_stats(stats))

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(stats, indent=2))
        print(f"JSON written to {out}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> None:
    from .items import load_items

    items_path = Path(args.items_file)
    schema_path = Path(args.schema) if args.schema else Path("items/schema.json")

    try:
        items = load_items(items_path, schema_path if schema_path.exists() else None)
        print(f"OK — {len(items)} items valid")
    except ValueError as e:
        print(f"INVALID:\n{e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vineland-runner",
        description="Vineland-adapted LLM assessment runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run (or resume) a pilot")
    p_run.add_argument("--config", required=True, help="Path to pilot.yaml")
    p_run.add_argument("--agents", default=None, help="Path to agents.yaml (default: next to pilot.yaml)")
    p_run.add_argument("--items", default=None, help="Path to items YAML file or directory")
    p_run.add_argument("--schema", default=None, help="Path to item schema.json")

    # score
    p_score = sub.add_parser("score", help="Compute polytomous scores from a run directory")
    p_score.add_argument("run_dir", help="Path to run output directory")
    p_score.add_argument("--items", default=None, help="Path to items YAML file or directory")
    p_score.add_argument("--tau1", default=None, help="Lower threshold (default 0.20)")
    p_score.add_argument("--tau2", default=None, help="Upper threshold (default 0.75)")

    # summary
    p_summary = sub.add_parser("summary", help="Print summary for a run directory")
    p_summary.add_argument("run_dir", help="Path to run output directory")
    p_summary.add_argument("--items", default=None, help="Path to items YAML file or directory")

    # stats
    p_stats = sub.add_parser("stats", help="Extract statistics from a runs.jsonl file")
    p_stats.add_argument("runs_jsonl", help="Path to runs.jsonl")
    p_stats.add_argument("--json", default=None, metavar="FILE", help="Also write stats as JSON to FILE")

    # validate
    p_validate = sub.add_parser("validate", help="Validate items YAML against schema")
    p_validate.add_argument("items_file", help="Path to items.yaml")
    p_validate.add_argument("--schema", default=None, help="Path to schema.json")

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "score": cmd_score,
        "summary": cmd_summary,
        "stats": cmd_stats,
        "validate": cmd_validate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

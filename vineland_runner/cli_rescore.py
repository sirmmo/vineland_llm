"""CLI: re-grade an existing runs.jsonl and emit full scoring outputs.

Usage:
    python -m vineland_runner.cli_rescore \
        --runs-jsonl runs.jsonl \
        --items items.yaml \
        --output-dir out/rescored/ \
        [--tau1 0.20] [--tau2 0.75] [--partial-credit 0.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vineland-runner-rescore",
        description="Post-hoc re-score an existing runs.jsonl (verdict regrade + full pipeline).",
    )
    parser.add_argument("--runs-jsonl", required=True, help="Path to input runs.jsonl")
    parser.add_argument("--items", required=True, help="Path to items.yaml with grader_type metadata")
    parser.add_argument("--output-dir", required=True, help="Directory for all output files")
    parser.add_argument("--tau1", type=float, default=0.20)
    parser.add_argument("--tau2", type=float, default=0.75)
    parser.add_argument("--partial-credit", type=float, default=0.5)
    parser.add_argument("--no-regrade", action="store_true",
                        help="Skip verdict regrade; score runs.jsonl as-is")
    args = parser.parse_args()

    from .scoring import run_full_pipeline

    runs = Path(args.runs_jsonl)
    items = Path(args.items)
    out = Path(args.output_dir)

    if not runs.exists():
        print(f"Error: {runs} not found", file=sys.stderr)
        sys.exit(1)
    if not items.exists():
        print(f"Error: {items} not found", file=sys.stderr)
        sys.exit(1)

    paths = run_full_pipeline(
        runs_jsonl_path=runs,
        items_meta_yaml_path=items,
        output_dir=out,
        tau1=args.tau1,
        tau2=args.tau2,
        partial_credit=args.partial_credit,
        regrade=not args.no_regrade,
    )

    print("Pipeline outputs:")
    for name, path in paths.items():
        print(f"  {name:<35} {path}")


if __name__ == "__main__":
    main()

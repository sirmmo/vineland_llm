"""Filter errored / null-response records out of a runs.jsonl in place.

Used to free the runner's resume mechanism to retry failed (e.g. 429) runs:
load_completed_ids() treats every record (success or error) as completed, so
errored records must be removed before re-running the pilot.

Usage:
    python scripts/strip_errors.py runs/pilot_2026_04/runs.jsonl
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"not found: {path}")

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)

    kept = dropped = 0
    out_lines: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if rec.get("error") or rec.get("response") is None:
                dropped += 1
                continue
            kept += 1
            out_lines.append(line)

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"kept={kept}  dropped={dropped}  backup={bak}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/strip_errors.py <runs.jsonl>")
    main(sys.argv[1])

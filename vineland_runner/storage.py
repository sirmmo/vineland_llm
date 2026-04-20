"""Append-only JSONL storage with run-ID deduplication."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .types import RunRecord


def load_completed_ids(jsonl_path: Path) -> set[str]:
    """Return set of run_ids already present in the JSONL file."""
    ids: set[str] = set()
    if not jsonl_path.exists():
        return ids
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "run_id" in obj:
                    ids.add(obj["run_id"])
            except json.JSONDecodeError:
                pass
    return ids


def append_record(jsonl_path: Path, record: RunRecord) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(record.model_dump_json() + "\n")


def iter_records(jsonl_path: Path) -> Iterator[RunRecord]:
    if not jsonl_path.exists():
        return
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield RunRecord.model_validate_json(line)
            except Exception:
                pass

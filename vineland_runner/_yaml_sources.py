"""Shared helper: walk a YAML file or directory and yield raw entry dicts.

Both items and agents support the same two YAML layouts:
  - multi-entry file:   {<list_key>: [ {...}, {...} ]}
  - single-entry file:  {id: ..., ...}

Directory mode recursively globs *.yaml/*.yml (sorted, hidden-skipping) and
concatenates the results. Unrelated YAMLs (no `<list_key>` and no `id`) are
silently ignored — that way spec documents can live alongside entry files
without interfering.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _yaml_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pat in ("*.yaml", "*.yml"):
        files.update(root.rglob(pat))
    return sorted(f for f in files if not f.name.startswith("."))


def _load_one(path: Path, list_key: str) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return []
    if list_key in data:
        return list(data.get(list_key) or [])
    if "id" in data:
        return [data]
    return []


def collect_entries(path: Path, list_key: str) -> list[tuple[Path, dict]]:
    """Walk `path` (file or directory) and return [(source_path, raw_dict), ...].

    Ordering is deterministic (sorted paths, file order preserved).
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    sources = [path] if path.is_file() else _yaml_files(path)
    out: list[tuple[Path, dict]] = []
    for source in sources:
        for raw in _load_one(source, list_key):
            out.append((source, raw))
    return out

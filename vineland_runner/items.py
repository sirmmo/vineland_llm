"""Item loader with JSON Schema validation.

`load_items()` accepts either a YAML file or a directory. In directory mode,
every `*.yaml` / `*.yml` under the path is scanned (recursively). Files may
contain either the multi-item format (top-level `items:` list) or a single-
item format (top-level `id:` with the item fields). Unrelated YAML files
(no `items:` and no `id:`) are silently ignored — that way spec docs can
live alongside item files without interfering.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
import jsonschema
from pydantic import ValidationError

from .types import Item


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with open(schema_path) as f:
        return json.load(f)


def _load_one_yaml(path: Path) -> list[dict]:
    """Return raw item dicts from a single YAML file.

    Supports two formats:
      - {items: [...]}   — multi-item file
      - {id: ..., ...}   — single-item file
    Anything else returns an empty list (safely skipped).
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return []
    if "items" in data:
        raw = data.get("items") or []
        return list(raw)
    if "id" in data:
        return [data]
    return []


def _yaml_files(root: Path) -> list[Path]:
    """Recursively list *.yaml / *.yml files under root, sorted, skipping hidden."""
    files: set[Path] = set()
    for pat in ("*.yaml", "*.yml"):
        files.update(root.rglob(pat))
    return sorted(f for f in files if not f.name.startswith("."))


def collect_raw_items(items_path: Path) -> list[tuple[Path, dict]]:
    """Walk a file or directory and return [(source_path, raw_item_dict), ...].

    Accepts a single file or a directory. Directory mode loads every
    *.yaml/*.yml under it (recursively). Ordering is deterministic
    (sorted paths, file order preserved).
    """
    if not items_path.exists():
        raise FileNotFoundError(f"Items path not found: {items_path}")

    sources: list[Path]
    if items_path.is_file():
        sources = [items_path]
    else:
        sources = _yaml_files(items_path)

    out: list[tuple[Path, dict]] = []
    for source in sources:
        for raw in _load_one_yaml(source):
            out.append((source, raw))
    return out


def load_items(
    items_path: Path,
    schema_path: Path | None = None,
) -> list[Item]:
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "items" / "schema.json"

    schema = _load_schema(schema_path) if schema_path.exists() else None

    items: list[Item] = []
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}

    for source, raw in collect_raw_items(items_path):
        item_id = raw.get("id", "?")
        if item_id in seen_ids and item_id != "?":
            errors.append(
                f"Duplicate item id {item_id!r}: {seen_ids[item_id]} and {source}"
            )
            continue
        if item_id != "?":
            seen_ids[item_id] = source

        if schema is not None:
            try:
                jsonschema.validate(raw, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"[{source}] Item {item_id} schema error: {e.message}")
                continue
        try:
            items.append(Item.model_validate(raw))
        except ValidationError as e:
            errors.append(f"[{source}] Item {item_id} pydantic error: {e}")

    if errors:
        raise ValueError("Item load errors:\n" + "\n".join(errors))

    return items


def select_items(
    items: list[Item],
    spec: Any,
    items_per_subdomain: int = 2,
) -> list[Item]:
    """Select items per pilot spec: 'all', 'auto', or a list of IDs."""
    if spec == "all":
        return items

    if isinstance(spec, list):
        by_id = {i.id: i for i in items}
        missing = [s for s in spec if s not in by_id]
        if missing:
            raise ValueError(f"Item IDs not found: {missing}")
        return [by_id[s] for s in spec]

    if spec == "auto":
        # Select items_per_subdomain items per subdomain, preferring middle tiers
        from collections import defaultdict
        by_subdomain: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            by_subdomain[item.subdomain].append(item)

        selected: list[Item] = []
        for subdomain_items in by_subdomain.values():
            sorted_items = sorted(subdomain_items, key=lambda i: abs(i.tier - 3))
            selected.extend(sorted_items[:items_per_subdomain])
        return selected

    raise ValueError(f"Unknown item spec: {spec!r}")

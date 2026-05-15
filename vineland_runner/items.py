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

import jsonschema
from pydantic import ValidationError

from ._yaml_sources import collect_entries
from .types import Item


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with open(schema_path) as f:
        return json.load(f)


def collect_raw_items(items_path: Path) -> list[tuple[Path, dict]]:
    """Walk a file or directory and return [(source_path, raw_item_dict), ...]."""
    return collect_entries(items_path, "items")


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

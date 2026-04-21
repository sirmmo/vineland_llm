"""Tests for item loading and schema validation."""
import textwrap
from pathlib import Path

import pytest
import yaml

from vineland_runner.items import load_items
from vineland_runner.types import Item


ITEMS_YAML = Path(__file__).parent.parent / "items" / "items.yaml"
SCHEMA_JSON = Path(__file__).parent.parent / "items" / "schema.json"


def test_load_stub_items():
    items = load_items(ITEMS_YAML, SCHEMA_JSON)
    assert len(items) >= 5
    for item in items:
        assert isinstance(item, Item)
        assert item.id
        assert item.domain in {"CM", "DLS", "SOC", "AF"}
        assert 1 <= item.tier <= 5


def test_item_rendered_prompt():
    items = load_items(ITEMS_YAML, SCHEMA_JSON)
    for item in items:
        rendered = item.rendered_prompt()
        assert rendered
        for key in item.prompt_variables:
            assert f"{{{key}}}" not in rendered


def test_rendered_prompt_tolerates_json_braces():
    item = Item(
        id="AF-TOL-9",
        domain="AF",
        subdomain="TOL",
        tier=1,
        prompt_template='Answer {question}\n\nRespond as JSON: {"tool": "<name>", "args": {}}',
        prompt_variables={"question": "What time is it?"},
        success_criterion={"type": "exact_match", "expected_answer": "clock"},
    )
    rendered = item.rendered_prompt()
    assert "What time is it?" in rendered
    assert '{"tool": "<name>", "args": {}}' in rendered


def test_item_criterion_types():
    items = load_items(ITEMS_YAML, SCHEMA_JSON)
    types = {item.success_criterion.type for item in items}
    assert "exact_match" in types
    assert "llm_judge" in types


def test_malformed_item_raises(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(textwrap.dedent("""\
        items:
          - id: "BAD"
            domain: CM
            subdomain: REC
            tier: 99
            prompt_template: "hello {x}"
            success_criterion:
              type: exact_match
              expected_answer: "foo"
    """))
    with pytest.raises(ValueError, match="Item load errors"):
        load_items(bad_yaml, SCHEMA_JSON)


def test_missing_criterion_raises(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(textwrap.dedent("""\
        items:
          - id: "CM-REC-9"
            domain: CM
            subdomain: REC
            tier: 1
            prompt_template: "hello"
            success_criterion:
              type: exact_match
    """))
    # No expected_answer or expected_regex — should fail validation
    with pytest.raises((ValueError, Exception)):
        load_items(bad_yaml, SCHEMA_JSON)


# ── directory-mode loading ────────────────────────────────────────────────────

def _single_item_dict(item_id: str) -> dict:
    return {
        "id": item_id,
        "domain": "CM",
        "subdomain": "REC",
        "tier": 1,
        "prompt_template": "please say hello to {x}",
        "prompt_variables": {"x": "world"},
        "success_criterion": {"type": "exact_match", "expected_answer": "foo"},
    }


def _write_multi(path: Path, item_ids: list[str]) -> None:
    path.write_text(yaml.safe_dump({"items": [_single_item_dict(i) for i in item_ids]}))


def _write_single(path: Path, item_id: str) -> None:
    path.write_text(yaml.safe_dump(_single_item_dict(item_id)))


def test_load_items_from_directory(tmp_path: Path):
    """Directory mode concatenates per-file items."""
    _write_multi(tmp_path / "a.yaml", ["CM-REC-1"])
    _write_multi(tmp_path / "b.yaml", ["CM-REC-2"])

    items = load_items(tmp_path, SCHEMA_JSON)
    ids = sorted(i.id for i in items)
    assert ids == ["CM-REC-1", "CM-REC-2"]


def test_load_items_single_item_format(tmp_path: Path):
    """A YAML file with a top-level `id:` is treated as a single-item file."""
    _write_single(tmp_path / "one.yaml", "CM-REC-3")
    _write_single(tmp_path / "two.yaml", "CM-REC-4")

    items = load_items(tmp_path, SCHEMA_JSON)
    ids = sorted(i.id for i in items)
    assert ids == ["CM-REC-3", "CM-REC-4"]


def test_load_items_ignores_unrelated_yaml(tmp_path: Path):
    """YAMLs with neither `items:` nor `id:` are safely ignored."""
    (tmp_path / "spec.yaml").write_text("subdomains:\n  WMM:\n    name: foo\n")
    _write_single(tmp_path / "one.yaml", "CM-REC-5")

    items = load_items(tmp_path, SCHEMA_JSON)
    assert [i.id for i in items] == ["CM-REC-5"]


def test_duplicate_ids_across_files_raise(tmp_path: Path):
    """Same id appearing in two files is a load error with source paths."""
    _write_single(tmp_path / "a.yaml", "CM-REC-6")
    _write_single(tmp_path / "b.yaml", "CM-REC-6")

    with pytest.raises(ValueError, match="Duplicate item id"):
        load_items(tmp_path, SCHEMA_JSON)


def test_load_items_declared_tier_and_passfail(tmp_path: Path):
    """Phase-2 items: declared_tier (no `tier`) + judge_passfail must load."""
    (tmp_path / "phase2.yaml").write_text(yaml.safe_dump({
        "items": [{
            "id": "AF-PLN-1",
            "domain": "AF",
            "subdomain": "PLN",
            "declared_tier": 3,
            "observed_tier": None,
            "prompt_template": "Please order these steps correctly",
            "prompt_variables": {},
            "success_criterion": {
                "type": "judge_passfail",
                "judge_prompt": "Evaluate {response}. Verdict MUST end: VERDICT: PASS / PARTIAL / FAIL",
            },
        }]
    }))
    [item] = load_items(tmp_path, SCHEMA_JSON)
    assert item.id == "AF-PLN-1"
    assert item.tier == 3
    assert item.declared_tier == 3
    assert item.observed_tier is None
    assert item.success_criterion.type == "judge_passfail"


def test_load_items_recursive_subdirs(tmp_path: Path):
    """Directory walk is recursive."""
    sub = tmp_path / "cm"
    sub.mkdir()
    _write_single(sub / "one.yaml", "CM-REC-7")
    _write_single(tmp_path / "two.yml", "CM-REC-8")

    items = load_items(tmp_path, SCHEMA_JSON)
    ids = sorted(i.id for i in items)
    assert ids == ["CM-REC-7", "CM-REC-8"]

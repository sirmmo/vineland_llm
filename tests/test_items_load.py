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

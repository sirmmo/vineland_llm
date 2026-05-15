"""Tests for load_agents() — file mode, directory mode, dedup."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vineland_runner.config import load_agents


def _agent_dict(agent_id: str) -> dict:
    return {
        "id": agent_id,
        "display_name": agent_id.replace("-", " ").title(),
        "base_url": "https://example.com/v1",
        "model_id": f"vendor/{agent_id}",
        "api_key_env": "TEST_KEY",
    }


def test_load_agents_single_file_multi_format(tmp_path: Path):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump({"agents": [_agent_dict("a1"), _agent_dict("a2")]}))

    agents = load_agents(path)
    assert sorted(agents.keys()) == ["a1", "a2"]
    assert agents["a1"].model_id == "vendor/a1"


def test_load_agents_directory_multi_files(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"agents": [_agent_dict("a1")]}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"agents": [_agent_dict("a2")]}))

    agents = load_agents(tmp_path)
    assert sorted(agents.keys()) == ["a1", "a2"]


def test_load_agents_directory_single_format(tmp_path: Path):
    """Each agent in its own file with top-level id:."""
    (tmp_path / "a1.yaml").write_text(yaml.safe_dump(_agent_dict("a1")))
    (tmp_path / "a2.yaml").write_text(yaml.safe_dump(_agent_dict("a2")))

    agents = load_agents(tmp_path)
    assert sorted(agents.keys()) == ["a1", "a2"]


def test_load_agents_recursive_subdirs(tmp_path: Path):
    sub = tmp_path / "open"
    sub.mkdir()
    (sub / "a1.yaml").write_text(yaml.safe_dump(_agent_dict("a1")))
    (tmp_path / "a2.yml").write_text(yaml.safe_dump(_agent_dict("a2")))

    agents = load_agents(tmp_path)
    assert sorted(agents.keys()) == ["a1", "a2"]


def test_load_agents_ignores_unrelated_yaml(tmp_path: Path):
    (tmp_path / "notes.yaml").write_text("description: unrelated\nversion: 1\n")
    (tmp_path / "a1.yaml").write_text(yaml.safe_dump(_agent_dict("a1")))

    agents = load_agents(tmp_path)
    assert list(agents.keys()) == ["a1"]


def test_duplicate_agent_ids_raise(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(_agent_dict("a1")))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(_agent_dict("a1")))

    with pytest.raises(ValueError, match="Duplicate agent id"):
        load_agents(tmp_path)

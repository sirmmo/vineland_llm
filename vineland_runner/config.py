"""Config loader: env vars + YAML files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ._yaml_sources import collect_entries
from .types import Agent, PilotConfig


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_agents(path: Path) -> dict[str, Agent]:
    """Load agents from a YAML file or a directory of YAML files.

    Supports the same layouts as items:
      - multi-agent file: {agents: [ {...}, {...} ]}
      - single-agent file: {id: ..., display_name: ..., ...}
    """
    agents: dict[str, Agent] = {}
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}

    for source, raw in collect_entries(path, "agents"):
        agent = Agent.model_validate(raw)
        if agent.id in seen_ids:
            errors.append(
                f"Duplicate agent id {agent.id!r}: {seen_ids[agent.id]} and {source}"
            )
            continue
        seen_ids[agent.id] = source
        agents[agent.id] = agent

    if errors:
        raise ValueError("Agent load errors:\n" + "\n".join(errors))

    return agents


def load_pilot_config(path: Path) -> PilotConfig:
    data = _load_yaml(path)
    pilot_data = data.get("pilot", data)
    return PilotConfig.model_validate(pilot_data)


def resolve_api_key(agent: Agent) -> str:
    key = os.environ.get(agent.api_key_env)
    if not key:
        raise EnvironmentError(
            f"API key env var '{agent.api_key_env}' is not set for agent '{agent.id}'. "
            f"Export it before running: export {agent.api_key_env}=<your-key>"
        )
    return key

"""Config loader: env vars + YAML files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .types import Agent, PilotConfig


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_agents(path: Path) -> dict[str, Agent]:
    data = _load_yaml(path)
    agents: dict[str, Agent] = {}
    for raw in data.get("agents", []):
        agent = Agent.model_validate(raw)
        agents[agent.id] = agent
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

"""Server configuration via environment variables."""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GitHub storage
    github_token: str = Field(description="GitHub personal access token with repo write scope")
    github_repo: str = Field(description="owner/repo where results are stored")
    github_branch: str = Field(default="main")

    # Paths to the runner's item and agent configs (mounted or relative)
    items_yaml: Path = Field(default=Path("items/items.yaml"))
    agents_yaml: Path = Field(default=Path("configs/agents.yaml"))
    pilot_yaml: Path = Field(default=Path("configs/pilot.yaml"))

    # Judge agent id (must exist in agents_yaml)
    judge_agent_id: str = Field(default="claude-opus-4-5")
    judge_api_key_env: str = Field(default="OPENROUTER_API_KEY")

    # Run behaviour
    n_replications: int = Field(default=5)
    max_concurrency: int = Field(default=2)
    runs_dir: Path = Field(default=Path("runs/api"))

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()  # type: ignore[call-arg]

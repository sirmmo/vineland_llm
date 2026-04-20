"""Server configuration via environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

# Vercel sets VERCEL=1 in all serverless function environments.
_is_vercel = bool(os.environ.get("VERCEL"))


class Settings(BaseSettings):
    # GitHub storage — empty defaults so the app loads even without env vars;
    # endpoints that need GitHub will fail with a clear HTTP error instead.
    github_token: str = Field(default="", description="GitHub personal access token with repo write scope")
    github_repo: str = Field(default="", description="owner/repo where results are stored")
    github_branch: str = Field(default="main")

    # Paths to the runner's item and agent configs (mounted or relative)
    items_yaml: Path = Field(default=Path("items/items.yaml"))
    agents_yaml: Path = Field(default=Path("configs/agents.yaml"))
    pilot_yaml: Path = Field(default=Path("configs/pilot.yaml"))

    # Judge agent id (must exist in agents_yaml)
    judge_agent_id: str = Field(default="claude-opus-4-5")
    judge_api_key_env: str = Field(default="OPENROUTER_API_KEY")

    # Run behaviour — use /tmp on Vercel since the filesystem is otherwise read-only
    n_replications: int = Field(default=5)
    max_concurrency: int = Field(default=2)
    runs_dir: Path = Field(default=Path("/tmp/runs/api") if _is_vercel else Path("runs/api"))

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()  # type: ignore[call-arg]

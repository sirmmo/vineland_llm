"""Read/write assessment data via the GitHub Contents API.

Layout in the target repo:
  data/scores/{agent_id}.jsonl   — ScoreRow lines for one agent
  data/runs/{agent_id}.jsonl     — RunRecord lines for one agent
  data/agents.json               — registered agent metadata (no keys)
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)

_BASE = "https://api.github.com"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _url(path: str) -> str:
    return f"{_BASE}/repos/{settings.github_repo}/contents/{path}"


async def _get_file(client: httpx.AsyncClient, path: str) -> tuple[str, str] | tuple[None, None]:
    """Return (decoded_content, sha) or (None, None) if file does not exist."""
    params = {"ref": settings.github_branch}
    r = await client.get(_url(path), headers=_headers(), params=params)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode()
    return content, data["sha"]


async def _put_file(
    client: httpx.AsyncClient,
    path: str,
    content: str,
    message: str,
    sha: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": settings.github_branch,
    }
    if sha:
        payload["sha"] = sha
    r = await client.put(_url(path), headers=_headers(), json=payload)
    r.raise_for_status()


# ── public API ────────────────────────────────────────────────────────────────

async def push_scores(agent_id: str, jsonl_lines: list[str]) -> None:
    path = f"data/scores/{agent_id}.jsonl"
    content = "\n".join(jsonl_lines) + "\n"
    async with httpx.AsyncClient(timeout=30) as client:
        _, sha = await _get_file(client, path)
        await _put_file(client, path, content, f"scores: {agent_id}", sha)


async def push_runs(agent_id: str, jsonl_lines: list[str]) -> None:
    path = f"data/runs/{agent_id}.jsonl"
    content = "\n".join(jsonl_lines) + "\n"
    async with httpx.AsyncClient(timeout=30) as client:
        _, sha = await _get_file(client, path)
        await _put_file(client, path, content, f"runs: {agent_id}", sha)


async def fetch_all_scores() -> list[dict[str, Any]]:
    """Return all ScoreRow dicts from every agent's scores file."""
    async with httpx.AsyncClient(timeout=30) as client:
        # list data/scores/ directory
        params = {"ref": settings.github_branch}
        r = await client.get(_url("data/scores"), headers=_headers(), params=params)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        files = [f for f in r.json() if f["name"].endswith(".jsonl")]

        rows: list[dict[str, Any]] = []
        for f in files:
            content, _ = await _get_file(client, f["path"])
            if not content:
                continue
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows


async def register_agent(meta: dict[str, Any]) -> None:
    """Append agent metadata to data/agents.json (no API key)."""
    path = "data/agents.json"
    async with httpx.AsyncClient(timeout=30) as client:
        content, sha = await _get_file(client, path)
        agents: list[dict] = json.loads(content) if content else []
        # replace if same id
        agents = [a for a in agents if a.get("id") != meta["id"]]
        agents.append(meta)
        await _put_file(
            client, path, json.dumps(agents, indent=2),
            f"register agent: {meta['id']}", sha,
        )


async def fetch_agents() -> list[dict[str, Any]]:
    path = "data/agents.json"
    async with httpx.AsyncClient(timeout=30) as client:
        content, _ = await _get_file(client, path)
        return json.loads(content) if content else []

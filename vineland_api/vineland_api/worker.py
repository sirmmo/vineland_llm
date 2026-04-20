"""Background worker: run pilot for a submitted agent, push results to GitHub."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import settings
from .models import JobStatus

log = logging.getLogger(__name__)

# In-memory job registry (survives only within one server process)
_jobs: dict[str, JobStatus] = {}


def get_job(job_id: str) -> JobStatus | None:
    return _jobs.get(job_id)


def list_jobs() -> list[JobStatus]:
    return list(_jobs.values())


async def run_agent_job(
    job_id: str,
    agent_id: str,
    base_url: str,
    model_id: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    reasoning: bool,
    display_name: str,
    notes: str,
) -> None:
    from vineland_runner.client import LLMClient
    from vineland_runner.config import load_items as _unused  # noqa
    from vineland_runner.items import load_items, select_items
    from vineland_runner.runner import run_pilot
    from vineland_runner.scoring import compute_scores, write_scores
    from vineland_runner.storage import iter_records
    from vineland_runner.types import Agent, PilotConfig

    job = _jobs[job_id]
    job.status = "running"

    try:
        # Build ephemeral agent — key lives only in this coroutine's scope
        agent = Agent(
            id=agent_id,
            display_name=display_name,
            base_url=base_url,
            model_id=model_id,
            api_key_env="__EPHEMERAL__",
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning,
            notes=notes,
        )

        # Inject key into a private env var so resolve_api_key works
        os.environ["__EPHEMERAL__"] = api_key

        # Load items
        items_path = settings.items_yaml
        if not items_path.exists():
            raise FileNotFoundError(f"Items file not found: {items_path}")
        all_items = load_items(items_path)
        items = select_items(all_items, "auto", items_per_subdomain=2)

        job.n_runs_total = len(items) * settings.n_replications

        # Judge agent setup
        judge_agent = None
        judge_key = os.environ.get(settings.judge_api_key_env)
        if judge_key:
            from vineland_runner.config import load_agents
            if settings.agents_yaml.exists():
                all_agents = load_agents(settings.agents_yaml)
                judge_agent = all_agents.get(settings.judge_agent_id)

        output_dir = settings.runs_dir / agent_id
        output_dir.mkdir(parents=True, exist_ok=True)
        runs_path = output_dir / "runs.jsonl"

        config = PilotConfig(
            name=f"api_{agent_id}",
            agents=[agent_id],
            items="all",
            n_replications=settings.n_replications,
            output_dir=str(output_dir),
            judge_agent=judge_agent.id if judge_agent else None,
            max_concurrency=settings.max_concurrency,
        )

        agents_map = {agent_id: agent}
        if judge_agent:
            agents_map[judge_agent.id] = judge_agent

        async with LLMClient() as client:  # type: ignore[attr-defined]
            await run_pilot(config, items, agents_map, client)

        # tally completed runs
        done = sum(1 for r in iter_records(runs_path) if r.success is not None)
        job.n_runs_done = done

        # Score
        items_meta = {i.id: (i.domain, i.subdomain, i.tier) for i in items}
        scores = compute_scores(runs_path, items_meta)
        _, _ = write_scores(scores, output_dir)

        # Push to GitHub
        from . import github_store

        run_lines = [line for line in runs_path.read_text().splitlines() if line.strip()]
        await github_store.push_runs(agent_id, run_lines)

        scores_path = output_dir / "scores.jsonl"
        score_lines = [line for line in scores_path.read_text().splitlines() if line.strip()]
        await github_store.push_scores(agent_id, score_lines)

        await github_store.register_agent({
            "id": agent_id,
            "display_name": display_name,
            "model_id": model_id,
            "base_url": base_url,
            "reasoning": reasoning,
            "notes": notes,
        })

        job.status = "done"
        job.scores_url = (
            f"https://github.com/{settings.github_repo}/blob/"
            f"{settings.github_branch}/data/scores/{agent_id}.jsonl"
        )

    except Exception as e:
        log.exception("Job %s failed", job_id)
        job.status = "failed"
        job.error = str(e)
    finally:
        # Always remove the ephemeral key from env
        os.environ.pop("__EPHEMERAL__", None)

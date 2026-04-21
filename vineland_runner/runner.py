"""Main pilot runner: (agent, item, replication) → graded outcome → JSONL."""
from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as tqdm_async

from .client import LLMClient
from .config import resolve_api_key
from .grading import make_grader
from .storage import append_record, load_completed_ids
from .types import Agent, Item, PilotConfig, RunRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_single(
    agent: Agent,
    item: Item,
    replication: int,
    client: LLMClient,
    api_key: str,
    judge_agent: Optional[Agent],
    judge_api_key: Optional[str],
    runs_path: Path,
    completed: set[str],
    semaphore: asyncio.Semaphore,
) -> None:
    run_id = f"{agent.id}__{item.id}__rep{replication}"
    if run_id in completed:
        return

    async with semaphore:
        prompt = item.rendered_prompt()
        record = RunRecord(
            run_id=run_id,
            agent_id=agent.id,
            item_id=item.id,
            replication=replication,
            prompt=prompt,
            timestamp=_now_iso(),
        )
        try:
            api_response = await client.complete(
                agent,
                api_key,
                [{"role": "user", "content": prompt}],
                seed=replication,
            )
            record.response = api_response.content
            record.prompt_tokens = api_response.prompt_tokens
            record.completion_tokens = api_response.completion_tokens
            record.reasoning_tokens = api_response.reasoning_tokens
            record.latency_s = api_response.latency_s
            record.seed_unsupported = api_response.seed_unsupported

            grader = make_grader(item, client)
            grade_result = await grader.grade(
                item, api_response.content, judge_agent, judge_api_key
            )
            record.success = grade_result.success
            record.grading_detail = grade_result.detail
            record.verdict = grade_result.verdict

        except Exception as e:
            record.success = None
            record.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        append_record(runs_path, record)
        completed.add(run_id)

        if agent.wait_between_requests_s > 0:
            await asyncio.sleep(agent.wait_between_requests_s)


async def run_pilot(
    config: PilotConfig,
    items: list[Item],
    agents: dict[str, Agent],
    client: LLMClient | None = None,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.jsonl"

    # Copy pilot config to run directory for reproducibility
    import shutil, os
    # (config file copy handled by CLI)

    selected_agents = [agents[aid] for aid in config.agents if aid in agents]
    missing_agents = [aid for aid in config.agents if aid not in agents]
    if missing_agents:
        raise ValueError(f"Agents not found in agents config: {missing_agents}")

    judge_agent: Optional[Agent] = None
    judge_api_key: Optional[str] = None
    if config.judge_agent and config.judge_agent in agents:
        judge_agent = agents[config.judge_agent]
        judge_api_key = resolve_api_key(judge_agent)

    owns_client = client is None
    if client is None:
        client = LLMClient()

    try:
        for agent in selected_agents:
            try:
                api_key = resolve_api_key(agent)
            except EnvironmentError as e:
                print(f"[skip] {e}", file=__import__("sys").stderr)
                continue
            semaphore = asyncio.Semaphore(config.max_concurrency)
            completed = load_completed_ids(runs_path)

            tasks = []
            for item in items:
                for rep in range(config.n_replications):
                    tasks.append(_run_single(
                        agent, item, rep, client, api_key,
                        judge_agent, judge_api_key,
                        runs_path, completed, semaphore,
                    ))

            desc = f"{agent.id} ({len(items)} items × {config.n_replications} reps)"
            await tqdm_async.gather(*tasks, desc=desc, file=__import__("sys").stderr)

    finally:
        if owns_client:
            await client.close()

    # Print token summary
    _print_token_summary(runs_path, selected_agents)


def _print_token_summary(runs_path: Path, agents: list[Agent]) -> None:
    from collections import defaultdict
    from .storage import iter_records

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "reasoning": 0})
    for rec in iter_records(runs_path):
        t = totals[rec.agent_id]
        t["prompt"] += rec.prompt_tokens
        t["completion"] += rec.completion_tokens
        t["reasoning"] += rec.reasoning_tokens

    print("\n=== Token Usage Summary ===")
    print(f"{'Agent':<30} {'Prompt':>12} {'Completion':>12} {'Reasoning':>12}")
    print("-" * 70)
    for agent in agents:
        t = totals[agent.id]
        print(f"{agent.id:<30} {t['prompt']:>12,} {t['completion']:>12,} {t['reasoning']:>12,}")

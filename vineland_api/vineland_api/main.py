"""FastAPI application — Vineland LLM Assessment leaderboard & submission API."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from .config import settings, _is_vercel
from .models import (
    AgentSubmission,
    ItemInfo,
    ItemsResponse,
    JobStatus,
    RankingResponse,
    SubmitResponse,
)

app = FastAPI(
    title="Vineland LLM Assessment API",
    description=(
        "Leaderboard and submission API for the Vineland-adapted LLM assessment instrument. "
        "Submit a model definition to have it automatically evaluated across 72 psychometric items; "
        "explore item metadata and browse the live ranking."
    ),
    version="0.1.0",
    contact={"name": "Vineland Research Team", "email": "marco.montanari@gmail.com"},
    license_info={"name": "MIT"},
)


# ── /ranking ──────────────────────────────────────────────────────────────────

@app.get(
    "/ranking",
    response_model=RankingResponse,
    summary="Model leaderboard",
    description=(
        "Returns all evaluated models ranked by mean polytomous score (0–2 scale). "
        "Includes per-domain breakdown and overall success rate."
    ),
    tags=["Leaderboard"],
)
async def get_ranking(
    domain: Annotated[str | None, Query(description="Filter to a single domain: CM, DLS, SOC, AF")] = None,
) -> RankingResponse:
    from . import github_store
    from .ranking import compute_ranking

    score_rows = await github_store.fetch_all_scores()
    agent_meta = await github_store.fetch_agents()

    if domain:
        domain = domain.upper()
        score_rows = [r for r in score_rows if r.get("domain") == domain]

    if not score_rows:
        return RankingResponse(leaderboard=[], n_items_total=0)

    return compute_ranking(score_rows, agent_meta)


# ── /items ────────────────────────────────────────────────────────────────────

@app.get(
    "/items",
    response_model=ItemsResponse,
    summary="Item catalog",
    description=(
        "Returns all assessment items with domain, subdomain, tier, grading type, "
        "and a preview of the prompt template. Judge prompts are not exposed."
    ),
    tags=["Items"],
)
async def get_items(
    domain: Annotated[str | None, Query(description="Filter by domain: CM, DLS, SOC, AF")] = None,
    subdomain: Annotated[str | None, Query(description="Filter by subdomain code")] = None,
    tier: Annotated[int | None, Query(ge=1, le=8, description="Filter by tier")] = None,
) -> ItemsResponse:
    from . import github_store
    from vineland_runner.items import load_items

    if not settings.items_yaml.exists():
        raise HTTPException(status_code=503, detail="Items file not configured on server")

    items = load_items(settings.items_yaml)

    # Enrich with evaluation stats from GitHub scores
    score_rows = await github_store.fetch_all_scores()
    from collections import defaultdict
    item_stats: dict[str, list[float]] = defaultdict(list)
    for row in score_rows:
        item_stats[row["item_id"]].append(row["s"])

    # Apply filters
    if domain:
        items = [i for i in items if i.domain == domain.upper()]
    if subdomain:
        items = [i for i in items if i.subdomain == subdomain.upper()]
    if tier is not None:
        items = [i for i in items if i.tier == tier]

    # Build domain→subdomains map (unfiltered)
    all_items = load_items(settings.items_yaml)
    domains: dict[str, list[str]] = defaultdict(list)
    for i in all_items:
        if i.subdomain not in domains[i.domain]:
            domains[i.domain].append(i.subdomain)

    item_infos = [
        ItemInfo(
            id=i.id,
            domain=i.domain,
            subdomain=i.subdomain,
            tier=i.tier,
            prompt_preview=i.prompt_template[:200].strip(),
            criterion_type=i.success_criterion.type,
            n_agents_evaluated=len(item_stats.get(i.id, [])),
            mean_success_rate=(
                round(sum(item_stats[i.id]) / len(item_stats[i.id]), 4)
                if item_stats.get(i.id) else None
            ),
        )
        for i in items
    ]

    return ItemsResponse(items=item_infos, domains=dict(domains), total=len(item_infos))


# ── /items/{item_id} ──────────────────────────────────────────────────────────

@app.get(
    "/items/{item_id}",
    response_model=ItemInfo,
    summary="Single item detail",
    tags=["Items"],
)
async def get_item(
    item_id: Annotated[str, Path(description="Item ID, e.g. CM-REC-1")],
) -> ItemInfo:
    from vineland_runner.items import load_items

    if not settings.items_yaml.exists():
        raise HTTPException(status_code=503, detail="Items file not configured on server")

    items = load_items(settings.items_yaml)
    by_id = {i.id: i for i in items}

    if item_id not in by_id:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")

    item = by_id[item_id]
    return ItemInfo(
        id=item.id,
        domain=item.domain,
        subdomain=item.subdomain,
        tier=item.tier,
        prompt_preview=item.prompt_template[:200].strip(),
        criterion_type=item.success_criterion.type,
    )


# ── /models/submit ────────────────────────────────────────────────────────────

@app.post(
    "/models/submit",
    response_model=SubmitResponse,
    status_code=202,
    summary="Submit a model for evaluation",
    description=(
        "Registers a new model and queues a full pilot run in the background. "
        "The API key is used only during the run and is never stored. "
        "Poll `/jobs/{job_id}` to track progress."
    ),
    tags=["Submission"],
)
async def submit_model(
    submission: AgentSubmission,
    background_tasks: BackgroundTasks,
) -> SubmitResponse:
    if _is_vercel:
        raise HTTPException(
            status_code=501,
            detail=(
                "Model submission is not supported in serverless mode. "
                "Background jobs cannot outlive a Vercel function invocation. "
                "Use the Docker deployment for self-hosted evaluation runs."
            ),
        )

    from . import worker

    job_id = str(uuid.uuid4())
    job = JobStatus(
        job_id=job_id,
        agent_id=submission.id,
        status="queued",
    )
    worker._jobs[job_id] = job

    background_tasks.add_task(
        worker.run_agent_job,
        job_id=job_id,
        agent_id=submission.id,
        base_url=submission.base_url,
        model_id=submission.model_id,
        api_key=submission.api_key.get_secret_value(),
        max_tokens=submission.max_tokens,
        temperature=submission.temperature,
        reasoning=submission.reasoning,
        display_name=submission.display_name,
        notes=submission.notes,
    )

    return SubmitResponse(
        job_id=job_id,
        agent_id=submission.id,
        status="queued",
        message=(
            f"Evaluation queued for '{submission.display_name}'. "
            f"Poll /jobs/{job_id} for progress."
        ),
    )


# ── /jobs ─────────────────────────────────────────────────────────────────────

@app.get(
    "/jobs",
    response_model=list[JobStatus],
    summary="List all evaluation jobs",
    tags=["Jobs"],
)
async def list_jobs() -> list[JobStatus]:
    if _is_vercel:
        raise HTTPException(status_code=501, detail="Job tracking is not supported in serverless mode.")
    from . import worker
    return worker.list_jobs()


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Get job status",
    tags=["Jobs"],
)
async def get_job(job_id: Annotated[str, Path(description="Job UUID")]) -> JobStatus:
    if _is_vercel:
        raise HTTPException(status_code=501, detail="Job tracking is not supported in serverless mode.")
    from . import worker
    job = worker.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health", include_in_schema=False)
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

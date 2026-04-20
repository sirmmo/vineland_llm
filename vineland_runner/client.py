"""OpenAI-compatible async HTTP client with retry and timeout."""
from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .types import Agent, APIResponse


_RETRY_STATUS = {429, 500, 502, 503, 504}
_DEFAULT_TIMEOUT = 120.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout)):
        return True
    return False


def _make_retry_decorator():
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )


class LLMClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._owned = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def close(self):
        if self._owned:
            await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def complete(
        self,
        agent: Agent,
        api_key: str,
        messages: list[dict[str, str]],
        seed: int | None = None,
    ) -> APIResponse:
        @_make_retry_decorator()
        async def _call() -> APIResponse:
            payload: dict[str, Any] = {
                "model": agent.model_id,
                "messages": messages,
                "max_tokens": agent.max_tokens,
                "temperature": agent.temperature,
            }
            if seed is not None:
                payload["seed"] = seed

            if agent.reasoning:
                payload.setdefault("extra_body", {})["include_reasoning"] = True

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            t0 = time.monotonic()
            response = await self._client.post(
                f"{agent.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )

            # Respect Retry-After on 429
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    import asyncio
                    await asyncio.sleep(float(retry_after))
                response.raise_for_status()

            response.raise_for_status()
            latency = time.monotonic() - t0

            body = response.json()
            choice = body["choices"][0]["message"]
            content = choice.get("content") or ""

            usage = body.get("usage", {})
            reasoning_tokens = (
                usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                or usage.get("reasoning_tokens", 0)
                or 0
            )

            seed_unsupported = seed is not None and "seed" not in body

            return APIResponse(
                content=content,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=reasoning_tokens,
                latency_s=latency,
                raw_response=body,
                seed_unsupported=seed_unsupported,
            )

        return await _call()

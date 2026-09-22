"""The one place that talks to an LLM provider.

Everything else in this project calls ``chat()``; nothing else imports the ``groq`` package
directly. Swapping providers later (OpenAI, Anthropic, a local server) means changing only this
file -- the rest of the codebase never sees a provider-specific type or client.

Configuration is read from the environment (a ``.env`` file, loaded once here via
python-dotenv): ``GROQ_API_KEY`` and ``LLM_MODEL``. ``LLM_MODEL`` has no hardcoded fallback --
an unset or empty value is a hard, immediate error, not a silent default that could pick a
model that's since been deprecated or isn't available on this account. See ``.env.example``.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_ENV_VAR = "LLM_MODEL"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"

logger = logging.getLogger(__name__)

# Groq's SDK default is a 60s read timeout with its own built-in retry (max_retries=2, stacking
# up to 3 attempts -- potentially minutes). That's too forgiving for an interactive demo: a
# silently-throttled call should fail fast and predictably rather than tying up the request for
# minutes with no feedback. We use a tighter per-call timeout and do exactly one retry ourselves,
# with max_retries=0 on the client so the SDK's own retry loop never stacks underneath ours.
REQUEST_TIMEOUT_SECONDS = 15.0
RETRY_BACKOFF_SECONDS = 3.0

# Ceiling on completion length -- the one variable we control when a call is being throttled.
# Sized with real margin above the largest clean completions observed for Q1's two-part
# disambiguation answer (1803 and 1889 completion tokens across separate runs; see
# eval_results.md and the token-budget note below) -- not picked arbitrarily. Note this cap
# alone cannot prevent rate-limiting on the free/on-demand tier: Groq enforces an account-level
# 8000 tokens-per-minute limit, and Q1's prompt alone is ~5265 tokens, so even a "successful"
# ~1900-token completion already consumes ~90% of the whole per-minute budget by itself.
MAX_COMPLETION_TOKENS = 3000


def _model_name() -> str:
    value = os.environ.get(LLM_MODEL_ENV_VAR, "").strip()
    if not value:
        raise RuntimeError(f"{LLM_MODEL_ENV_VAR} is not set; see .env.example")
    return value


@lru_cache(maxsize=1)
def _client():
    from groq import Groq  # deferred: no SDK/network work happens just from importing this module

    api_key = os.environ.get(GROQ_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise RuntimeError(f"{GROQ_API_KEY_ENV_VAR} is not set; see .env.example")
    return Groq(api_key=api_key, max_retries=0)


def _create(client, model: str, messages: list[dict[str, str]]):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_tokens=MAX_COMPLETION_TOKENS,
    )


def chat(messages: list[dict[str, str]], capture_usage: dict[str, Any] | None = None) -> str:
    """Send ``messages`` (OpenAI-style ``{"role": ..., "content": ...}`` dicts) to the
    configured model and return the reply text. Checks ``LLM_MODEL`` before touching the
    client, so a missing model name fails immediately, before any API key check or network call.

    Bounded by ``REQUEST_TIMEOUT_SECONDS``; on a timeout or connection error, retries exactly
    once after ``RETRY_BACKOFF_SECONDS`` and logs a warning. A second failure propagates -- this
    exists so a silently-throttled call fails fast and predictably instead of hanging, not to
    paper over a genuine outage. Completions are capped at ``MAX_COMPLETION_TOKENS``.

    ``capture_usage``, when given a dict, gets Groq's own reported ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, and its server-side ``queue_time``/``prompt_time``/
    ``completion_time``/``total_time`` (seconds) -- real usage/timing from the API response
    itself, not a client-side estimate. Diagnostic only; omitted, behavior is unchanged."""
    from groq import APIConnectionError, APITimeoutError  # deferred, same reason as _client()

    model = _model_name()
    client = _client()
    try:
        response = _create(client, model, messages)
    except (APITimeoutError, APIConnectionError) as exc:
        logger.warning(
            "llm_client.chat: request failed after %.0fs (%s: %s); retrying once after %.0fs backoff",
            REQUEST_TIMEOUT_SECONDS, type(exc).__name__, exc, RETRY_BACKOFF_SECONDS,
        )
        time.sleep(RETRY_BACKOFF_SECONDS)
        response = _create(client, model, messages)
    if capture_usage is not None and response.usage is not None:
        u = response.usage
        capture_usage.update(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
            queue_time=getattr(u, "queue_time", None),
            prompt_time=getattr(u, "prompt_time", None),
            completion_time=getattr(u, "completion_time", None),
            total_time=getattr(u, "total_time", None),
        )
    return response.choices[0].message.content

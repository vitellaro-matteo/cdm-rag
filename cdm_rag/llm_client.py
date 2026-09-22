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

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_ENV_VAR = "LLM_MODEL"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"


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
    return Groq(api_key=api_key)


def chat(messages: list[dict[str, str]]) -> str:
    """Send ``messages`` (OpenAI-style ``{"role": ..., "content": ...}`` dicts) to the
    configured model and return the reply text. Checks ``LLM_MODEL`` before touching the
    client, so a missing model name fails immediately, before any API key check or network call."""
    model = _model_name()
    client = _client()
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

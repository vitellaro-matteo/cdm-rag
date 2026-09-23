"""Optional, deliberately lightweight protection for a public demo deployment of ``POST /ask``.

Not security-grade -- there is no such thing as a secure client-visible shared secret, and the
rate limiter is a single in-memory counter, not real infrastructure. The actual goal is narrower:
stop *accidental or automated* Groq-quota exhaustion from outside traffic while a public demo URL
isn't actively being shown, without adding any real friction for local development or Docker use
that doesn't opt in.

Both mechanisms are independently toggled by an environment variable and are strictly additive:
unset either variable and that mechanism is a true no-op -- not merely permissive, literally never
invoked beyond reading `os.environ` -- so existing behavior (local dev, Docker without these vars
set, every test written before this module existed) is completely unaffected. Wired into
``api.py`` on the ``POST /ask`` route only; ``/health`` and ``/entities/{name}`` never import or
touch this module at all, since neither one calls Groq.
"""

from __future__ import annotations

import os
import time
from collections import deque

from fastapi import Header, HTTPException

DEMO_ACCESS_KEY_ENV_VAR = "DEMO_ACCESS_KEY"
DEMO_RATE_LIMIT_ENV_VAR = "DEMO_RATE_LIMIT_PER_HOUR"
DEMO_KEY_HEADER = "X-Demo-Key"

_WINDOW_SECONDS = 3600.0
# Global, in-memory, single-process request timestamps for the rolling-hour rate limit -- reset
# on process restart, shared across every caller (this is a demo-traffic brake, not a per-user
# quota). See ``reset_rate_limit_state`` for tests.
_request_times: deque[float] = deque()


def require_demo_access(x_demo_key: str | None = Header(default=None, alias=DEMO_KEY_HEADER)) -> None:
    """FastAPI dependency: 401s if ``DEMO_ACCESS_KEY`` is set in the environment and the request's
    ``X-Demo-Key`` header doesn't match it exactly (including a missing header). A true no-op --
    the header is never even compared -- when ``DEMO_ACCESS_KEY`` is unset or empty."""
    expected = os.environ.get(DEMO_ACCESS_KEY_ENV_VAR, "")
    if not expected:
        return
    if x_demo_key != expected:
        raise HTTPException(status_code=401, detail=f"missing or invalid {DEMO_KEY_HEADER} header")


def _configured_rate_limit() -> int | None:
    """``DEMO_RATE_LIMIT_PER_HOUR`` as a positive int, or ``None`` when unset, empty, not a valid
    integer, or not positive -- all of which mean "rate limiting is off," not an error, since this
    is an optional deployment knob, not required configuration."""
    value = os.environ.get(DEMO_RATE_LIMIT_ENV_VAR, "").strip()
    if not value:
        return None
    try:
        limit = int(value)
    except ValueError:
        return None
    return limit if limit > 0 else None


def enforce_rate_limit() -> None:
    """FastAPI dependency: a simple global rolling-hour counter across all callers, 429ing once
    ``DEMO_RATE_LIMIT_PER_HOUR`` requests have been made in the trailing 60 minutes. A true no-op
    when that env var is unset -- ``_request_times`` is never even touched. Deliberately not
    distributed or persistent: a single in-memory deque, cleared on restart, is exactly the amount
    of protection a single-instance demo deployment needs against accidental or automated
    over-use, not an attempt at real rate-limiting infrastructure. Runs after
    ``require_demo_access`` on the route (see ``api.py``), so a request rejected for a bad key
    never consumes a slot of this budget."""
    limit = _configured_rate_limit()
    if limit is None:
        return
    now = time.time()
    while _request_times and now - _request_times[0] > _WINDOW_SECONDS:
        _request_times.popleft()
    if len(_request_times) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"demo rate limit reached ({limit} requests/hour); please try again later",
        )
    _request_times.append(now)


def reset_rate_limit_state() -> None:
    """Clears the in-memory request-time window. Not used by the app itself -- only by tests, so
    one test's rate-limit usage doesn't leak into the next."""
    _request_times.clear()

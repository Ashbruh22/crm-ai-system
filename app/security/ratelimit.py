"""Rate limiting (spec section 10).

The demo is public and every endpoint costs real CPU on a free-tier container,
so the limits exist to keep one enthusiastic visitor — or a crawler — from
starving everyone else.

Limits are tiered by what the call actually costs:

* reads are cheap and cached, so they get a generous ceiling
* ``/what-if`` runs both models plus SHAP on every drag of a slider
* ``/api/events`` enqueues work the consumer must do, so it is the tightest

Behind Render the client IP arrives in ``X-Forwarded-For``; taking
``request.client.host`` would see the proxy and rate-limit the whole world as
one caller.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

#: Generous: listing deals and reading a deal are cache-friendly.
READ_LIMIT = "120/minute"
#: Scoring is the expensive path.
SCORE_LIMIT = "30/minute"
#: Sliders fire on every change, debounced client-side to ~4/second.
WHATIF_LIMIT = "60/minute"
#: Ingestion creates downstream work for the consumer.
EVENT_LIMIT = "20/minute"
#: Admin actions are destructive and token-gated; this is defence in depth.
ADMIN_LIMIT = "5/minute"


def client_key(request: Request) -> str:
    """Identify the caller, honouring the proxy Render puts in front of us."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Left-most entry is the original client; the rest are proxies.
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return get_remote_address(request)


#: Storage is in-process. That is correct for a single Render service: a Redis
#: backend would add a network round trip to every request to coordinate
#: workers that do not exist.
#: headers_enabled stays off deliberately. With it on, slowapi injects
#: X-RateLimit-* into a `response` kwarg and therefore requires every decorated
#: endpoint to declare `response: Response` — an unused parameter on each route
#: purely to satisfy the library. The 429 handler sets Retry-After itself, which
#: is the header a client can actually act on.
limiter = Limiter(
    key_func=client_key,
    default_limits=[READ_LIMIT],
    headers_enabled=False,
)

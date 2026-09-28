"""Server-Sent Events (spec section 7).

``GET /api/stream`` — the live channel behind the demo's "fire an event and watch
the score move" moment.

SSE rather than WebSockets because the traffic is one-directional (server tells
dashboard that a score changed), it survives proxies that mishandle WebSocket
upgrades, and ``EventSource`` reconnects on its own. There is no client-to-server
channel to give up.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.bus.broadcast import Broadcaster, sse_comment, sse_format

log = logging.getLogger("crm_ai.stream")

router = APIRouter()

#: Emit a comment frame at least this often. Idle connections get closed by
#: proxies and by Render's router; a periodic no-op keeps them open.
KEEPALIVE_SECONDS = 15


@router.get("")
async def stream(request: Request) -> StreamingResponse:
    """Stream ``score_updated`` and ``action_created`` events to the dashboard."""
    redis = getattr(request.app.state, "redis", None)
    broadcaster = Broadcaster(redis)

    async def generate():
        # Tell the client immediately that the channel is open, so the UI can
        # drop its "connecting" state without waiting for real traffic.
        yield sse_format("connected", {"status": "ok"})

        last_sent = asyncio.get_event_loop().time()
        try:
            async for message in broadcaster.subscribe():
                if await request.is_disconnected():
                    break

                now = asyncio.get_event_loop().time()
                if not message:
                    if now - last_sent >= KEEPALIVE_SECONDS:
                        yield sse_comment()
                        last_sent = now
                    continue

                yield sse_format(
                    message.get("event", "message"), message.get("data", {})
                )
                last_sent = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("stream ended: %s", exc)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Nginx buffers proxied responses by default, which holds SSE
            # frames back until the buffer fills — the stream looks dead.
            "X-Accel-Buffering": "no",
        },
    )

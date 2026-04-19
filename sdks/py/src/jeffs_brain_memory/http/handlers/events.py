# SPDX-License-Identifier: Apache-2.0
"""`/events` SSE endpoint. Emits ready + change frames for a brain."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ..daemon import ChangeEvent
from ._shared import resolve_brain

_EVENTS_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
_PING_INTERVAL = 25.0


def _format_event(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


async def events(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br

    queue: asyncio.Queue[ChangeEvent] = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()

    def on_change(evt: ChangeEvent) -> None:
        # May fire from outside the event loop; RuntimeError means the
        # loop has already closed.
        try:
            loop.call_soon_threadsafe(_offer, queue, evt)
        except RuntimeError:
            return

    unsubscribe = await br.store.subscribe(on_change)

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            yield _format_event("ready", "ok")
            while True:
                if await request.is_disconnected():
                    return
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=_PING_INTERVAL)
                except asyncio.TimeoutError:
                    yield _format_event("ping", "keepalive")
                    continue
                payload = {
                    "kind": evt.kind,
                    "path": evt.path,
                    "when": evt.when.isoformat(),
                }
                if evt.old_path:
                    payload["old_path"] = evt.old_path
                if evt.reason:
                    payload["reason"] = evt.reason
                yield _format_event("change", json.dumps(payload))
        finally:
            unsubscribe()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_EVENTS_SSE_HEADERS,
    )


def _offer(queue: asyncio.Queue[ChangeEvent], evt: ChangeEvent) -> None:
    """Non-blocking enqueue; drops on overflow to match the Go daemon."""
    try:
        queue.put_nowait(evt)
    except asyncio.QueueFull:
        return

# SPDX-License-Identifier: Apache-2.0
"""Bearer-token auth middleware.

Matches the reference Go daemon: when a token is configured, every
request except `/healthz` must carry `Authorization: Bearer <token>`.
Missing or mismatched headers return 401 Problem+JSON.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..problem import forbidden, unauthorized

_EXEMPT_PATHS = frozenset({"/healthz"})


class AuthMiddleware:
    """ASGI middleware enforcing a shared bearer token."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or ())
        raw = headers.get(b"authorization", b"")
        header = raw.decode("latin-1") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if not header:
            response = unauthorized()
            await response(scope, receive, send)
            return
        expected = f"Bearer {self.token}"
        if header != expected:
            # Go's reference emits 403 when a token is present but
            # wrong; the spec tolerates either. We match Go.
            response = forbidden()
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def _noop(message: Message) -> None:
    """Placeholder for type-checker completeness."""
    return None

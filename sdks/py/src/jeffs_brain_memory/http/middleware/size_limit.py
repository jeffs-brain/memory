# SPDX-License-Identifier: Apache-2.0
"""Request body size limits per `spec/PROTOCOL.md`.

Enforced per-endpoint via route-specific reads in the handlers so this
middleware only short-circuits on obvious overflows announced via a
`Content-Length` header larger than the global cap.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from ..problem import payload_too_large

# 16 MiB: no endpoint in the protocol accepts more than 8 MiB even
# after base64 decode. The fine-grained 2 MiB caps are applied in
# handlers, since Content-Length on a base64 body overstates the
# decoded size.
GLOBAL_BODY_CAP = 16 * 1024 * 1024


class SizeLimitMiddleware:
    """Reject requests whose declared body clearly exceeds the cap."""

    def __init__(self, app: ASGIApp, cap: int = GLOBAL_BODY_CAP) -> None:
        self.app = app
        self.cap = cap

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or ())
        raw = headers.get(b"content-length")
        if raw is not None:
            try:
                declared = int(raw.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                declared = -1
            if declared > self.cap:
                response = payload_too_large(
                    f"request body exceeds {self.cap} bytes"
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

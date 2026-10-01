# SPDX-License-Identifier: Apache-2.0
"""Bearer-token auth and the loopback guard.

Matches the reference Go daemon. With a token configured, every request
except ``/healthz`` must carry ``Authorization: Bearer <token>``: a
missing header is 401 and a wrong token 403. Without a token the daemon
only answers local clients addressing it by a loopback name: a foreign
``Host`` is 421 and a foreign browser ``Origin`` is 403, which closes
DNS rebinding and cross-site requests into an unauthenticated daemon.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

from ..problem import forbidden, misdirected_request, unauthorized

__all__ = [
    "AuthMiddleware",
    "LoopbackGuardMiddleware",
    "is_loopback_host",
    "valid_bearer_token",
]

_EXEMPT_PATHS = frozenset({"/healthz"})
_BEARER_RE = re.compile(r"^bearer +(.+)$", re.IGNORECASE)


def valid_bearer_token(header: str, expected: str) -> bool:
    """Check an Authorization header against the expected token. The
    scheme is case-insensitive (RFC 7235) and both sides are hashed
    before a constant-time comparison, so neither content nor length
    leaks through timing."""
    match = _BEARER_RE.match(header.strip())
    presented = match.group(1).strip() if match else ""
    if not presented:
        return False
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals in any spelling,
    including IPv4-mapped IPv6. Brackets are tolerated."""
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if bare.lower() == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(bare)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped.is_loopback
    return addr.is_loopback


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return bytes(value).decode("latin-1")
    return ""


def _host_only(host_header: str) -> str | None:
    """Return the host of a ``Host`` header, or ``None`` when the header
    is malformed (userinfo, a path, or an unparsable port)."""
    try:
        parts = urlsplit(f"http://{host_header}")
        _ = parts.port
    except ValueError:
        return None
    if not parts.hostname or "@" in parts.netloc or parts.path or parts.query or parts.fragment:
        return None
    return parts.hostname


def _is_loopback_origin(origin: str) -> bool:
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and is_loopback_host(parts.hostname or "")


class AuthMiddleware:
    """ASGI middleware enforcing a shared bearer token."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.token or scope.get("path", "") in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        header = _header(scope, b"authorization")
        if not header:
            await unauthorized("missing Authorization header")(scope, receive, send)
            return
        if not valid_bearer_token(header, self.token):
            await forbidden("invalid bearer token")(scope, receive, send)
            return
        await self.app(scope, receive, send)


class LoopbackGuardMiddleware:
    """ASGI middleware active only when no token is configured."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.token or scope.get("path", "") in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        host = _host_only(_header(scope, b"host"))
        if host is None or not is_loopback_host(host):
            response = misdirected_request("unauthenticated daemon only serves loopback hosts")
            await response(scope, receive, send)
            return
        origin = _header(scope, b"origin")
        if origin and not _is_loopback_origin(origin):
            response = forbidden("cross-origin requests are refused by an unauthenticated daemon")
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

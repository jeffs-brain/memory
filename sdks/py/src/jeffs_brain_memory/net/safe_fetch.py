# SPDX-License-Identifier: Apache-2.0
"""SSRF-safe outbound fetch for user- or model-supplied URLs.

Every hop, including each redirect, is validated: the scheme must be
http or https, and every address the host resolves to must be public.
The request is then sent to the validated address with the original
``Host`` header and TLS server name, so a second DNS answer (rebinding)
cannot swap in an internal address between the check and the connect.
Bodies are capped while streaming and the whole fetch carries a
timeout.

The blocklist matches ``go/knowledge/safefetch.go`` and
``sdks/ts/memory/src/net/safe-fetch.ts``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import httpx

__all__ = [
    "DEFAULT_SAFE_FETCH_MAX_BYTES",
    "DEFAULT_SAFE_FETCH_MAX_REDIRECTS",
    "DEFAULT_SAFE_FETCH_TIMEOUT",
    "FetchFailedError",
    "Resolver",
    "SafeFetchResult",
    "UnsafeUrlError",
    "is_blocked_address",
    "resolve_public_address",
    "safe_fetch",
]

DEFAULT_SAFE_FETCH_TIMEOUT = 30.0
DEFAULT_SAFE_FETCH_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_SAFE_FETCH_MAX_REDIRECTS = 5

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

_BLOCKED_V4 = tuple(
    ipaddress.IPv4Network(net)
    for net in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_BLOCKED_V6 = tuple(
    ipaddress.IPv6Network(net)
    for net in (
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    )
)
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")

Resolver = Callable[[str], Awaitable[Sequence[str]]]


class UnsafeUrlError(ValueError):
    """The URL is malformed, uses a scheme other than http(s), or
    resolves to a non-public address. Maps to 400 at the HTTP layer."""


class FetchFailedError(Exception):
    """The upstream could not be reached, answered with a non-2xx
    status, or sent a body over the limit. Maps to 502 at the HTTP
    layer."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class SafeFetchResult:
    url: str
    status: int
    content_type: str
    body: bytes


def is_blocked_address(ip: str) -> bool:
    """True when ``ip`` is loopback, private, link-local, CGN, multicast,
    reserved or unspecified, including IPv4-mapped and NAT64 forms.
    Anything that is not an IP literal is treated as blocked."""
    clean = ip[1:-1] if ip.startswith("[") and ip.endswith("]") else ip
    clean = clean.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(clean)
    except ValueError:
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        embedded = addr.ipv4_mapped
        if embedded is None and addr in _NAT64:
            embedded = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
        if embedded is not None:
            return any(embedded in net for net in _BLOCKED_V4)
        return any(addr in net for net in _BLOCKED_V6)
    return any(addr in net for net in _BLOCKED_V4)


async def _system_resolver(host: str) -> Sequence[str]:
    try:
        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        pass
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


def _parse_external_url(raw: str) -> httpx.URL:
    try:
        url = httpx.URL(raw.strip())
    except (httpx.InvalidURL, TypeError, ValueError) as exc:
        raise UnsafeUrlError("invalid URL") from exc
    if url.scheme not in ("http", "https"):
        scheme = f"{url.scheme}:" if url.scheme else "(none)"
        raise UnsafeUrlError(f"unsupported scheme {scheme} (only http and https allowed)")
    if not url.host:
        raise UnsafeUrlError("URL missing host")
    if url.userinfo:
        raise UnsafeUrlError("URL must not carry credentials")
    return url


async def resolve_public_address(
    url: httpx.URL,
    resolver: Resolver | None = None,
    is_blocked: Callable[[str], bool] = is_blocked_address,
) -> str:
    """Resolve the URL's host and return the first address, refusing the
    URL when any resolved address is non-public."""
    host = url.host
    try:
        addresses = list(await (resolver or _system_resolver)(host))
    except (OSError, UnicodeError) as exc:
        raise FetchFailedError(f"DNS lookup failed for {host}") from exc
    if not addresses:
        raise FetchFailedError(f"no DNS records for {host}")
    if any(is_blocked(address) for address in addresses):
        raise UnsafeUrlError(f"{host} resolves to a non-public address")
    return addresses[0]


@dataclass(frozen=True, slots=True)
class _Hop:
    status: int
    location: str | None
    content_type: str
    body: bytes


async def _fetch_hop(
    client: httpx.AsyncClient,
    url: httpx.URL,
    address: str,
    headers: dict[str, str],
    max_bytes: int,
) -> _Hop:
    pinned = url.copy_with(host=address)
    request_headers = {**headers, "Host": url.netloc.decode("ascii")}
    extensions = {"sni_hostname": url.host} if url.scheme == "https" else {}
    try:
        async with client.stream(
            "GET", pinned, headers=request_headers, extensions=extensions
        ) as resp:
            location = resp.headers.get("location")
            if resp.status_code in _REDIRECT_STATUSES and location:
                return _Hop(resp.status_code, location, "", b"")
            if resp.status_code < 200 or resp.status_code >= 300:
                raise FetchFailedError(
                    f"upstream responded HTTP {resp.status_code}", resp.status_code
                )
            chunks: list[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise FetchFailedError(f"response exceeds {max_bytes} byte limit")
                chunks.append(chunk)
            return _Hop(
                resp.status_code,
                None,
                resp.headers.get("content-type", ""),
                b"".join(chunks),
            )
    except httpx.HTTPError as exc:
        raise FetchFailedError(f"fetch failed: {type(exc).__name__}") from exc


async def safe_fetch(
    raw: str,
    *,
    timeout: float = DEFAULT_SAFE_FETCH_TIMEOUT,
    max_bytes: int = DEFAULT_SAFE_FETCH_MAX_BYTES,
    max_redirects: int = DEFAULT_SAFE_FETCH_MAX_REDIRECTS,
    headers: dict[str, str] | None = None,
    resolver: Resolver | None = None,
    is_blocked: Callable[[str], bool] = is_blocked_address,
) -> SafeFetchResult:
    """GET ``raw`` with the SSRF guard applied to every hop.

    Raises :class:`UnsafeUrlError` for refused URLs and
    :class:`FetchFailedError` for upstream failures. ``resolver`` and
    ``is_blocked`` exist for tests; production uses the system resolver
    and :func:`is_blocked_address`.
    """
    request_headers = {
        "Accept": "text/plain, text/markdown, text/html, application/pdf;q=0.9, */*;q=0.5",
        **(headers or {}),
    }
    url = _parse_external_url(raw)
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=timeout,
            ) as client:
                for hop in range(max_redirects + 1):
                    address = await resolve_public_address(url, resolver, is_blocked)
                    res = await _fetch_hop(client, url, address, request_headers, max_bytes)
                    if res.location is None:
                        return SafeFetchResult(str(url), res.status, res.content_type, res.body)
                    if hop >= max_redirects:
                        break
                    try:
                        target = url.join(res.location)
                    except (httpx.InvalidURL, ValueError) as exc:
                        raise FetchFailedError("redirect to an invalid URL") from exc
                    url = _parse_external_url(str(target))
    except TimeoutError as exc:
        raise FetchFailedError(f"timed out after {timeout}s") from exc
    raise FetchFailedError(f"stopped after {max_redirects} redirects")

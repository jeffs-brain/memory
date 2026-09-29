# SPDX-License-Identifier: Apache-2.0
"""SSRF guard for user- and model-supplied URLs. Mirrors
``sdks/ts/memory/src/net/safe-fetch.test.ts``."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jeffs_brain_memory.net import (
    FetchFailedError,
    UnsafeUrlError,
    is_blocked_address,
    safe_fetch,
)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "127.8.8.8",
        "10.1.2.3",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "192.0.0.8",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "255.255.255.255",
        "::",
        "::1",
        "fd00::1",
        "fe80::1",
        "fe80::1%eth0",
        "fec0::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:7f00:1",
        "64:ff9b::a00:1",
        "[::1]",
        "not-an-ip",
    ],
)
def test_blocks(ip: str) -> None:
    assert is_blocked_address(ip)


@pytest.mark.parametrize(
    "ip",
    [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",
        "2607:f8b0:4004:800::200e",
        "::ffff:8.8.8.8",
        "64:ff9b::808:808",
    ],
)
def test_allows(ip: str) -> None:
    assert not is_blocked_address(ip)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _send(self, status: int, headers: dict[str, str], body: bytes = b"") -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        routes: dict[str, tuple[int, dict[str, str], bytes]] = {
            "/ok": (200, {"Content-Type": "text/markdown; charset=utf-8"}, b"# hello"),
            "/host": (200, {"Content-Type": "text/plain"}, self.headers["Host"].encode()),
            "/to-internal": (302, {"Location": "http://internal.test/secret"}, b""),
            "/to-file": (302, {"Location": "file:///etc/passwd"}, b""),
            "/relative": (301, {"Location": "/ok"}, b""),
            "/loop": (302, {"Location": "/loop"}, b""),
            "/big": (200, {"Content-Type": "text/plain"}, b"x" * 2048),
        }
        if self.path == "/slow":
            time.sleep(0.5)
            # The client has timed out and hung up by now.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self._send(200, {}, b"late")
            return
        status, headers, body = routes.get(self.path, (404, {}, b"missing"))
        self._send(status, headers, body)


@pytest.fixture(scope="module")
def port() -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


async def _resolver(host: str) -> Sequence[str]:
    # `public.test` is treated as public and pinned to the local server;
    # `internal.test` resolves to a private address.
    table = {
        "public.test": ["127.0.0.1"],
        "internal.test": ["10.0.0.1"],
        "mixed.test": ["127.0.0.1", "10.0.0.1"],
    }
    if host not in table:
        raise OSError("NXDOMAIN")
    return table[host]


def _is_blocked(ip: str) -> bool:
    return ip.startswith("10.")


def _at(port: int, path: str, host: str = "public.test") -> str:
    return f"http://{host}:{port}{path}"


async def test_refuses_loopback_with_default_policy(port: int) -> None:
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(f"http://127.0.0.1:{port}/ok")


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:1/ok",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
async def test_refuses_internal_literals(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/",
        "not a url",
        "http://user:pw@example.com/",
    ],
)
async def test_refuses_bad_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(url)


async def test_fetches_through_the_pinned_address(port: int) -> None:
    res = await safe_fetch(_at(port, "/ok"), resolver=_resolver, is_blocked=_is_blocked)
    assert res.status == 200
    assert res.content_type == "text/markdown; charset=utf-8"
    assert res.body == b"# hello"


async def test_keeps_the_original_host_header(port: int) -> None:
    res = await safe_fetch(_at(port, "/host"), resolver=_resolver, is_blocked=_is_blocked)
    assert res.body.decode() == f"public.test:{port}"


async def test_refuses_host_when_any_address_is_blocked(port: int) -> None:
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(_at(port, "/ok", "mixed.test"), resolver=_resolver, is_blocked=_is_blocked)


async def test_revalidates_redirect_targets(port: int) -> None:
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(_at(port, "/to-internal"), resolver=_resolver, is_blocked=_is_blocked)
    with pytest.raises(UnsafeUrlError):
        await safe_fetch(_at(port, "/to-file"), resolver=_resolver, is_blocked=_is_blocked)


async def test_follows_relative_redirects(port: int) -> None:
    res = await safe_fetch(_at(port, "/relative"), resolver=_resolver, is_blocked=_is_blocked)
    assert res.body == b"# hello"
    assert res.url == _at(port, "/ok")


async def test_caps_the_redirect_chain(port: int) -> None:
    with pytest.raises(FetchFailedError, match="redirects"):
        await safe_fetch(
            _at(port, "/loop"), resolver=_resolver, is_blocked=_is_blocked, max_redirects=2
        )


async def test_maps_non_2xx_with_status(port: int) -> None:
    with pytest.raises(FetchFailedError) as info:
        await safe_fetch(_at(port, "/nope"), resolver=_resolver, is_blocked=_is_blocked)
    assert info.value.status == 404


async def test_caps_the_body_size(port: int) -> None:
    with pytest.raises(FetchFailedError, match="byte limit"):
        await safe_fetch(
            _at(port, "/big"), resolver=_resolver, is_blocked=_is_blocked, max_bytes=1024
        )


async def test_times_out(port: int) -> None:
    with pytest.raises(FetchFailedError):
        await safe_fetch(
            _at(port, "/slow"), resolver=_resolver, is_blocked=_is_blocked, timeout=0.05
        )


async def test_reports_dns_failures_as_upstream_errors(port: int) -> None:
    with pytest.raises(FetchFailedError):
        await safe_fetch(
            _at(port, "/ok", "nowhere.test"), resolver=_resolver, is_blocked=_is_blocked
        )

# SPDX-License-Identifier: Apache-2.0
"""Daemon hardening: bearer comparison, the loopback guard for daemons
without a token, detail-free 500s, confined server-path ingest, the SSRF
guard on URL ingest and the bind policy. Mirrors
``sdks/ts/memory/src/http/security.test.ts``,
``go/internal/httpd/local_test.go`` and
``go/cmd/memory/handler_ingest_test.go``."""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import click
import pytest
from starlette.testclient import TestClient

from jeffs_brain_memory.cli.commands.serve import _assert_bind_allowed, _parse_addr
from jeffs_brain_memory.http import create_app
from jeffs_brain_memory.http.middleware.auth import is_loopback_host, valid_bearer_token
from jeffs_brain_memory.http.problem import INTERNAL_ERROR_DETAIL
from jeffs_brain_memory.knowledge import ingest as knowledge_ingest
from jeffs_brain_memory.net import FetchFailedError

LOCAL = "http://127.0.0.1"


@pytest.mark.parametrize(
    ("header", "ok"),
    [
        ("Bearer s3cret", True),
        ("bearer s3cret", True),
        ("BEARER   s3cret", True),
        ("Bearer s3cre", False),
        ("Bearer s3cret-and-more", False),
        ("Basic s3cret", False),
        ("Bearer", False),
        ("Bearer ", False),
        ("", False),
    ],
)
def test_valid_bearer_token(header: str, ok: bool) -> None:
    assert valid_bearer_token(header, "s3cret") is ok


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("127.0.0.1", True),
        ("127.9.9.9", True),
        ("::1", True),
        ("[::1]", True),
        ("0:0:0:0:0:0:0:1", True),
        ("::ffff:127.0.0.1", True),
        ("0.0.0.0", False),
        ("10.0.0.1", False),
        ("evil.test", False),
        ("localhost.evil.test", False),
        ("", False),
    ],
)
def test_is_loopback_host(host: str, ok: bool) -> None:
    assert is_loopback_host(host) is ok


@pytest.fixture
def root(tmp_path: Path) -> Path:
    brains = tmp_path / "brains"
    brains.mkdir()
    return brains


def _client(app, base_url: str = LOCAL) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(app, base_url=base_url, raise_server_exceptions=False)


# -- Loopback guard -------------------------------------------------------


def test_unauthenticated_daemon_refuses_foreign_host(root: Path) -> None:
    with _client(create_app(root=root), "http://evil.test") as client:
        resp = client.get("/v1/brains")
        assert resp.status_code == 421
        assert resp.json()["code"] == "misdirected_request"
        assert client.get("/healthz").status_code == 200


def test_unauthenticated_daemon_refuses_malformed_host(root: Path) -> None:
    with _client(create_app(root=root)) as client:
        for host in ("x@127.0.0.1", "127.0.0.1/evil", "127.0.0.1:notaport"):
            assert client.get("/v1/brains", headers={"Host": host}).status_code == 421


def test_unauthenticated_daemon_refuses_foreign_origin(root: Path) -> None:
    with _client(create_app(root=root)) as client:
        resp = client.get("/v1/brains", headers={"Origin": "https://evil.test"})
        assert resp.status_code == 403
        ok = client.get("/v1/brains", headers={"Origin": "http://localhost:5173"})
        assert ok.status_code == 200


@pytest.mark.parametrize("host", ["127.0.0.1:8080", "localhost", "[::1]:9", "[::ffff:7f00:1]"])
def test_unauthenticated_daemon_serves_loopback_hosts(root: Path, host: str) -> None:
    with _client(create_app(root=root)) as client:
        assert client.get("/v1/brains", headers={"Host": host}).status_code == 200


def test_token_replaces_the_loopback_guard(root: Path) -> None:
    with _client(create_app(root=root, auth_token="s3cret"), "http://memory.internal") as client:
        assert client.get("/v1/brains").status_code == 401
        assert (
            client.get("/v1/brains", headers={"Authorization": "Bearer wrong"}).status_code == 403
        )
        good = client.get("/v1/brains", headers={"Authorization": "bearer s3cret"})
        assert good.status_code == 200
        assert client.get("/healthz").status_code == 200


# -- Internal errors ------------------------------------------------------


def test_internal_errors_are_logged_not_returned(
    root: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(root=root)
    with _client(app) as client:
        assert client.post("/v1/brains", json={"brainId": "boom"}).status_code == 201

        async def explode(_: str) -> None:
            raise RuntimeError("secret detail /var/lib/private.db")

        monkeypatch.setattr(app.state.daemon.brains, "get", explode)
        with caplog.at_level(logging.ERROR):
            resp = client.post("/v1/brains/boom/search", json={"query": "x"})
        assert resp.status_code == 500
        assert resp.json() == {
            "status": 500,
            "title": "Internal Server Error",
            "detail": INTERNAL_ERROR_DETAIL,
            "code": "internal_error",
        }
        assert "secret detail" not in resp.text
        assert "secret detail" in caplog.text


def test_unhandled_exceptions_become_problem_json(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jeffs_brain_memory.http.handlers import brains as brains_mod

    async def explode(_request: object) -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(brains_mod, "list_brains", explode)
    with _client(create_app(root=root)) as client:
        resp = client.get("/v1/brains")
    assert resp.status_code == 500
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["detail"] == INTERNAL_ERROR_DETAIL


# -- Server-path ingest ---------------------------------------------------


@pytest.fixture
def ingest_tree(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    allowed = tmp_path / "allowed"
    (allowed / "sub").mkdir(parents=True)
    (allowed / "note.md").write_text("# Allowed\n\nInside the ingest root.\n")
    outside = tmp_path / "secret.md"
    outside.write_text("# Secret\n")
    os.symlink(outside, allowed / "escape.md")
    yield allowed, outside


def test_path_ingest_is_disabled_without_a_root(root: Path, ingest_tree: tuple[Path, Path]) -> None:
    allowed, _ = ingest_tree
    with _client(create_app(root=root)) as client:
        client.post("/v1/brains", json={"brainId": "noroot"})
        resp = client.post("/v1/brains/noroot/ingest/file", json={"path": str(allowed / "note.md")})
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"]


def test_path_ingest_is_confined_to_the_root(root: Path, ingest_tree: tuple[Path, Path]) -> None:
    allowed, outside = ingest_tree
    with _client(create_app(root=root, ingest_root=allowed)) as client:
        client.post("/v1/brains", json={"brainId": "rooted"})
        url = "/v1/brains/rooted/ingest/file"
        cases = [
            ("note.md", 200),
            (str(allowed / "note.md"), 200),
            (str(outside), 403),
            ("../secret.md", 403),
            ("../does-not-exist.md", 403),
            ("escape.md", 403),
            ("missing.md", 400),
            ("sub", 400),
        ]
        for path, want in cases:
            resp = client.post(url, json={"path": path})
            assert resp.status_code == want, (path, resp.text)


def test_inline_content_needs_no_path(root: Path) -> None:
    with _client(create_app(root=root)) as client:
        client.post("/v1/brains", json={"brainId": "inline"})
        content = base64.b64encode(b"# Inline\n\nNo path at all.\n").decode()
        resp = client.post(
            "/v1/brains/inline/ingest/file",
            json={"contentBase64": content, "contentType": "text/markdown"},
        )
        assert resp.status_code == 200, resp.text
        empty = client.post("/v1/brains/inline/ingest/file", json={})
        assert empty.status_code == 400


# -- URL ingest -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:1/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "file:///etc/passwd",
    ],
)
def test_url_ingest_refuses_internal_targets(root: Path, url: str) -> None:
    with _client(create_app(root=root)) as client:
        client.post("/v1/brains", json={"brainId": "urls"})
        resp = client.post("/v1/brains/urls/ingest/url", json={"url": url})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "validation_error"


def test_url_ingest_maps_upstream_failures_to_502(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(*_args: object, **_kwargs: object) -> None:
        raise FetchFailedError("upstream responded HTTP 503", 503)

    monkeypatch.setattr(knowledge_ingest, "safe_fetch", failing)
    with _client(create_app(root=root)) as client:
        client.post("/v1/brains", json={"brainId": "upstream"})
        resp = client.post("/v1/brains/upstream/ingest/url", json={"url": "https://example.com/"})
        assert resp.status_code == 502
        assert resp.json()["code"] == "bad_gateway"


# -- Bind policy ----------------------------------------------------------


@pytest.mark.parametrize(
    ("addr", "host", "port"),
    [
        (":8080", "127.0.0.1", 8080),
        ("127.0.0.1:18841", "127.0.0.1", 18841),
        ("0.0.0.0:9000", "0.0.0.0", 9000),
        ("[::1]:8080", "::1", 8080),
    ],
)
def test_parse_addr(addr: str, host: str, port: int) -> None:
    assert _parse_addr(addr) == (host, port)


def test_bind_policy() -> None:
    _assert_bind_allowed("127.0.0.1", None)
    _assert_bind_allowed("::1", None)
    _assert_bind_allowed("0.0.0.0", "s3cret")
    with pytest.raises(click.UsageError):
        _assert_bind_allowed("0.0.0.0", None)
    with pytest.raises(click.UsageError):
        _assert_bind_allowed("192.168.1.10", "")

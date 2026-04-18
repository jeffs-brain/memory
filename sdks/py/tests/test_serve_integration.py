# SPDX-License-Identifier: Apache-2.0
"""Integration tests mirroring Go's `serve_integration_test.go`.

Every case drives the Python daemon via a background uvicorn server
so SSE streams and auth middleware behave as they will in production.
The LLM provider is replaced with a deterministic `FakeProvider` on
cases that exercise `/ask`, keeping the assertions reproducible.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import socket
import threading
import time
from typing import Iterator

import httpx
import pytest
import uvicorn

from jeffs_brain_memory.http import Daemon, create_app
from jeffs_brain_memory.llm import FakeProvider


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextlib.contextmanager
def _run_app(app) -> Iterator[str]:  # type: ignore[no-untyped-def]
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                probe.connect(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _must_create_brain(client: httpx.Client, brain_id: str) -> None:
    resp = client.post("/v1/brains", json={"brainId": brain_id})
    if resp.status_code != 201:
        pytest.fail(f"create brain {brain_id}: {resp.status_code} {resp.text}")


# -- 1. Health --------------------------------------------------------------


def test_serve_health(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}


# -- 2. Brain lifecycle -----------------------------------------------------


def test_serve_brain_lifecycle(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            create = client.post("/v1/brains", json={"brainId": "alpha"})
            assert create.status_code == 201, create.text

            get = client.get("/v1/brains/alpha")
            assert get.status_code == 200

            listed = client.get("/v1/brains")
            assert listed.status_code == 200
            items = listed.json().get("items", [])
            assert items and items[0].get("brainId") == "alpha"

            no_confirm = client.delete("/v1/brains/alpha")
            assert no_confirm.status_code != 204

            confirmed = client.delete(
                "/v1/brains/alpha",
                headers={"X-Confirm-Delete": "yes"},
            )
            assert confirmed.status_code == 204


# -- 3. Document CRUD -------------------------------------------------------


def test_serve_document_crud(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "docs")

            put = client.put(
                "/v1/brains/docs/documents",
                params={"path": "memory/global/a.md"},
                content=b"hello",
                headers={"Content-Type": "application/octet-stream"},
            )
            assert put.status_code == 204

            read = client.get(
                "/v1/brains/docs/documents/read",
                params={"path": "memory/global/a.md"},
            )
            assert read.status_code == 200
            assert read.content == b"hello"

            head = client.head(
                "/v1/brains/docs/documents",
                params={"path": "memory/global/a.md"},
            )
            assert head.status_code == 200

            listed = client.get(
                "/v1/brains/docs/documents",
                params={"dir": "memory/global", "recursive": "true"},
            )
            assert listed.status_code == 200
            assert "memory/global/a.md" in listed.text

            batch_body = {
                "reason": "test",
                "ops": [
                    {
                        "type": "write",
                        "path": "memory/global/b.md",
                        "content_base64": base64.b64encode(b"world").decode(),
                    },
                    {"type": "delete", "path": "memory/global/a.md"},
                ],
            }
            batch = client.post(
                "/v1/brains/docs/documents/batch-ops", json=batch_body
            )
            assert batch.status_code == 200, batch.text

            deleted = client.delete(
                "/v1/brains/docs/documents",
                params={"path": "memory/global/b.md"},
            )
            assert deleted.status_code == 204


# -- 4. Ingest + search -----------------------------------------------------


def test_serve_ingest_search(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "ingest")
            ingest_body = {
                "path": "test.md",
                "contentType": "text/markdown",
                "contentBase64": base64.b64encode(
                    b"# hedgehog\n\nThe hedgehog lives in hedgerows."
                ).decode(),
            }
            ingest = client.post(
                "/v1/brains/ingest/ingest/file", json=ingest_body
            )
            assert ingest.status_code == 200, ingest.text

            search = client.post(
                "/v1/brains/ingest/search",
                json={"query": "hedgehog", "topK": 5, "mode": "auto"},
            )
            assert search.status_code == 200
            chunks = search.json().get("chunks", [])
            assert chunks, "expected at least one search hit"


# -- 5. Ask SSE -------------------------------------------------------------


def test_serve_ask_sse(tmp_path) -> None:
    # Inject a deterministic Fake provider so the streamed deltas are
    # reproducible.
    async def _build_daemon() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["The hedgehog lives in hedgerows."]),
        )

    daemon = asyncio.run(_build_daemon())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "asksse")
            ingest_body = {
                "path": "test.md",
                "contentType": "text/markdown",
                "contentBase64": base64.b64encode(
                    b"# hedgehog\n\nThe hedgehog lives in hedgerows."
                ).decode(),
            }
            resp = client.post(
                "/v1/brains/asksse/ingest/file", json=ingest_body
            )
            assert resp.status_code == 200, resp.text

            events: dict[str, str] = {}
            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/asksse/ask",
                json={"question": "where does the hedgehog live", "topK": 1},
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                deadline = time.monotonic() + 5.0
                event_name = ""
                data_buf: list[str] = []
                for line in stream.iter_lines():
                    if time.monotonic() > deadline:
                        break
                    if line == "":
                        if event_name:
                            events.setdefault(event_name, "\n".join(data_buf))
                        event_name = ""
                        data_buf = []
                        if "done" in events:
                            break
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line[len("data:"):].lstrip())

            for expected_event in ("retrieve", "answer_delta", "done"):
                assert expected_event in events, f"missing {expected_event!r} in {events!r}"
    asyncio.run(daemon.close())


# -- 6. Events SSE ----------------------------------------------------------


def test_serve_events_sse(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "evt")

            change_event = threading.Event()
            subscriber_events: list[str] = []
            stop_flag = threading.Event()

            def subscribe() -> None:
                try:
                    with httpx.stream(
                        "GET",
                        f"{base_url}/v1/brains/evt/events",
                        headers={"Accept": "text/event-stream"},
                        timeout=httpx.Timeout(5.0),
                    ) as resp:
                        event_name = ""
                        for line in resp.iter_lines():
                            if stop_flag.is_set():
                                return
                            if line == "":
                                if event_name:
                                    subscriber_events.append(event_name)
                                    if event_name == "change":
                                        change_event.set()
                                        return
                                event_name = ""
                                continue
                            if line.startswith("event:"):
                                event_name = line[len("event:"):].strip()
                except Exception:
                    pass

            thread = threading.Thread(target=subscribe, daemon=True)
            thread.start()
            # Give the server a moment to subscribe.
            time.sleep(0.2)

            put = client.put(
                "/v1/brains/evt/documents",
                params={"path": "evt.md"},
                content=b"hi",
                headers={"Content-Type": "application/octet-stream"},
            )
            assert put.status_code == 204

            received = change_event.wait(timeout=5.0)
            stop_flag.set()
            thread.join(timeout=2.0)
            assert received, f"change event never arrived. events={subscriber_events}"


# -- 7. Problem+JSON errors (404 / 413) ------------------------------------


def test_serve_problem_json_errors(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            missing = client.get(
                "/v1/brains/missing/documents/read",
                params={"path": "a.md"},
            )
            assert missing.status_code == 404
            assert "not_found" in missing.text

            _must_create_brain(client, "size")
            huge = b"x" * (3 * 1024 * 1024)
            oversized = client.put(
                "/v1/brains/size/documents",
                params={"path": "memory/global/big.md"},
                content=huge,
                headers={"Content-Type": "application/octet-stream"},
            )
            assert oversized.status_code == 413


# -- 8. Auth (rejected + accepted) -----------------------------------------


def test_serve_auth_token(tmp_path) -> None:
    async def _build_daemon() -> Daemon:
        return await Daemon.create(root=tmp_path, auth_token="secret")

    daemon = asyncio.run(_build_daemon())
    app = create_app(daemon=daemon, auth_token="secret")
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # /healthz exempt
            resp = client.get("/healthz")
            assert resp.status_code == 200

            # Unauth request rejected
            anon = client.get("/v1/brains")
            assert anon.status_code == 401

            # Valid bearer token passes through
            ok = client.get(
                "/v1/brains",
                headers={"Authorization": "Bearer secret"},
            )
            assert ok.status_code == 200
    asyncio.run(daemon.close())


# -- 9. Version / smoke -----------------------------------------------------


def test_serve_version_smoke(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # Healthz still works when version endpoint is exercised.
            assert client.get("/healthz").status_code == 200
            version = client.get("/version")
            assert version.status_code == 200
            body = version.json()
            assert isinstance(body, dict)
            assert "version" in body

# SPDX-License-Identifier: Apache-2.0
"""Tests exercising the real (non-stub) wiring of the memory serve
handlers now that the sibling packages are ported.

Each case runs the full HTTP daemon via uvicorn so the behaviour
matches what a remote client sees. The LLM provider is deterministic
(``FakeProvider``) and the embedder is left unset so the retriever
falls back to BM25 over the SQLite index.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
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


def _fake_daemon_factory(tmp_path, responses: list[str] | None = None):  # type: ignore[no-untyped-def]
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(responses or ["ok"]),
        )

    return build


# -- 1. Remember + recall round-trip ---------------------------------------


def test_serve_remember_recall(tmp_path) -> None:
    """Writing via /remember lands on disk and /recall surfaces it via
    the real :class:`MemoryManager`. The Fake provider returns a JSON
    selector payload so recall's LLM pick is deterministic."""
    # Prime the Fake provider to pick the memory file we write below.
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider([json.dumps({"selected": ["hedgehog.md"]})]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "rem")

            saved = client.post(
                "/v1/brains/rem/remember",
                json={
                    "note": "Hedgehogs live in hedgerows.",
                    "slug": "hedgehog",
                    "scope": "global",
                    "tags": ["fact", "wildlife"],
                },
            )
            assert saved.status_code == 201, saved.text
            saved_body = saved.json()
            assert saved_body["path"].endswith("hedgehog.md")
            assert saved_body["slug"] == "hedgehog"

            recalled = client.post(
                "/v1/brains/rem/recall",
                json={"query": "Where do hedgehogs live?", "topK": 5},
            )
            assert recalled.status_code == 200, recalled.text
            memories = recalled.json().get("memories", [])
            assert memories, "expected at least one recalled memory"
            assert memories[0]["path"].endswith("hedgehog.md")
            assert "hedgerows" in memories[0]["content"]
    asyncio.run(daemon.close())


# -- 2. Search after ingest returns the ingested path ----------------------


def test_serve_search_real(tmp_path) -> None:
    app = create_app(root=tmp_path)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "search")
            ingest = client.post(
                "/v1/brains/search/ingest/file",
                json={
                    "path": "badger.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(
                        b"# badger\n\nBadgers are nocturnal mustelids."
                    ).decode(),
                },
            )
            assert ingest.status_code == 200, ingest.text
            resp = ingest.json()
            assert resp.get("documentId")
            assert str(resp.get("path", "")).startswith("raw/documents/")

            search = client.post(
                "/v1/brains/search/search",
                json={"query": "badger", "topK": 5, "mode": "bm25"},
            )
            assert search.status_code == 200, search.text
            chunks = search.json().get("chunks", [])
            assert chunks, "expected a hit for 'badger'"
            paths = [c["path"] for c in chunks]
            assert any(p.startswith("raw/documents/") for p in paths)


# -- 3. Ask citations reference the retrieved chunk ------------------------


def test_serve_ask_citations(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["Badgers are nocturnal."]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "askreal")
            resp = client.post(
                "/v1/brains/askreal/ingest/file",
                json={
                    "path": "badger.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(
                        b"# badger\n\nBadgers are nocturnal mustelids."
                    ).decode(),
                },
            )
            assert resp.status_code == 200, resp.text

            events: dict[str, list[str]] = {}
            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/askreal/ask",
                json={"question": "what are badgers", "topK": 1},
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
                            events.setdefault(event_name, []).append(
                                "\n".join(data_buf)
                            )
                        event_name = ""
                        data_buf = []
                        if "done" in events:
                            break
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line[len("data:"):].lstrip())

            assert "retrieve" in events
            assert "citation" in events
            assert "done" in events
            citation_payload = json.loads(events["citation"][0])
            assert citation_payload["path"].startswith("raw/documents/")
    asyncio.run(daemon.close())


# -- 4. Extract returns structured memories --------------------------------


def test_serve_extract_returns_memories(tmp_path) -> None:
    extraction_output = json.dumps(
        {
            "memories": [
                {
                    "action": "create",
                    "filename": "user_snacks.md",
                    "name": "user snacks",
                    "description": "User prefers Sheffield HP sauce.",
                    "type": "user",
                    "scope": "global",
                    "content": "User loves HP sauce.",
                    "index_entry": "User snacks",
                }
            ]
        }
    )

    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider([extraction_output]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "extract")
            resp = client.post(
                "/v1/brains/extract/extract",
                json={
                    "project": "",
                    "messages": [
                        {"role": "user", "content": "I love HP sauce."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            memories = resp.json().get("memories", [])
            assert memories and memories[0]["filename"] == "user_snacks.md"
            assert memories[0]["scope"] == "global"
    asyncio.run(daemon.close())


# -- 5. Reflect produces either a structured result or null ----------------


def test_serve_reflect_null_when_no_outcome(tmp_path) -> None:
    # Reflection parses JSON with an "outcome" key. Feed back garbage so
    # the reflector returns None and the handler emits {"result": null}.
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["not-json"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "reflect")
            resp = client.post(
                "/v1/brains/reflect/reflect",
                json={
                    "messages": [
                        {"role": "user", "content": "please refactor foo"},
                        {"role": "assistant", "content": "done"},
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"result": None}
    asyncio.run(daemon.close())


# -- 6. Consolidate full mode returns a report shape ------------------------


def test_serve_consolidate_full(tmp_path) -> None:
    daemon = asyncio.run(_fake_daemon_factory(tmp_path)())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "cons")
            resp = client.post(
                "/v1/brains/cons/consolidate",
                json={"mode": "full"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["mode"] == "full"
            # The report is empty (no memories yet) but every field
            # is present and the right shape.
            for key in (
                "episodesReviewed",
                "memoriesMerged",
                "heuristicsUpdated",
                "indexesRebuilt",
                "staleMemoriesFlagged",
                "insightsPromoted",
                "errors",
            ):
                assert key in body, f"missing {key}: {body}"
    asyncio.run(daemon.close())


# -- 7. Consolidate quick mode handles missing body ------------------------


def test_serve_consolidate_quick_empty_body(tmp_path) -> None:
    daemon = asyncio.run(_fake_daemon_factory(tmp_path)())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "qcons")
            resp = client.post(
                "/v1/brains/qcons/consolidate",
                json={"mode": "quick"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["mode"] == "quick"
    asyncio.run(daemon.close())


# -- 8. Ingest + search path survives a brain close / reopen ---------------


def test_serve_ingest_search_warm_restart(tmp_path) -> None:
    """Create brain, ingest, close daemon, reopen — the index rebuilds
    from disk on reopen so subsequent searches still hit the prior
    ingest."""

    async def build() -> Daemon:
        return await Daemon.create(root=tmp_path)

    first = asyncio.run(build())
    app = create_app(daemon=first)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "warm")
            ingest = client.post(
                "/v1/brains/warm/ingest/file",
                json={
                    "path": "otter.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(
                        b"# otter\n\nOtters swim in rivers."
                    ).decode(),
                },
            )
            assert ingest.status_code == 200, ingest.text
    asyncio.run(first.close())

    second = asyncio.run(build())
    app2 = create_app(daemon=second)
    with _run_app(app2) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            resp = client.post(
                "/v1/brains/warm/search",
                json={"query": "otter", "topK": 5, "mode": "bm25"},
            )
            assert resp.status_code == 200, resp.text
            chunks = resp.json().get("chunks", [])
            assert chunks, "expected warm-restart index to re-surface the ingest"
    asyncio.run(second.close())

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

from jeffs_brain_memory import retrieval
from jeffs_brain_memory.http import Daemon, create_app
from jeffs_brain_memory.llm import FakeEmbedder, FakeProvider


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


def test_serve_search_filters_and_memory_scope_alias_on_fallback(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchfilters")
            for path, body in {
                "memory/global/coffee.md": (
                    b"---\nname: Coffee\n"
                    b"tags:\n- drink\n---\nAlex likes coffee.\n"
                ),
                "memory/project/brain/coffee.md": (
                    b"---\nname: Brain coffee\n"
                    b"tags:\n- drink\n---\nProject coffee budget.\n"
                ),
                "wiki/coffee.md": b"---\ntitle: Coffee\n---\nCoffee wiki.\n",
            }.items():
                resp = client.put(
                    f"/v1/brains/searchfilters/documents?path={path}",
                    content=body,
                    headers={"Content-Type": "text/markdown"},
                )
                assert resp.status_code == 204, resp.text

            br = asyncio.run(daemon.brains.get("searchfilters"))

            class _EmptyRetriever:
                async def retrieve(self, _req):
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _EmptyRetriever()  # type: ignore[assignment]

            search = client.post(
                "/v1/brains/searchfilters/search",
                json={
                    "query": "coffee",
                    "topK": 10,
                    "filters": {
                        "scope": "memory",
                        "pathPrefix": "memory/project/",
                        "tags": ["drink"],
                    },
                },
            )
            assert search.status_code == 200, search.text
            chunks = search.json().get("chunks", [])
            assert [chunk["path"] for chunk in chunks] == [
                "memory/project/brain/coffee.md"
            ]
    asyncio.run(daemon.close())


def test_serve_search_retriever_errors_fall_back_to_bm25(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchfallback")
            resp = client.put(
                "/v1/brains/searchfallback/documents?path=memory/project/brain/coffee.md",
                content=b"---\nname: Brain coffee\n---\nProject coffee budget.\n",
                headers={"Content-Type": "text/markdown"},
            )
            assert resp.status_code == 204, resp.text

            br = asyncio.run(daemon.brains.get("searchfallback"))

            class _FailingRetriever:
                async def retrieve(self, _req):
                    raise RuntimeError("boom")

            br.retriever = _FailingRetriever()  # type: ignore[assignment]

            search = client.post(
                "/v1/brains/searchfallback/search",
                json={"query": "coffee", "topK": 10},
            )
            assert search.status_code == 200, search.text
            chunks = search.json().get("chunks", [])
            assert [chunk["path"] for chunk in chunks] == [
                "memory/project/brain/coffee.md"
            ]
    asyncio.run(daemon.close())


def test_serve_search_temporal_fallback_uses_question_date(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchtemporal")
            resp = client.put(
                "/v1/brains/searchtemporal/documents?path=raw/lme/session-1.md",
                content=(
                    b"---\nsession_id: s1\nsession_date: 2024-03-08\n---\n"
                    b"The user bought apples on 2024/03/08.\n"
                ),
                headers={"Content-Type": "text/markdown"},
            )
            assert resp.status_code == 204, resp.text

            br = asyncio.run(daemon.brains.get("searchtemporal"))

            class _EmptyRetriever:
                async def retrieve(self, _req):
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _EmptyRetriever()  # type: ignore[assignment]

            search = client.post(
                "/v1/brains/searchtemporal/search",
                json={
                    "query": "What did the user buy last Friday?",
                    "questionDate": "2024/03/13 (Wed) 10:00",
                    "topK": 5,
                },
            )
            assert search.status_code == 200, search.text
            chunks = search.json().get("chunks", [])
            assert chunks, "expected temporal fallback hit"
            assert chunks[0]["path"] == "raw/lme/session-1.md"
            assert "2024/03/08" in chunks[0]["text"]
    asyncio.run(daemon.close())


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


def test_serve_search_forwards_candidate_knobs(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchknobs")
            br = asyncio.run(daemon.brains.get("searchknobs"))
            captured: dict[str, retrieval.Request] = {}

            class _CapturingRetriever:
                async def retrieve(self, req: retrieval.Request) -> retrieval.Response:
                    captured["req"] = req
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _CapturingRetriever()  # type: ignore[assignment]

            resp = client.post(
                "/v1/brains/searchknobs/search",
                json={
                    "query": "apples",
                    "topK": 3,
                    "candidateK": 80,
                    "rerankTopN": 40,
                    "mode": "hybrid-rerank",
                },
            )
            assert resp.status_code == 200, resp.text
            assert captured["req"].candidate_k == 80
            assert captured["req"].rerank_top_n == 40
    asyncio.run(daemon.close())


def test_serve_search_forwards_exact_path_filters(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchpaths")
            br = asyncio.run(daemon.brains.get("searchpaths"))
            captured: dict[str, retrieval.Request] = {}

            class _CapturingRetriever:
                async def retrieve(self, req: retrieval.Request) -> retrieval.Response:
                    captured["req"] = req
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _CapturingRetriever()  # type: ignore[assignment]

            resp = client.post(
                "/v1/brains/searchpaths/search",
                json={
                    "query": "apples",
                    "topK": 3,
                    "filters": {
                        "paths": [
                            "raw/documents/allowed.md",
                            "raw/documents/allowed.md",
                            " raw/documents/other.md ",
                        ]
                    },
                },
            )
            assert resp.status_code == 200, resp.text
            assert captured["req"].filters.paths == [
                "raw/documents/allowed.md",
                "raw/documents/other.md",
            ]
    asyncio.run(daemon.close())


def test_serve_search_fallback_respects_exact_path_filters(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "searchpathfallback")
            for path, body in {
                "raw/documents/allowed.md": b"# allowed\n\nTarget note about apples.\n",
                "raw/documents/blocked.md": b"# blocked\n\nAnother note about apples.\n",
            }.items():
                resp = client.put(
                    f"/v1/brains/searchpathfallback/documents?path={path}",
                    content=body,
                    headers={"Content-Type": "text/markdown"},
                )
                assert resp.status_code == 204, resp.text

            br = asyncio.run(daemon.brains.get("searchpathfallback"))

            class _EmptyRetriever:
                async def retrieve(self, _req):
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _EmptyRetriever()  # type: ignore[assignment]

            resp = client.post(
                "/v1/brains/searchpathfallback/search",
                json={
                    "query": "apples",
                    "topK": 10,
                    "filters": {"paths": ["raw/documents/allowed.md"]},
                },
            )
            assert resp.status_code == 200, resp.text
            chunks = resp.json().get("chunks", [])
            assert [chunk["path"] for chunk in chunks] == ["raw/documents/allowed.md"]
    asyncio.run(daemon.close())


def test_serve_ask_forwards_candidate_knobs(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "askknobs")
            br = asyncio.run(daemon.brains.get("askknobs"))
            captured: dict[str, retrieval.Request] = {}

            class _CapturingRetriever:
                async def retrieve(self, req: retrieval.Request) -> retrieval.Response:
                    captured["req"] = req
                    return retrieval.Response(
                        chunks=[
                            retrieval.RetrievedChunk(
                                chunk_id="chunk-1",
                                document_id="doc-1",
                                path="raw/documents/note.md",
                                score=1.0,
                                text="A note about apples.",
                                title="note",
                                summary="",
                            )
                        ],
                        trace=retrieval.Trace(),
                    )

            br.retriever = _CapturingRetriever()  # type: ignore[assignment]

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/askknobs/ask",
                json={
                    "question": "what about apples",
                    "topK": 3,
                    "candidateK": 80,
                    "rerankTopN": 40,
                    "mode": "hybrid-rerank",
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                assert stream.status_code == 200
                for line in stream.iter_lines():
                    if line == "" or line == "data: {\"ok\": true}":
                        continue
                    if line.startswith("event: done"):
                        break

            assert captured["req"].candidate_k == 80
            assert captured["req"].rerank_top_n == 40
    asyncio.run(daemon.close())


def test_serve_ask_forwards_exact_path_filters(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "askpaths")
            br = asyncio.run(daemon.brains.get("askpaths"))
            captured: dict[str, retrieval.Request] = {}

            class _CapturingRetriever:
                async def retrieve(self, req: retrieval.Request) -> retrieval.Response:
                    captured["req"] = req
                    return retrieval.Response(
                        chunks=[
                            retrieval.RetrievedChunk(
                                chunk_id="chunk-1",
                                document_id="doc-1",
                                path="raw/documents/allowed.md",
                                score=1.0,
                                text="A note about apples.",
                                title="note",
                                summary="",
                            )
                        ],
                        trace=retrieval.Trace(),
                    )

            br.retriever = _CapturingRetriever()  # type: ignore[assignment]

            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/askpaths/ask",
                json={
                    "question": "what about apples",
                    "topK": 3,
                    "filters": {
                        "paths": [
                            "raw/documents/allowed.md",
                            " raw/documents/other.md ",
                        ]
                    },
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                assert stream.status_code == 200
                for line in stream.iter_lines():
                    if line.startswith("event: done"):
                        break

            assert captured["req"].filters.paths == [
                "raw/documents/allowed.md",
                "raw/documents/other.md",
            ]
    asyncio.run(daemon.close())


def test_serve_ask_fallback_respects_exact_path_filters(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["Badgers are nocturnal."]),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "askpathfallback")
            for path, body in {
                "raw/documents/allowed.md": b"# allowed\n\nA note about apples.\n",
                "raw/documents/blocked.md": b"# blocked\n\nAnother note about apples.\n",
            }.items():
                resp = client.put(
                    f"/v1/brains/askpathfallback/documents?path={path}",
                    content=body,
                    headers={"Content-Type": "text/markdown"},
                )
                assert resp.status_code == 204, resp.text

            br = asyncio.run(daemon.brains.get("askpathfallback"))

            class _EmptyRetriever:
                async def retrieve(self, _req):
                    return retrieval.Response(trace=retrieval.Trace())

            br.retriever = _EmptyRetriever()  # type: ignore[assignment]

            events: dict[str, list[str]] = {}
            with httpx.stream(
                "POST",
                f"{base_url}/v1/brains/askpathfallback/ask",
                json={
                    "question": "what about apples",
                    "topK": 10,
                    "filters": {"paths": ["raw/documents/allowed.md"]},
                },
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(5.0),
            ) as stream:
                assert stream.status_code == 200
                event_name = ""
                data_buf: list[str] = []
                for line in stream.iter_lines():
                    if line == "":
                        if event_name:
                            events.setdefault(event_name, []).append(
                                "\n".join(data_buf)
                            )
                        if event_name == "done":
                            break
                        event_name = ""
                        data_buf = []
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line[len("data:"):].lstrip())

            assert "retrieve" in events
            payload = json.loads(events["retrieve"][0])
            assert [chunk["path"] for chunk in payload["chunks"]] == [
                "raw/documents/allowed.md"
            ]
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


def test_serve_extract_applies_contextual_prefix_and_session_fields(tmp_path) -> None:
    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(
                [
                    json.dumps(
                        {
                            "memories": [
                                {
                                    "action": "create",
                                    "filename": "bike_status.md",
                                    "name": "Bike status",
                                    "description": "Latest bike colour",
                                    "type": "project",
                                    "scope": "project",
                                    "content": "The bike is blue now.",
                                    "index_entry": "Bike status",
                                }
                            ]
                        }
                    ),
                    "The session was a bike-status update in February, and this fact records the latest confirmed colour.",
                ]
            ),
            contextualise=True,
            contextualise_cache_dir=str(tmp_path / "ctx-cache"),
        )

    daemon = asyncio.run(build())
    app = create_app(daemon=daemon)
    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            _must_create_brain(client, "extractctx")
            resp = client.post(
                "/v1/brains/extractctx/extract",
                json={
                    "project": "",
                    "sessionId": "session-42",
                    "sessionDate": "2024/02/20 (Tue) 09:15",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Session on 2024-02-20 about the bike status.",
                        },
                        {"role": "user", "content": "The bike is blue now."},
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            memories = resp.json().get("memories", [])
            assert memories
            assert memories[0]["sessionId"] == "session-42"
            assert memories[0]["sessionDate"] == "2024-02-20"
            assert "bike-status update" in memories[0]["contextPrefix"]
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


def test_serve_vector_backfill_populates_embeddings(tmp_path) -> None:
    """The daemon kicks off a vector backfill after the initial FTS
    scan; /search in hybrid mode should surface vector hits once the
    backfill lands.

    Mirrors the Go reference ``daemon_vectors.go``: the backfill runs
    once after the initial scan, so the brain cache is pre-seeded on
    disk rather than ingested post-open.

    Uses :class:`FakeEmbedder` so no network is touched and the ranking
    is deterministic (unit-vector outputs seeded by SHA-256 of each
    input). The search handler is driven in ``hybrid`` mode so the
    vector leg is exercised regardless of the auto-resolve heuristics.
    """

    # Pre-seed the brain on disk so the initial scan has something to
    # feed into the backfill. The daemon builds lazily on the first
    # request.
    seeded = tmp_path / "brains" / "vectors" / "memory" / "global" / "owl.md"
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_text(
        "---\n"
        "name: Owl\n"
        "description: Nocturnal raptor.\n"
        "---\n"
        "\n"
        "Owls hunt at night in the forest.\n",
        encoding="utf-8",
    )

    async def build() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
            embedder=FakeEmbedder(16),
        )

    daemon = asyncio.run(build())
    # Pin the model so the backfill stamps rows with a known value.
    daemon.embed_model = "fake-embed-16"
    app = create_app(daemon=daemon)

    with _run_app(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            # First search request triggers lazy brain open + scan +
            # backfill task. The pre-seeded document is already on disk
            # so the backfill populates knowledge_embeddings before the
            # next poll cycle.
            warmup = client.post(
                "/v1/brains/vectors/search",
                json={"query": "owl", "topK": 3, "mode": "bm25"},
            )
            assert warmup.status_code == 200, warmup.text

            # Probe the SQLite file directly: the search.Index
            # connection is thread-bound and lives on the uvicorn
            # worker, so reading vector_count from this thread would
            # raise sqlite3.ProgrammingError. WAL mode lets an
            # independent reader co-exist with the backfill writer.
            import sqlite3 as _sqlite3

            db_path = tmp_path / "indices" / "vectors" / "search.sqlite"

            def _vector_rows() -> int:
                if not db_path.exists():
                    return 0
                conn = _sqlite3.connect(str(db_path))
                try:
                    row = conn.execute(
                        "SELECT count(*) FROM knowledge_embeddings "
                        "WHERE model = ?",
                        (daemon.embed_model,),
                    ).fetchone()
                    return int((row and row[0]) or 0)
                finally:
                    conn.close()

            deadline = time.monotonic() + 5.0
            vector_count = 0
            while time.monotonic() < deadline:
                vector_count = _vector_rows()
                if vector_count > 0:
                    break
                time.sleep(0.05)

            assert vector_count > 0, "expected backfill to persist vectors"

            resp = client.post(
                "/v1/brains/vectors/search",
                json={"query": "owl", "topK": 5, "mode": "hybrid"},
            )
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            chunks = payload.get("chunks", [])
            assert chunks, "hybrid search returned no chunks"
            trace = payload.get("trace") or {}
            assert trace.get("embedder_used") is True, (
                f"expected embedder to have run, trace={trace}"
            )
            assert trace.get("vector_hits", 0) > 0, (
                f"expected vector leg to return hits, trace={trace}"
            )
    asyncio.run(daemon.close())


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

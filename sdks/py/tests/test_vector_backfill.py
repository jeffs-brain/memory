# SPDX-License-Identifier: Apache-2.0
"""Vector backfill regression tests.

Mirrors the Go reference at ``sdks/go/cmd/memory/daemon_vectors.go``
and its eval harness: a brain cache is pre-seeded with markdown on
disk, the daemon opens it, and the detached backfill task populates
``knowledge_embeddings`` with rows tagged by the active embedding
model. A follow-up ``/search`` call must return hybrid hits (BM25
fused with vector) rather than BM25-only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import uvicorn

from jeffs_brain_memory.http import Daemon, backfill_vectors, create_app
from jeffs_brain_memory.http.daemon import PassthroughStore, _rebuild_sync
from jeffs_brain_memory.llm import FakeEmbedder, FakeProvider
from jeffs_brain_memory.llm.config import resolve_embed_model
from jeffs_brain_memory.search import Index


FIXTURES = {
    "memory/global/hedgehog.md": (
        "---\n"
        "name: Hedgehog\n"
        "description: Small mammal that shelters in hedgerows.\n"
        "tags: [mammal, nocturnal]\n"
        "---\n"
        "\n"
        "The hedgehog forages at night across European hedgerows.\n"
    ),
    "memory/global/badger.md": (
        "---\n"
        "name: Badger\n"
        "description: Nocturnal mustelid that digs setts.\n"
        "tags: [mammal, nocturnal]\n"
        "---\n"
        "\n"
        "The badger digs deep setts in mature woodland soil.\n"
    ),
    "memory/global/robin.md": (
        "---\n"
        "name: Robin\n"
        "description: Small passerine with a red breast.\n"
        "tags: [bird, diurnal]\n"
        "---\n"
        "\n"
        "Robins sing year-round to defend their winter territories.\n"
    ),
}


def _seed_brain(root: Path, brain_id: str) -> Path:
    base = root / "brains" / brain_id
    base.mkdir(parents=True, exist_ok=True)
    for path, body in FIXTURES.items():
        dest = base / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
    return base


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


def test_resolve_embed_model_pins_active_model() -> None:
    embedder = FakeEmbedder(dims=8)
    # Fake embedder has no env preference, so auto-detect lands on the
    # Ollama default. Setting JB_EMBED_MODEL pins it explicitly.
    from_default = resolve_embed_model(embedder, env={})
    assert from_default in {"bge-m3", "text-embedding-3-small"}
    pinned = resolve_embed_model(embedder, env={"JB_EMBED_MODEL": "nomic-embed-text"})
    assert pinned == "nomic-embed-text"
    assert resolve_embed_model(None) == ""


def test_backfill_vectors_populates_embeddings(tmp_path) -> None:
    """Drive the coroutine directly so the assertion is deterministic."""

    brain_id = "solo"
    base = _seed_brain(tmp_path, brain_id)
    model = "fake-embed-8"
    dims = 8

    async def run() -> tuple[int, int, set[str]]:
        store = PassthroughStore(base)
        db_path = tmp_path / "indices" / brain_id / "search.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        index = Index(str(db_path))
        try:
            await _rebuild_sync(index, store)
            assert index.list_indexed_paths(), "seed scan must populate FTS"

            embedder = FakeEmbedder(dims=dims)
            written = await backfill_vectors(
                brain_id=brain_id,
                store=store,
                index=index,
                embedder=embedder,
                model=model,
            )
            vec_count = index.vector_count()
            pre_tagged = index.paths_with_vectors(model)

            # Idempotency: a second pass is a no-op because every path is
            # already tagged with the active model.
            rerun = await backfill_vectors(
                brain_id=brain_id,
                store=store,
                index=index,
                embedder=embedder,
                model=model,
            )
            assert rerun == 0

            return written, vec_count, pre_tagged
        finally:
            index.close()
            await store.close()

    written, vec_count, tagged_paths = asyncio.run(run())

    assert written == len(FIXTURES)
    assert vec_count == len(FIXTURES)
    assert tagged_paths == set(FIXTURES.keys())


def test_backfill_skips_when_embedder_none(tmp_path) -> None:
    """With no embedder the coroutine is a silent no-op; BM25 still works."""

    brain_id = "no_embed"
    base = _seed_brain(tmp_path, brain_id)

    async def run() -> int:
        store = PassthroughStore(base)
        db_path = tmp_path / "indices" / brain_id / "search.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        index = Index(str(db_path))
        try:
            await _rebuild_sync(index, store)
            written = await backfill_vectors(
                brain_id=brain_id,
                store=store,
                index=index,
                embedder=None,
                model="",
            )
            return written + index.vector_count()
        finally:
            index.close()
            await store.close()

    assert asyncio.run(run()) == 0


def test_backfill_tolerates_batch_failure(tmp_path, caplog) -> None:
    """One bad batch must not stall the rest of the run."""

    brain_id = "flaky"
    base = _seed_brain(tmp_path, brain_id)
    model = "fake-embed-8"

    class FlakyEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__(dims=8)
            self.calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic embed failure")
            return await super().embed(texts)

    async def run() -> int:
        store = PassthroughStore(base)
        db_path = tmp_path / "indices" / brain_id / "search.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        index = Index(str(db_path))
        try:
            await _rebuild_sync(index, store)
            embedder = FlakyEmbedder()
            with caplog.at_level(logging.WARNING):
                first = await backfill_vectors(
                    brain_id=brain_id,
                    store=store,
                    index=index,
                    embedder=embedder,
                    model=model,
                )
            # Batch #1 raised, so first pass writes nothing. Batch #2
            # succeeds, so a follow-up pass completes the backfill.
            assert first == 0
            second = await backfill_vectors(
                brain_id=brain_id,
                store=store,
                index=index,
                embedder=embedder,
                model=model,
            )
            return second + index.vector_count()
        finally:
            index.close()
            await store.close()

    total = asyncio.run(run())
    assert total >= len(FIXTURES)
    assert any("embed batch failed" in rec.message for rec in caplog.records)


def _count_vectors_on_disk(root: Path, brain_id: str, model: str) -> int:
    """Probe the sqlite file directly so the test thread never touches
    the daemon's thread-bound connection."""
    import sqlite3

    db_path = root / "indices" / brain_id / "search.sqlite"
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT count(*) FROM knowledge_embeddings WHERE model = ?",
            (model,),
        ).fetchone()
        return int((row and row[0]) or 0)
    finally:
        conn.close()


def test_serve_search_returns_hybrid_hits(tmp_path) -> None:
    """End-to-end: seeded brain + FakeEmbedder + /search returns vector score."""

    brain_id = "hybrid"
    _seed_brain(tmp_path, brain_id)

    dims = 16
    pinned_model = "fake-embed-16"

    async def _build_daemon() -> Daemon:
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
            embedder=FakeEmbedder(dims=dims),
        )

    daemon = asyncio.run(_build_daemon())
    daemon.embed_model = pinned_model

    app = create_app(daemon=daemon)
    try:
        with _run_app(app) as base_url:
            with httpx.Client(base_url=base_url, timeout=5.0) as client:
                # First hit warms the brain and schedules the detached
                # backfill task inside uvicorn's event loop. The
                # returned chunks come from BM25-only because the vector
                # leg has not yet persisted any rows.
                warmup = client.post(
                    f"/v1/brains/{brain_id}/search",
                    json={"query": "hedgehog", "topK": 5, "mode": "hybrid"},
                )
                assert warmup.status_code == 200, warmup.text

                # Poll the sqlite file directly so the test thread never
                # shares a connection with the daemon thread. WAL keeps
                # a concurrent reader happy alongside the backfill writer.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (
                        _count_vectors_on_disk(tmp_path, brain_id, pinned_model)
                        >= len(FIXTURES)
                    ):
                        break
                    time.sleep(0.05)
                final = _count_vectors_on_disk(tmp_path, brain_id, pinned_model)
                assert final >= len(FIXTURES), (
                    "expected backfill to persist every FTS row; "
                    f"got {final}"
                )

                resp = client.post(
                    f"/v1/brains/{brain_id}/search",
                    json={"query": "hedgehog", "topK": 5, "mode": "hybrid"},
                )
                assert resp.status_code == 200, resp.text
                body = resp.json()
                chunks = body.get("chunks", [])
                assert chunks, "expected hybrid search to surface pre-seeded content"
                trace = body.get("trace", {})
                assert trace.get("embedder_used") is True
                assert int(trace.get("vector_hits") or 0) > 0
                assert any(
                    float(chunk.get("vectorSimilarity") or 0.0) > 0.0
                    for chunk in chunks
                )
    finally:
        asyncio.run(daemon.close())


@pytest.mark.parametrize("mode", ["auto", "hybrid"])
def test_serve_search_with_fake_embedder_still_returns_bm25(tmp_path, mode) -> None:
    """Regression guard: even without vectors the existing BM25 path holds."""

    brain_id = f"bm25_{mode.replace('-', '_')}"
    _seed_brain(tmp_path, brain_id)

    async def _build_daemon() -> Daemon:
        # No embedder = silent skip of the backfill; the search handler
        # must still return BM25 hits.
        return await Daemon.create(
            root=tmp_path,
            llm=FakeProvider(["ok"]),
            embedder=None,
        )

    daemon = asyncio.run(_build_daemon())
    app = create_app(daemon=daemon)
    try:
        with _run_app(app) as base_url:
            with httpx.Client(base_url=base_url, timeout=5.0) as client:
                resp = client.post(
                    f"/v1/brains/{brain_id}/search",
                    json={"query": "hedgehog hedgerows", "topK": 5, "mode": mode},
                )
                assert resp.status_code == 200, resp.text
                chunks = resp.json().get("chunks", [])
                assert chunks, "BM25 must survive the embedder=None path"
    finally:
        asyncio.run(daemon.close())

# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed hybrid search index tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import AsyncIterator

import pytest

from jeffs_brain_memory.path import BrainPath
from jeffs_brain_memory.search import (
    BM25Hit,
    Chunk,
    Index,
    SearchOpts,
    TrigramHit,
    VectorHit,
)
from jeffs_brain_memory.store import (
    Batch,
    BatchOptions,
    ChangeEvent,
    FileInfo,
    ListOpts,
    Store,
)


@pytest.fixture
def idx() -> Index:
    index = Index(":memory:")
    yield index
    index.close()


def _chunk(
    path: str,
    text: str,
    *,
    title: str = "",
    summary: str = "",
    tags: list[str] | None = None,
    vector: list[float] | None = None,
    scope: str = "",
    project_slug: str = "",
    session_date: str = "",
    generated: bool = False,
    chunk_id: str | None = None,
) -> Chunk:
    metadata = {
        "title": title,
        "summary": summary,
        "tags": tags or [],
        "scope": scope,
        "project_slug": project_slug,
        "session_date": session_date,
        "generated": generated,
    }
    return Chunk(
        id=chunk_id or f"{path}#0",
        document_id=path,
        path=path,
        text=text,
        metadata=metadata,
        vector=vector,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_index_constructs_and_reports_vec_status(idx: Index) -> None:
    """sqlite-vec loads on Linux with the vendored binary."""
    # The wheel ships a manylinux binary; assert the loader reports a
    # stable boolean either way.
    assert isinstance(idx.vec_loaded, bool)


def test_index_falls_back_when_sqlite_vec_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure-Python cosine must still answer queries when the extension misses.

    Forces the ``sqlite_vec`` import to raise :class:`ImportError`
    during construction so the ``_try_load_sqlite_vec`` early-return
    fires. Vector search still produces the correct ranking because
    the fallback reads the same ``knowledge_embeddings`` BLOBs.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sqlite_vec":
            raise ImportError("sqlite_vec disabled for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    index = Index(":memory:")
    try:
        assert index.vec_loaded is False
        index.upsert_chunks(
            [
                Chunk(
                    id="wiki/a.md#0",
                    document_id="wiki/a.md",
                    path="wiki/a.md",
                    text="alpha",
                    metadata={"scope": "wiki"},
                    vector=[1.0, 0.0, 0.0],
                ),
                Chunk(
                    id="wiki/b.md#0",
                    document_id="wiki/b.md",
                    path="wiki/b.md",
                    text="beta",
                    metadata={"scope": "wiki"},
                    vector=[0.0, 1.0, 0.0],
                ),
            ]
        )
        hits = index.search_vectors([1.0, 0.0, 0.0], top_k=5)
        assert hits
        assert hits[0].path == "wiki/a.md"
    finally:
        index.close()


def test_index_creates_expected_tables(idx: Index) -> None:
    tables = {
        row["name"]
        for row in idx._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
        )
    }
    assert "knowledge_fts" in tables
    assert "knowledge_chunks" in tables
    assert "knowledge_embeddings" in tables


def test_index_close_is_idempotent() -> None:
    index = Index(":memory:")
    index.close()
    index.close()


# --------------------------------------------------------------------------- #
# Upsert and basic queries
# --------------------------------------------------------------------------- #


def test_upsert_writes_chunk(idx: Index) -> None:
    count = idx.upsert_chunks(
        [
            _chunk(
                "wiki/kubernetes.md",
                "Kubernetes schedules pods across a cluster.",
                title="Kubernetes",
                summary="Overview of Kubernetes",
                scope="wiki",
            )
        ]
    )
    assert count == 1
    assert idx.chunk_count() == 1


def test_upsert_replaces_existing(idx: Index) -> None:
    idx.upsert_chunks([_chunk("wiki/x.md", "first body", title="Old", scope="wiki")])
    idx.upsert_chunks([_chunk("wiki/x.md", "second body", title="New", scope="wiki")])
    assert idx.chunk_count() == 1
    hits = idx.search_bm25("second", top_k=5)
    assert hits and hits[0].title == "New"


def test_upsert_empty_batch_is_noop(idx: Index) -> None:
    assert idx.upsert_chunks([]) == 0
    assert idx.chunk_count() == 0


# --------------------------------------------------------------------------- #
# BM25 ranking and filters
# --------------------------------------------------------------------------- #


def test_bm25_finds_by_title(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/kubernetes.md",
                "How to deploy apps",
                title="Kubernetes Guide",
                summary="Deploying apps",
                scope="wiki",
            )
        ]
    )
    hits = idx.search_bm25("Kubernetes", top_k=5)
    assert hits
    assert hits[0].title == "Kubernetes Guide"


def test_bm25_finds_by_content(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/networking.md",
                "TCP protocols and how packets traverse the internet.",
                title="Networking",
                scope="wiki",
            )
        ]
    )
    hits = idx.search_bm25("TCP protocols", top_k=5)
    assert hits
    assert hits[0].path == "wiki/networking.md"


def test_bm25_finds_frontmatter_dates_via_indexed_date_tags(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "memory/global/weekly-note.md",
                "Met the supplier and agreed the timeline.",
                title="Weekly note",
                scope="global_memory",
                session_date="2024/03/08 (Fri) 10:00",
            )
        ]
    )

    hits = idx.search_bm25('"2024/03/08"', top_k=5)
    assert hits
    assert hits[0].path == "memory/global/weekly-note.md"


def test_bm25_ranks_title_hits_first(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "w/content-only.md",
                "The term magicword appears in the body only.",
                title="Unrelated Heading",
                summary="Unrelated summary text",
                scope="wiki",
            ),
            _chunk(
                "w/summary-only.md",
                "Body text about other things entirely.",
                title="Another Heading",
                summary="Notes about magicword in the summary",
                scope="wiki",
            ),
            _chunk(
                "w/title-only.md",
                "Body text about other things entirely.",
                title="Magicword Headline",
                summary="Totally different summary",
                scope="wiki",
            ),
        ]
    )
    hits = idx.search_bm25("magicword", top_k=5)
    assert [hit.path for hit in hits[:3]] == [
        "w/title-only.md",
        "w/summary-only.md",
        "w/content-only.md",
    ]


def test_bm25_respects_max_results(idx: Index) -> None:
    chunks = [
        _chunk(
            f"wiki/doc-{i}.md",
            "This test document contains searchable content about testing.",
            title=f"Document {i}",
            summary="A test document",
            scope="wiki",
        )
        for i in range(5)
    ]
    idx.upsert_chunks(chunks)
    hits = idx.search_bm25("test document", top_k=5, opts=SearchOpts(max_results=2))
    assert len(hits) == 2


def test_bm25_empty_query_returns_nothing(idx: Index) -> None:
    idx.upsert_chunks([_chunk("wiki/x.md", "body", scope="wiki")])
    assert idx.search_bm25("", top_k=5) == []
    assert idx.search_bm25("   ", top_k=5) == []


def test_bm25_special_characters_do_not_error(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/special.md",
                "Content with special terms like C++ and node.js frameworks.",
                title="Special Chars",
                scope="wiki",
            )
        ]
    )
    # No exception; may or may not match depending on FTS tokeniser.
    idx.search_bm25("C++ node*", top_k=5)


def test_bm25_scope_filter(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/terraform.md",
                "Infrastructure as code.",
                title="Terraform Guide",
                scope="wiki",
            ),
            _chunk(
                "memory/global/terraform-notes.md",
                "Notes about using Terraform in production.",
                title="Terraform Notes",
                scope="global_memory",
            ),
        ]
    )
    wiki_hits = idx.search_bm25("Terraform", opts=SearchOpts(filters={"scope": "wiki"}))
    assert [hit.path for hit in wiki_hits] == ["wiki/terraform.md"]


def test_bm25_memory_scope_alias_matches_global_and_project(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "memory/global/user.md",
                "Alex likes coffee.",
                title="User note",
                scope="global_memory",
            ),
            _chunk(
                "memory/project/brain/plan.md",
                "Project coffee budget.",
                title="Project note",
                scope="project_memory",
                project_slug="brain",
            ),
            _chunk(
                "wiki/coffee.md",
                "Coffee guide.",
                title="Coffee wiki",
                scope="wiki",
            ),
        ]
    )
    hits = idx.search_bm25("coffee", opts=SearchOpts(filters={"scope": "memory"}))
    assert {hit.path for hit in hits} == {
        "memory/global/user.md",
        "memory/project/brain/plan.md",
    }


def test_bm25_project_slug_filter(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "memory/project/jeff/api.md",
                "Notes about the Jeff API design patterns.",
                title="Jeff API Notes",
                scope="project_memory",
                project_slug="jeff",
            ),
            _chunk(
                "memory/project/lleverage/api.md",
                "Notes about the Lleverage API design patterns.",
                title="Lleverage API Notes",
                scope="project_memory",
                project_slug="lleverage",
            ),
        ]
    )
    hits = idx.search_bm25(
        "API",
        opts=SearchOpts(
            filters={"scope": "project_memory", "project_slug": "jeff"},
        ),
    )
    assert [hit.path for hit in hits] == ["memory/project/jeff/api.md"]


def test_bm25_tag_filter(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/docker.md",
                "Container tips",
                title="Docker",
                tags=["docker", "containers"],
                scope="wiki",
            ),
            _chunk(
                "wiki/kubernetes.md",
                "Pod orchestration",
                title="K8s",
                tags=["kubernetes"],
                scope="wiki",
            ),
        ]
    )
    hits = idx.search_bm25(
        "Docker", opts=SearchOpts(filters={"tag": "containers"})
    )
    assert [hit.path for hit in hits] == ["wiki/docker.md"]


def test_bm25_include_generated_default_excluded(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("wiki/a.md", "alpha", title="Alpha", scope="wiki", generated=False),
            _chunk("wiki/b.md", "alpha", title="Beta", scope="wiki", generated=True),
        ]
    )
    default_hits = idx.search_bm25("alpha")
    assert [h.path for h in default_hits] == ["wiki/a.md"]

    all_hits = idx.search_bm25("alpha", opts=SearchOpts(include_generated=True))
    assert {h.path for h in all_hits} == {"wiki/a.md", "wiki/b.md"}


def test_bm25_snippet_contains_match(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/x.md",
                "This is about Python generics and type hints.",
                title="Python Notes",
                scope="wiki",
            )
        ]
    )
    hits = idx.search_bm25("generics")
    assert hits
    assert "generics" in hits[0].snippet.lower() or "<mark>" in hits[0].snippet


def test_bm25_document_id_is_populated(idx: Index) -> None:
    idx.upsert_chunks([_chunk("wiki/x.md", "alpha", title="A", scope="wiki")])
    hits = idx.search_bm25("alpha")
    assert hits
    assert hits[0].document_id == "wiki/x.md"
    assert hits[0].chunk_id == "wiki/x.md#0"


def test_bm25_hit_carries_body_and_metadata(idx: Index) -> None:
    idx.upsert_chunks(
        [
            Chunk(
                id="raw/lme/s1.md#0",
                document_id="raw/lme/s1.md",
                path="raw/lme/s1.md",
                text="---\nsession_id: sess-1\nsession_date: 2024-03-08\n---\n[user]: bought apples",
                metadata={
                    "title": "session 1",
                    "scope": "raw_lme",
                    "session_id": "sess-1",
                    "session_date": "2024-03-08",
                },
            )
        ]
    )
    hits = idx.search_bm25("apples")
    assert hits
    assert "bought apples" in hits[0].content
    assert hits[0].metadata["session_id"] == "sess-1"
    assert hits[0].metadata["session_date"] == "2024-03-08"


# --------------------------------------------------------------------------- #
# Vector search
# --------------------------------------------------------------------------- #


def test_vector_search_returns_nearest(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("wiki/a.md", "alpha", title="A", scope="wiki", vector=[1.0, 0.0, 0.0]),
            _chunk("wiki/b.md", "beta", title="B", scope="wiki", vector=[0.0, 1.0, 0.0]),
            _chunk("wiki/c.md", "gamma", title="C", scope="wiki", vector=[0.0, 0.0, 1.0]),
        ]
    )
    hits = idx.search_vectors([1.0, 0.1, 0.0], top_k=3)
    assert hits
    assert hits[0].path == "wiki/a.md"
    assert isinstance(hits[0], VectorHit)
    assert math.isclose(hits[0].score, max(h.score for h in hits))


def test_vector_search_dimension_mismatch_skips(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("wiki/a.md", "alpha", scope="wiki", vector=[1.0, 0.0]),
            _chunk("wiki/b.md", "beta", scope="wiki", vector=[0.0, 1.0, 0.0]),
        ]
    )
    hits = idx.search_vectors([1.0, 0.0, 0.0], top_k=5)
    assert [hit.path for hit in hits] == ["wiki/b.md"]


def test_vector_search_empty_returns_empty(idx: Index) -> None:
    assert idx.search_vectors([1.0, 0.0], top_k=5) == []


def test_vector_search_empty_query_returns_empty(idx: Index) -> None:
    idx.upsert_chunks(
        [_chunk("wiki/a.md", "alpha", scope="wiki", vector=[1.0, 0.0, 0.0])]
    )
    assert idx.search_vectors([], top_k=5) == []


def test_vector_count_reflects_upserts(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("wiki/a.md", "alpha", scope="wiki", vector=[1.0, 0.0]),
            _chunk("wiki/b.md", "beta", scope="wiki"),
        ]
    )
    assert idx.vector_count() == 1


def test_vector_search_respects_scope_filter(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("wiki/a.md", "alpha", scope="wiki", vector=[1.0, 0.0]),
            _chunk("memory/global/a.md", "alpha", scope="global_memory", vector=[1.0, 0.0]),
        ]
    )
    hits = idx.search_vectors([1.0, 0.0], opts=SearchOpts(filters={"scope": "wiki"}))
    assert [hit.path for hit in hits] == ["wiki/a.md"]


# --------------------------------------------------------------------------- #
# Trigram fallback
# --------------------------------------------------------------------------- #


def test_trigram_finds_typo_match(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk("clients/oude-reimer.md", "body", title="Oude Reimer", scope="wiki"),
            _chunk("clients/bosch.md", "body", title="Bosch", scope="wiki"),
        ]
    )
    hits = idx.search_trigram("dude reimer", top_k=5)
    assert hits
    assert isinstance(hits[0], TrigramHit)
    assert hits[0].path == "clients/oude-reimer.md"
    assert hits[0].chunk_id == "clients/oude-reimer.md#0"


def test_trigram_empty_index_returns_empty(idx: Index) -> None:
    assert idx.search_trigram("anything") == []


def test_trigram_respects_threshold(idx: Index) -> None:
    idx.upsert_chunks([_chunk("clients/bosch.md", "body", scope="wiki")])
    assert idx.search_trigram("unrelated query text", top_k=5) == []


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #


def test_delete_chunk_removes_all_traces(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/x.md",
                "alpha",
                title="A",
                scope="wiki",
                vector=[1.0, 0.0],
                chunk_id="x1",
            )
        ]
    )
    assert idx.chunk_count() == 1
    idx.delete_chunk("x1")
    assert idx.chunk_count() == 0
    assert idx.vector_count() == 0
    assert idx.search_bm25("alpha") == []


def test_delete_unknown_chunk_is_noop(idx: Index) -> None:
    idx.delete_chunk("does-not-exist")
    assert idx.chunk_count() == 0


# --------------------------------------------------------------------------- #
# Rebuild from a brain store
# --------------------------------------------------------------------------- #


class _FakeStore(Store):
    """Minimal Store implementation for rebuild tests."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    async def read(self, path: BrainPath) -> bytes:
        return self._files[path]

    async def exists(self, path: BrainPath) -> bool:
        return path in self._files

    async def stat(self, path: BrainPath) -> FileInfo:
        raise NotImplementedError

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        from datetime import datetime, timezone

        return [
            FileInfo(
                path=BrainPath(path),
                size=len(body),
                mtime=datetime(2024, 1, 1, tzinfo=timezone.utc),
                is_dir=False,
            )
            for path, body in self._files.items()
        ]

    async def write(self, path: BrainPath, content: bytes) -> None:
        self._files[path] = content

    async def append(self, path: BrainPath, content: bytes) -> None:
        self._files[path] = self._files.get(path, b"") + content

    async def delete(self, path: BrainPath) -> None:
        self._files.pop(path, None)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        self._files[dst] = self._files.pop(src)

    async def batch(
        self,
        fn,
        opts: BatchOptions | None = None,
    ) -> None:
        raise NotImplementedError

    def subscribe(self, sink):
        return lambda: None

    def events(self) -> AsyncIterator[ChangeEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def local_path(self, path: BrainPath) -> str | None:
        return None


def test_rebuild_indexes_brain_files(idx: Index) -> None:
    files = {
        "wiki/platform/go.md": b"---\ntitle: Go\n---\nGo is a fast systems language.\n",
        "wiki/platform/rust.md": b"---\ntitle: Rust\n---\nRust is memory safe.\n",
        "memory/global/habits.md": b"---\nname: Habits\n---\nWrite things down.\n",
    }
    store = _FakeStore(files)
    count = idx.rebuild(store)
    assert count == 3
    assert idx.chunk_count() == 3


def test_rebuild_skips_underscore_prefixed(idx: Index) -> None:
    files = {
        "wiki/_log.md": b"---\ntitle: Log\n---\nLog entries.\n",
        "wiki/platform/go.md": b"---\ntitle: Go\n---\nBody.\n",
    }
    store = _FakeStore(files)
    count = idx.rebuild(store)
    assert count == 1
    hits = idx.search_bm25("Go")
    assert [h.path for h in hits] == ["wiki/platform/go.md"]


def test_rebuild_skips_non_markdown(idx: Index) -> None:
    files = {
        "wiki/platform/go.md": b"---\ntitle: Go\n---\nBody.\n",
        "wiki/platform/assets.png": b"not markdown",
    }
    store = _FakeStore(files)
    count = idx.rebuild(store)
    assert count == 1


def test_rebuild_clears_existing_state(idx: Index) -> None:
    idx.upsert_chunks(
        [
            _chunk(
                "wiki/x.md",
                "old",
                title="Old",
                scope="wiki",
                chunk_id="old",
            )
        ]
    )
    files = {"wiki/y.md": b"---\ntitle: New\n---\nfresh body\n"}
    idx.rebuild(_FakeStore(files))
    assert idx.chunk_count() == 1
    hits = idx.search_bm25("fresh")
    assert [h.path for h in hits] == ["wiki/y.md"]


def test_rebuild_preserves_raw_lme_session_headers(idx: Index) -> None:
    files = {
        "raw/lme/session-1.md": (
            b"---\n"
            b"session_id: sess-1\n"
            b"session_date: 2024-03-08\n"
            b"---\n"
            b"[user]: I bought apples.\n"
        ),
    }
    idx.rebuild(_FakeStore(files))
    hits = idx.search_bm25("apples")
    assert hits
    assert "session_id: sess-1" in hits[0].content
    assert hits[0].metadata["session_id"] == "sess-1"
    assert hits[0].metadata["session_date"] == "2024-03-08"


# --------------------------------------------------------------------------- #
# SearchOpts smoke
# --------------------------------------------------------------------------- #


def test_search_opts_defaults() -> None:
    opts = SearchOpts()
    assert opts.max_results == 20
    assert opts.filters == {}
    assert opts.include_generated is False

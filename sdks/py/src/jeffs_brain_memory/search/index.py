# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed hybrid search index.

Mirrors the Go SDK's ``Index`` contract (``sdks/go/search/index.go``)
for cross-language parity. One SQLite file holds three tables:

- ``knowledge_fts``: FTS5 virtual table keyed by ``path`` with the
  indexed body columns (``title``, ``summary``, ``tags``, ``content``,
  ``scope``, ``project_slug``, ``session_date``).
- ``knowledge_chunks``: metadata rows (chunk id, document id, path,
  frontmatter payload) so callers can hydrate results without a
  second lookup through the brain store.
- ``knowledge_embeddings``: one row per chunk, ``vector`` stored as
  a little-endian float32 ``BLOB``. When the optional ``sqlite-vec``
  extension loads successfully a parallel ``vec0`` virtual table is
  also maintained for faster lookup; otherwise the pure-Python cosine
  loop serves the same API.

The primary API takes :class:`Chunk` records rather than raw SQL rows
so callers can feed chunks produced by the ingestion pipeline without
synthesising markdown bodies.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .frontmatter import parse_memory_frontmatter, parse_wiki_frontmatter
from .query_parser import compile as compile_fts_ast
from .query_parser import parse as parse_query
from .query_parser import sanitise_query
from .trigram import TRIGRAM_JACCARD_THRESHOLD, TrigramIndex

__all__ = [
    "Chunk",
    "BM25Hit",
    "VectorHit",
    "TrigramHit",
    "SearchOpts",
    "Index",
    "EMBED_TEXT_MAX",
    "EMBED_BATCH_SIZE",
]

_log = logging.getLogger(__name__)

_SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    path,
    title,
    summary,
    tags,
    content,
    scope UNINDEXED,
    project_slug UNINDEXED,
    session_date UNINDEXED,
    tokenize='porter unicode61'
);
"""

_SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    path        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    tags        TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    scope       TEXT NOT NULL DEFAULT '',
    project_slug TEXT NOT NULL DEFAULT '',
    session_date TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}',
    checksum    TEXT NOT NULL DEFAULT '',
    generated   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON knowledge_chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON knowledge_chunks(document_id);
"""

_SCHEMA_VECTORS = """
CREATE TABLE IF NOT EXISTS knowledge_embeddings (
    chunk_id TEXT PRIMARY KEY,
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL,
    model    TEXT NOT NULL DEFAULT '',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chunk_id) REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON knowledge_embeddings(model);
"""

# Text cap applied to each document before it reaches the embedder.
# 8k chars is roughly 2k tokens, comfortably within most embedding
# context windows while keeping a single bad document from starving a
# whole batch. Matches the Go SDK's cap in backfill_vectors.
EMBED_TEXT_MAX = 8192

# Backfill batch size. Matches the Go SDK's constant so cross-language
# behaviour stays lockstep when comparing LongMemEval runs.
EMBED_BATCH_SIZE = 100

_RANK_EXPR = "bm25(3.0, 10.0, 5.0, 4.0, 1.0)"


@dataclass(slots=True)
class Chunk:
    """A single ingestable unit of text."""

    id: str
    document_id: str
    path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One BM25 result. ``score`` is the raw FTS5 ``rank`` value."""

    path: str
    title: str
    snippet: str
    score: float
    document_id: str
    chunk_id: str = ""


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One vector-search result. ``score`` is cosine similarity in ``[-1, 1]``."""

    path: str
    title: str
    score: float
    document_id: str
    chunk_id: str = ""


@dataclass(frozen=True, slots=True)
class TrigramHit:
    """One fuzzy fallback result. ``score`` is Jaccard similarity in ``[0, 1]``."""

    path: str
    score: float
    chunk_id: str = ""


@dataclass(slots=True)
class SearchOpts:
    """Options threaded through the search API.

    ``filters`` is a free-form mapping that narrows results by metadata
    columns (``scope``, ``project_slug``, ``tags``) when present.
    """

    max_results: int = 20
    filters: dict[str, Any] = field(default_factory=dict)
    include_generated: bool = False


def _pack_float32(values: Sequence[float]) -> bytes:
    """Encode ``values`` as little-endian float32 bytes."""
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_float32(blob: bytes) -> list[float]:
    """Decode a little-endian float32 ``BLOB`` into a Python list."""
    if not blob:
        return []
    if len(blob) % 4:
        raise ValueError(f"vector blob length {len(blob)} is not a multiple of 4")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _l2_norm(values: Sequence[float]) -> float:
    total = 0.0
    for v in values:
        total += float(v) * float(v)
    return math.sqrt(total)


def _cosine(a: Sequence[float], b: Sequence[float], norm_a: float, norm_b: float) -> float:
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = 0.0
    for av, bv in zip(a, b, strict=False):
        dot += float(av) * float(bv)
    return dot / (norm_a * norm_b)


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _classify_path(path: str) -> tuple[str, str]:
    """Return ``(scope, project_slug)`` for a logical brain path."""
    if not path.endswith(".md"):
        return ("", "")
    if path.startswith("memory/global/"):
        return ("global_memory", "")
    if path.startswith("memory/project/"):
        rest = path[len("memory/project/") :]
        slug, sep, _ = rest.partition("/")
        if not sep or not slug:
            return ("", "")
        return ("project_memory", slug)
    if path.startswith("wiki/"):
        return ("wiki", "")
    if path.startswith("raw/documents/"):
        return ("raw_document", "")
    if path.startswith("raw/.sources/"):
        return ("sources", "")
    if path.startswith("raw/lme/"):
        # LongMemEval bulk-ingested sessions; indexed so the tri-SDK
        # eval falls back gracefully when extraction produces no facts.
        return ("raw_lme", "")
    return ("", "")


class Index:
    """Hybrid search index over an SQLite file.

    Callers supply a SQLite path; the index opens its own connection
    with WAL + a 10s busy timeout. Passing ``:memory:`` (the default)
    keeps state inside the process so tests can run without a
    temporary file.

    ``sqlite-vec`` is loaded lazily at construction time. When loading
    fails (missing binary, Python sqlite3 built without extension
    support) the index transparently falls back to pure-Python cosine
    similarity. Both paths share the ``knowledge_embeddings`` table so
    data is portable.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._vec_loaded = self._try_load_sqlite_vec()
        self._apply_schema()

    def _try_load_sqlite_vec(self) -> bool:
        """Attempt to load the ``sqlite-vec`` extension.

        Returns ``True`` when loading succeeded and the ``vec0`` virtual
        table machinery is available. The pure-Python cosine path works
        regardless of this result.
        """
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except ImportError:
            _log.debug("sqlite-vec not installed; falling back to pure-Python cosine")
            return False
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except (sqlite3.OperationalError, AttributeError) as err:
            _log.info("sqlite-vec load failed (%s); using pure-Python cosine", err)
            return False
        return True

    def _apply_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA_FTS)
            self._conn.executescript(_SCHEMA_CHUNKS)
            self._conn.executescript(_SCHEMA_VECTORS)
            self._conn.execute(
                "INSERT INTO knowledge_fts(knowledge_fts, rank) VALUES('rank', ?)",
                (_RANK_EXPR,),
            )
            self._evolve_embeddings_columns()

    def _evolve_embeddings_columns(self) -> None:
        """Add missing columns to ``knowledge_embeddings`` on upgrade.

        Existing deployments written before the ``model`` column landed
        keep their BLOB payload and inherit an empty model string.
        Re-running against an already-evolved schema is a no-op.
        """
        cursor = self._conn.execute("PRAGMA table_info(knowledge_embeddings)")
        existing = {row["name"] for row in cursor.fetchall()}
        if "model" not in existing:
            self._conn.execute(
                "ALTER TABLE knowledge_embeddings ADD COLUMN model TEXT NOT NULL DEFAULT ''"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON knowledge_embeddings(model)"
            )

    @property
    def vec_loaded(self) -> bool:
        """Whether ``sqlite-vec`` loaded successfully."""
        return self._vec_loaded

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Idempotent: a second call is a no-op.
        """
        try:
            self._conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def upsert_chunks(self, chunks: Iterable[Chunk], *, model: str = "") -> int:
        """Insert or replace ``chunks``.

        Vectors (if present on the chunk) are persisted in the same
        transaction so the FTS and embedding rows are always consistent.
        ``model`` is stored alongside each vector so the retrieval path
        can filter by the active embedding model and a deployment can
        A/B between models without mixed-dim ranking. Returns the
        number of chunks written.
        """
        batch = list(chunks)
        if not batch:
            return 0

        count = 0
        with self._conn:
            for chunk in batch:
                self._write_chunk(chunk, model=model)
                count += 1
        return count

    def _write_chunk(self, chunk: Chunk, *, model: str = "") -> None:
        scope = str(chunk.metadata.get("scope") or "")
        project_slug = str(chunk.metadata.get("project_slug") or "")
        if not scope:
            scope, inferred_slug = _classify_path(chunk.path)
            project_slug = project_slug or inferred_slug
        title = str(chunk.metadata.get("title") or "")
        summary = str(chunk.metadata.get("summary") or "")
        tags_raw = chunk.metadata.get("tags") or ""
        if isinstance(tags_raw, list):
            tags = " ".join(str(t) for t in tags_raw)
        else:
            tags = str(tags_raw)
        session_date = str(chunk.metadata.get("session_date") or "")
        body = chunk.text or ""
        generated = 1 if chunk.metadata.get("generated") else 0
        metadata_blob = _serialise_metadata(chunk.metadata)

        # FTS5 is a virtual table and lacks ON CONFLICT semantics, so
        # clear the path first. The chunks row uses an UPSERT so any
        # foreign-keyed embedding is preserved across an FTS rebuild
        # that has no new vector to offer; a cascade delete would drop
        # backfilled embeddings and force a re-embed on every mutation.
        self._conn.execute("DELETE FROM knowledge_fts WHERE path = ?", (chunk.path,))

        self._conn.execute(
            """
            INSERT INTO knowledge_fts (
                path, title, summary, tags, content, scope, project_slug, session_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk.path, title, summary, tags, body, scope, project_slug, session_date),
        )

        self._conn.execute(
            """
            INSERT INTO knowledge_chunks (
                chunk_id, document_id, path, title, summary, tags, content,
                scope, project_slug, session_date, metadata, checksum, generated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id = excluded.document_id,
                path = excluded.path,
                title = excluded.title,
                summary = excluded.summary,
                tags = excluded.tags,
                content = excluded.content,
                scope = excluded.scope,
                project_slug = excluded.project_slug,
                session_date = excluded.session_date,
                metadata = excluded.metadata,
                checksum = excluded.checksum,
                generated = excluded.generated
            """,
            (
                chunk.id,
                chunk.document_id,
                chunk.path,
                title,
                summary,
                tags,
                body,
                scope,
                project_slug,
                session_date,
                metadata_blob,
                _checksum(body),
                generated,
            ),
        )

        if chunk.vector:
            self._conn.execute(
                """
                INSERT INTO knowledge_embeddings (chunk_id, dim, vector, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    dim = excluded.dim,
                    vector = excluded.vector,
                    model = excluded.model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chunk.id, len(chunk.vector), _pack_float32(chunk.vector), model),
            )

    def delete_chunk(self, chunk_id: str) -> None:
        """Remove a single chunk from the index."""
        with self._conn:
            row = self._conn.execute(
                "SELECT path FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row is None:
                return
            self._conn.execute("DELETE FROM knowledge_fts WHERE path = ?", (row["path"],))
            self._conn.execute("DELETE FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))
            self._conn.execute("DELETE FROM knowledge_embeddings WHERE chunk_id = ?", (chunk_id,))

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
        opts: SearchOpts | None = None,
    ) -> list[BM25Hit]:
        """Run a BM25 search over the FTS5 index.

        ``query`` is compiled to an FTS5 expression via
        :func:`sanitise_query`. Empty input returns an empty list so
        callers can skip the round-trip.
        """
        opts = opts or SearchOpts(max_results=top_k)
        expr = sanitise_query(query)
        if not expr:
            return []

        limit = opts.max_results or top_k or 20
        sql = (
            """
            SELECT
                fts.path       AS path,
                fts.title      AS title,
                snippet(knowledge_fts, 4, '<mark>', '</mark>', '...', 64) AS snippet,
                fts.rank       AS score,
                fts.scope      AS scope,
                fts.project_slug AS project_slug,
                fts.session_date AS session_date,
                c.document_id  AS document_id,
                c.chunk_id     AS chunk_id,
                c.generated    AS generated,
                c.tags         AS tags
            FROM knowledge_fts AS fts
            LEFT JOIN knowledge_chunks AS c ON c.path = fts.path
            WHERE knowledge_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
            """
        )
        rows = self._conn.execute(sql, (expr, max(limit * 3, 100))).fetchall()

        hits: list[BM25Hit] = []
        for row in rows:
            if not self._passes_filters(row, opts):
                continue
            hits.append(
                BM25Hit(
                    path=row["path"],
                    title=row["title"] or "",
                    snippet=row["snippet"] or "",
                    score=float(row["score"] or 0.0),
                    document_id=row["document_id"] or "",
                    chunk_id=row["chunk_id"] or "",
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _passes_filters(self, row: sqlite3.Row, opts: SearchOpts) -> bool:
        """Apply scope / project / tag filters.

        Rows without a metadata twin in ``knowledge_chunks`` (rare:
        direct FTS inserts in tests) are treated as visible.
        """
        if not opts.include_generated and (row["generated"] or 0):
            return False
        filters = opts.filters or {}
        scope = filters.get("scope")
        if scope and row["scope"] and row["scope"] != scope:
            return False
        project_slug = filters.get("project_slug")
        if project_slug and row["project_slug"] and row["project_slug"] != project_slug:
            return False
        tag = filters.get("tag")
        if tag:
            tags = (row["tags"] or "").split()
            if tag not in tags:
                return False
        path_prefix = filters.get("path_prefix")
        if path_prefix and not row["path"].startswith(path_prefix):
            return False
        return True

    def search_vectors(
        self,
        query_vec: Sequence[float],
        top_k: int = 20,
        opts: SearchOpts | None = None,
        *,
        model: str = "",
    ) -> list[VectorHit]:
        """Cosine-rank every stored vector against ``query_vec``.

        Uses the ``sqlite-vec`` virtual table when the extension loaded;
        falls back to a pure-Python scan otherwise. Both paths share
        the same persistent storage so swapping between them is a
        no-op for callers.

        When ``model`` is non-empty, the scan is restricted to rows
        written under that embedding model so a dimension mismatch
        between stored vectors and the query cannot poison the ranking.
        """
        opts = opts or SearchOpts(max_results=top_k)
        if not query_vec:
            return []
        limit = opts.max_results or top_k or 20

        sql = """
            SELECT e.chunk_id AS chunk_id, e.vector AS vector,
                   c.path AS path, c.title AS title,
                   c.document_id AS document_id, c.generated AS generated,
                   c.scope AS scope, c.project_slug AS project_slug,
                   c.tags AS tags
            FROM knowledge_embeddings AS e
            JOIN knowledge_chunks AS c ON c.chunk_id = e.chunk_id
        """
        params: tuple[Any, ...] = ()
        if model:
            sql += " WHERE e.model = ?"
            params = (model,)
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        query_norm = _l2_norm(query_vec)
        scored: list[VectorHit] = []
        for row in rows:
            if not self._passes_filters(row, opts):
                continue
            stored = _unpack_float32(row["vector"])
            if len(stored) != len(query_vec):
                continue
            score = _cosine(query_vec, stored, query_norm, _l2_norm(stored))
            scored.append(
                VectorHit(
                    path=row["path"],
                    title=row["title"] or "",
                    score=score,
                    document_id=row["document_id"] or "",
                    chunk_id=row["chunk_id"],
                )
            )
        scored.sort(key=lambda h: (-h.score, h.path))
        return scored[:limit]

    def list_indexed_paths(self) -> list[str]:
        """Return every path currently carried by the FTS / chunk tables.

        Mirrors the Go ``Index.IndexedPaths`` surface so the vector
        backfill walker can decide which documents still need embedding.
        Paths are returned in ascending lexicographic order.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT path FROM knowledge_chunks ORDER BY path"
        ).fetchall()
        return [row["path"] for row in rows if row["path"]]

    def paths_with_vectors(self, model: str) -> set[str]:
        """Return the set of paths already embedded under ``model``.

        Used by the vector backfill to skip documents it has already
        processed: a restart is a no-op when every indexed path has a
        row here, and a partial run only embeds the remainder.
        """
        if not model:
            rows = self._conn.execute(
                """
                SELECT DISTINCT c.path AS path
                FROM knowledge_embeddings AS e
                JOIN knowledge_chunks AS c ON c.chunk_id = e.chunk_id
                """
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT DISTINCT c.path AS path
                FROM knowledge_embeddings AS e
                JOIN knowledge_chunks AS c ON c.chunk_id = e.chunk_id
                WHERE e.model = ?
                """,
                (model,),
            ).fetchall()
        return {row["path"] for row in rows if row["path"]}

    def upsert_embeddings(
        self,
        items: Sequence[tuple[str, Sequence[float]]],
        *,
        model: str,
    ) -> int:
        """Persist ``(path, vector)`` pairs under ``model`` atomically.

        Writes one row per distinct chunk backing ``path``. When multiple
        chunks share a path the vector is replicated across them so hit
        ranking stays consistent. Empty vectors are skipped to match
        the Go SDK's ``StoreBatch`` guard. Returns the number of
        ``knowledge_embeddings`` rows written.
        """
        if not items:
            return 0
        count = 0
        with self._conn:
            for path, vector in items:
                if not path or not vector:
                    continue
                rows = self._conn.execute(
                    "SELECT chunk_id FROM knowledge_chunks WHERE path = ?",
                    (path,),
                ).fetchall()
                if not rows:
                    continue
                blob = _pack_float32(vector)
                dim = len(vector)
                for row in rows:
                    self._conn.execute(
                        """
                        INSERT INTO knowledge_embeddings (chunk_id, dim, vector, model)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            dim = excluded.dim,
                            vector = excluded.vector,
                            model = excluded.model,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (row["chunk_id"], dim, blob, model),
                    )
                    count += 1
        return count

    def search_trigram(
        self,
        query: str,
        top_k: int = 20,
        *,
        threshold: float = TRIGRAM_JACCARD_THRESHOLD,
    ) -> list[TrigramHit]:
        """Jaccard-trigram fallback over the indexed slugs.

        Builds a fresh :class:`~jeffs_brain_memory.search.trigram.TrigramIndex`
        on every call. The cost is linear in the number of indexed
        chunks; acceptable for the small brains this fallback targets.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, path FROM knowledge_chunks ORDER BY path"
        ).fetchall()
        if not rows:
            return []
        path_to_chunk: dict[str, str] = {row["path"]: row["chunk_id"] for row in rows}
        paths = list(path_to_chunk.keys())
        index = TrigramIndex(paths)
        raw_hits = index.fuzzy_search(query, top_k=top_k, threshold=threshold)
        return [
            TrigramHit(path=hit.path, score=hit.score, chunk_id=path_to_chunk.get(hit.path, ""))
            for hit in raw_hits
        ]

    def rebuild(self, store: Any) -> int:
        """Rebuild the index from a brain store.

        The store is consumed via its :meth:`list` and :meth:`read`
        coroutines. We accept any object that looks like
        :class:`jeffs_brain_memory.store.Store`; the Python store is
        async-first so this helper schedules an event loop on the
        caller's behalf. Returns the number of files indexed.
        """
        import asyncio

        from ..store import ListOpts

        if hasattr(store, "list_sync") and hasattr(store, "read_sync"):
            entries = store.list_sync(
                ListOpts(recursive=True, include_generated=True),
            )
            reader = store.read_sync
        else:
            entries = asyncio.run(
                store.list("", ListOpts(recursive=True, include_generated=True)),
            )

            def reader(path: str) -> bytes:
                return asyncio.run(store.read(path))

        count = 0
        with self._conn:
            self._conn.execute("DELETE FROM knowledge_fts")
            self._conn.execute("DELETE FROM knowledge_chunks")
            self._conn.execute("DELETE FROM knowledge_embeddings")
            for entry in entries:
                path = getattr(entry, "path", None) or entry
                if not isinstance(path, str):
                    path = str(path)
                if not path.endswith(".md"):
                    continue
                basename = path.rsplit("/", 1)[-1]
                if basename.startswith("_"):
                    continue
                scope, project_slug = _classify_path(path)
                if not scope:
                    continue
                try:
                    raw_bytes = reader(path)
                except Exception as err:  # pragma: no cover - store errors are noisy
                    _log.debug("rebuild skipping %s: %s", path, err)
                    continue
                raw = raw_bytes.decode("utf-8", errors="replace")
                if scope == "wiki":
                    wiki, body = parse_wiki_frontmatter(raw)
                    title, summary, tags = wiki.title, wiki.summary, wiki.tags
                else:
                    mem, body = parse_memory_frontmatter(raw)
                    title, summary, tags = mem.name, mem.description, mem.tags
                chunk = Chunk(
                    id=f"{path}#0",
                    document_id=path,
                    path=path,
                    text=body,
                    metadata={
                        "title": title,
                        "summary": summary,
                        "tags": tags,
                        "scope": scope,
                        "project_slug": project_slug,
                    },
                )
                self._write_chunk(chunk)
                count += 1
        return count

    def chunk_count(self) -> int:
        """Return the number of chunks currently indexed."""
        row = self._conn.execute("SELECT count(*) AS n FROM knowledge_chunks").fetchone()
        return int(row["n"] or 0)

    def vector_count(self) -> int:
        """Return the number of stored embeddings."""
        row = self._conn.execute("SELECT count(*) AS n FROM knowledge_embeddings").fetchone()
        return int(row["n"] or 0)


def _serialise_metadata(metadata: dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(metadata, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "{}"


__all__.extend(["parse_query", "compile_fts_ast"])

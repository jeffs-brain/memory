# SPDX-License-Identifier: Apache-2.0
"""Concrete :class:`Base` implementation plus the :func:`new` factory.

Ported from ``sdks/go/knowledge/knowledge.go`` and the sibling
``ingest.go`` / ``compile.go`` / ``search.go`` modules. The Python
surface is asynchronous throughout so it slots into the rest of the
SDK without an event loop adapter.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ..path import BrainPath, DocumentID
from .compile import segment_document
from .frontmatter import parse_frontmatter
from .ingest import (
    CONTENT_TYPE_MARKDOWN,
    DefaultFetcher,
    Fetcher,
    RAW_DOCUMENTS_PREFIX,
    build_document,
    build_frontmatter_yaml,
    extract_plain,
    normalise_url,
    read_body,
)
from .search import InMemoryScorer, IndexLike, Retriever, run_search
from .types import (
    Chunk,
    CompileOptions,
    CompileResult,
    Document,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

__all__ = ["Base", "Options", "new"]


@runtime_checkable
class Base(Protocol):
    """Top-level knowledge surface.

    Implementations are safe for concurrent use. :func:`new` returns a
    concrete ``_KBase`` that satisfies the contract.
    """

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        """Persist a single document. See :class:`IngestRequest`."""
        ...

    async def ingest_url(self, url: str) -> IngestResponse:
        """Fetch a URL and persist the extracted plain-text body."""
        ...

    async def compile(self, opts: CompileOptions) -> CompileResult:
        """Chunk persisted documents into the bound search index."""
        ...

    async def search(self, req: SearchRequest) -> SearchResponse:
        """Run a hybrid retrieval against the bound index."""
        ...

    async def close(self) -> None:
        """Release package-local state. The store lifecycle stays with the caller."""
        ...


@dataclass(slots=True)
class Options:
    """Arguments to :func:`new`."""

    brain_id: str = ""
    store: "StoreLike | None" = None
    index: IndexLike | None = None
    retriever: Retriever | None = None
    fetcher: Fetcher | None = None


@runtime_checkable
class StoreLike(Protocol):
    """Tiny store surface the knowledge package needs.

    Compatible with :class:`jeffs_brain_memory.store.Store` but declared
    here so the module stays importable even when the store stubs in the
    rest of the SDK are not yet concrete.
    """

    async def read(self, path: BrainPath) -> bytes: ...

    async def write(self, path: BrainPath, content: bytes) -> None: ...

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: Any = None,
    ) -> list[Any]: ...


class _KBase:
    """Concrete :class:`Base` backed by the supplied options."""

    def __init__(self, opts: Options) -> None:
        if opts.store is None:
            raise ValueError("knowledge: Options.store is required")
        self._brain_id = opts.brain_id
        self._store = opts.store
        self._index = opts.index
        self._retriever = opts.retriever
        self._fetcher: Fetcher = opts.fetcher or DefaultFetcher()
        self._lock = asyncio.Lock()

    # --- setters ---------------------------------------------------------

    def set_search_index(self, index: IndexLike | None) -> None:
        self._index = index

    def set_retriever(self, retriever: Retriever | None) -> None:
        self._retriever = retriever

    def store(self) -> StoreLike:
        return self._store

    # --- ingest ----------------------------------------------------------

    async def ingest(self, req: IngestRequest) -> IngestResponse:
        start = time.perf_counter()
        raw, ctype, source_label = read_body(req)
        extension = _extension(req.path)
        extracted = extract_plain(raw, ctype, extension)
        doc = build_document(
            req=req,
            content_type=ctype,
            source_label=source_label,
            extracted=extracted,
            raw=raw,
            brain_id=self._brain_id,
        )
        await self._write_document(doc)
        chunks = await self._chunk_and_index(doc)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return IngestResponse(
            document_id=doc.id,
            path=doc.path,
            chunk_count=chunks,
            bytes=doc.bytes,
            took_ms=elapsed_ms,
        )

    async def ingest_url(self, url: str) -> IngestResponse:
        parsed = normalise_url(url)
        body, ctype = await self._fetcher.fetch(parsed)
        req = IngestRequest(
            brain_id=self._brain_id,
            path=parsed,
            content_type=ctype,
            content=body,
        )
        return await self.ingest(req)

    async def _write_document(self, doc: Document) -> None:
        """Serialise the document with a regenerated frontmatter header."""
        fm = build_frontmatter_yaml(doc)
        payload = fm + "\n\n" + doc.body + "\n"
        await self._store.write(doc.path, payload.encode("utf-8"))

    # --- compile ---------------------------------------------------------

    async def compile(self, opts: CompileOptions) -> CompileResult:
        start = time.perf_counter()
        targets = await self._resolve_targets(opts.paths)
        res = CompileResult()

        for i, path in enumerate(targets):
            if opts.max_batch > 0 and i >= opts.max_batch:
                break
            try:
                data = await self._store.read(path)
            except Exception:  # noqa: BLE001
                res.errors += 1
                continue
            doc = _document_from_stored(path, data)
            if doc is None:
                res.skipped += 1
                continue
            if opts.dry_run:
                chunks = segment_document(doc)
                res.documents += 1
                res.chunks += len(chunks)
                continue
            try:
                n = await self._chunk_and_index(doc)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                res.errors += 1
                continue
            res.documents += 1
            res.chunks += n

        res.elapsed_ms = int((time.perf_counter() - start) * 1000)
        return res

    async def _resolve_targets(self, explicit: list[BrainPath]) -> list[BrainPath]:
        if explicit:
            return list(explicit)

        # ListOpts is optional; pass an object that mirrors the Go shape
        # but stay defensive in case the bound store has a different
        # signature.
        try:
            from ..store import ListOpts

            opts = ListOpts(recursive=True, include_generated=True)
        except Exception:  # noqa: BLE001 - ListOpts might not exist yet
            opts = None

        try:
            entries = await self._store.list(RAW_DOCUMENTS_PREFIX, opts)
        except TypeError:
            entries = await self._store.list(RAW_DOCUMENTS_PREFIX)

        out: list[BrainPath] = []
        for entry in entries or []:
            path = _entry_path(entry)
            if path is None:
                continue
            is_dir = getattr(entry, "is_dir", False)
            if is_dir:
                continue
            if not str(path).endswith(".md"):
                continue
            out.append(path)
        return out

    async def _chunk_and_index(self, doc: Document) -> int:
        """Segment ``doc`` and, when bound, tell the index to reindex.

        Mirrors Go's behaviour: the document is already persisted by the
        time we get here, so the index only has to walk the new path.
        """
        chunks = segment_document(doc)
        if self._index is None:
            return len(chunks)
        update = getattr(self._index, "update", None)
        if callable(update):
            try:
                result = update()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                # Swallow: the next Update() heals any gap, matching Go.
                pass
        return len(chunks)

    # --- search ----------------------------------------------------------

    async def search(self, req: SearchRequest) -> SearchResponse:
        start = time.perf_counter()
        fallback = InMemoryScorer(
            list_documents=self._list_raw_documents,
            read_document=self._store.read,
        )
        resp = await run_search(
            req=req,
            index=self._index,
            retriever=self._retriever,
            in_memory_fallback=fallback,
            brain_id=self._brain_id,
        )
        resp.elapsed_ms = int((time.perf_counter() - start) * 1000)
        return resp

    async def _list_raw_documents(self) -> list[BrainPath]:
        try:
            from ..store import ListOpts

            opts = ListOpts(recursive=True, include_generated=True)
        except Exception:  # noqa: BLE001
            opts = None

        try:
            entries = await self._store.list(RAW_DOCUMENTS_PREFIX, opts)
        except TypeError:
            entries = await self._store.list(RAW_DOCUMENTS_PREFIX)
        except Exception:  # noqa: BLE001
            return []
        out: list[BrainPath] = []
        for entry in entries or []:
            path = _entry_path(entry)
            if path is None:
                continue
            if getattr(entry, "is_dir", False):
                continue
            if not str(path).endswith(".md"):
                continue
            out.append(path)
        return out

    # --- lifecycle -------------------------------------------------------

    async def close(self) -> None:
        """Idempotent teardown. Matches Go: the store lifecycle is caller-owned."""
        return None


def new(opts: Options) -> Base:
    """Factory. Mirrors Go's ``knowledge.New``."""
    return _KBase(opts)


def _extension(path: str) -> str:
    """Return the dot-prefixed lowercased extension."""
    idx = path.rfind(".")
    if idx < 0:
        return ""
    return path[idx:].lower()


def _entry_path(entry: Any) -> BrainPath | None:
    """Best-effort extraction of a path string from a list entry."""
    for attr in ("path", "Path", "logical", "name"):
        value = getattr(entry, attr, None)
        if value:
            return BrainPath(str(value))
    if isinstance(entry, str):
        return BrainPath(entry)
    return None


def _document_from_stored(path: BrainPath, data: bytes) -> Document | None:
    """Reconstruct a :class:`Document` from a persisted markdown file."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    fm, body = parse_frontmatter(text)
    title = fm.title or fm.name
    if not title:
        title = str(path).rsplit("/", 1)[-1]
        if title.endswith(".md"):
            title = title[:-3]
    doc_id = DocumentID(
        __import__("hashlib").sha256(data).hexdigest()[:12]
    )
    return Document(
        id=doc_id,
        brain_id="",
        path=path,
        title=title,
        source=fm.source,
        content_type=CONTENT_TYPE_MARKDOWN,
        summary=fm.summary or fm.description,
        body=body.strip(),
        bytes=len(data),
        tags=list(fm.tags),
    )

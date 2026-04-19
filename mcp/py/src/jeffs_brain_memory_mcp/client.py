# SPDX-License-Identifier: Apache-2.0
"""Unified memory client used by every tool handler.

Hides whether we are wired to a local filesystem brain (plus a SQLite
FTS5 index and optional Ollama embeddings) or to the hosted platform
over HTTP.

Both backends expose the same eleven-tool surface. Local mode builds a
self-contained BM25-only retrieval pipeline out of the standard library
plus SQLite; hosted mode proxies through to the platform REST API
described in ``spec/MCP-TOOLS.md``.

The local implementation mirrors the TypeScript wrapper's fallback path
but does not depend on the Python ``jeffs_brain_memory`` SDK's store,
search, and ingest modules because those are still stubbed at the time
of writing. We import the SDK's ``path`` module for ``BrainPath`` so the
ecosystem stays consistent.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from .config import ConfigMode, HostedConfig, LocalConfig


Role = Literal["system", "user", "assistant", "tool"]
Scope = Literal["all", "global", "project", "agent"]
RecallScope = Literal["global", "project", "agent"]
Sort = Literal["relevance", "recency", "relevance_then_recency"]
IngestAs = Literal["markdown", "text", "pdf", "json"]


# ---------------------------------------------------------------------------
# Input models (shared between tool handlers and the client surface)
# ---------------------------------------------------------------------------


class ExtractMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class RememberArgs:
    content: str
    title: str | None = None
    brain: str | None = None
    tags: tuple[str, ...] = ()
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SearchArgs:
    query: str
    brain: str | None = None
    top_k: int | None = None
    scope: Scope | None = None
    sort: Sort | None = None


@dataclass(frozen=True, slots=True)
class RecallArgs:
    query: str
    brain: str | None = None
    scope: RecallScope | None = None
    session_id: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class AskArgs:
    query: str
    brain: str | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class IngestFileArgs:
    path: str
    brain: str | None = None
    as_: IngestAs | None = None


@dataclass(frozen=True, slots=True)
class IngestUrlArgs:
    url: str
    brain: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractArgs:
    messages: tuple[ExtractMessage, ...]
    brain: str | None = None
    actor_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReflectArgs:
    session_id: str
    brain: str | None = None


@dataclass(frozen=True, slots=True)
class ConsolidateArgs:
    brain: str | None = None


@dataclass(frozen=True, slots=True)
class CreateBrainArgs:
    name: str
    slug: str | None = None
    visibility: Literal["private", "tenant", "public"] | None = None


# ---------------------------------------------------------------------------
# Progress plumbing
# ---------------------------------------------------------------------------


ProgressEmitter = Callable[[float, str | None], Awaitable[None]]


class MemoryClient(Protocol):
    """Unified dispatch surface every tool handler speaks against."""

    mode: Literal["local", "hosted"]

    async def remember(self, args: RememberArgs) -> dict[str, Any]: ...

    async def recall(self, args: RecallArgs) -> dict[str, Any]: ...

    async def search(self, args: SearchArgs) -> dict[str, Any]: ...

    async def ask(
        self, args: AskArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def ingest_file(
        self, args: IngestFileArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def ingest_url(
        self, args: IngestUrlArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def extract(
        self, args: ExtractArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def reflect(
        self, args: ReflectArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def consolidate(
        self, args: ConsolidateArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]: ...

    async def create_brain(self, args: CreateBrainArgs) -> dict[str, Any]: ...

    async def list_brains(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------


DEFAULT_BRAIN_ID = "default"
FILE_LIMIT_BYTES = 25 * 1024 * 1024
URL_FETCH_LIMIT_BYTES = 5 * 1024 * 1024

MIME_BY_AS: dict[str, str] = {
    "markdown": "text/markdown",
    "text": "text/plain",
    "pdf": "application/pdf",
    "json": "application/json",
}

MIME_BY_EXT: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_from_title(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", title.lower().strip()).strip("-")
    if not cleaned:
        return f"note-{int(time.time() * 1000)}"
    return cleaned[:64]


def _sanitise_brain_id(brain_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", brain_id)
    if not cleaned or cleaned.startswith("."):
        return "_invalid"
    return cleaned


def _derive_title(content: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "Untitled memory"


def _mime_for(path: str, hint: IngestAs | None) -> str:
    if hint:
        return MIME_BY_AS[hint]
    ext = Path(path).suffix.lower()
    if ext in MIME_BY_EXT:
        return MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _build_frontmatter(values: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(json.dumps(str(v)) for v in value)
            lines.append(f"{key}: [{rendered}]")
            continue
        lines.append(f"{key}: {json.dumps(value)}")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local mode — FS store + SQLite FTS5 index + optional Ollama embedder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BrainResources:
    brain_id: str
    root: Path
    conn: sqlite3.Connection

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover - best-effort
            pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_index(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
        "id UNINDEXED, path UNINDEXED, ordinal UNINDEXED, title, content,"
        "metadata UNINDEXED, tokenize='unicode61 remove_diacritics 2')"
    )
    conn.commit()
    return conn


def _upsert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    path: str,
    ordinal: int,
    title: str,
    content: str,
    metadata: dict[str, Any],
) -> None:
    conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    conn.execute(
        "INSERT INTO chunks (id, path, ordinal, title, content, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            path,
            ordinal,
            title,
            content,
            json.dumps(metadata, separators=(",", ":")),
        ),
    )
    conn.commit()


_FTS_RESERVED = re.compile(r"[^A-Za-z0-9_]+")


def _fts_query(raw: str) -> str:
    tokens = [t for t in _FTS_RESERVED.split(raw) if t]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def _search_index(
    conn: sqlite3.Connection, query: str, top_k: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, path, ordinal, title, content, metadata, bm25(chunks) AS score "
        "FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
        (_fts_query(query), top_k),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        metadata: dict[str, Any] = {}
        try:
            metadata = json.loads(row[5]) if row[5] else {}
        except json.JSONDecodeError:
            metadata = {}
        results.append(
            {
                "chunk_id": row[0],
                "path": row[1],
                "ordinal": row[2],
                "title": row[3],
                "content": row[4],
                "metadata": metadata,
                "score": -float(row[6]) if row[6] is not None else 0.0,
            }
        )
    return results


async def _ollama_reachable(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


class LocalMemoryClient:
    """Local filesystem implementation of :class:`MemoryClient`."""

    mode: Literal["local", "hosted"] = "local"

    def __init__(self, cfg: LocalConfig) -> None:
        self._cfg = cfg
        self._brains: dict[str, _BrainResources] = {}
        self._bootstrapped = False
        self._ollama_ok: bool | None = None

    # --- bootstrap ------------------------------------------------------

    async def _ensure_bootstrap(self) -> None:
        if self._bootstrapped:
            return
        _ensure_dir(self._cfg.brain_root)
        self._bootstrapped = True

    def _resolve_brain(self, override: str | None) -> str:
        if override:
            return override
        if self._cfg.default_brain:
            return self._cfg.default_brain
        return DEFAULT_BRAIN_ID

    def _open_brain(self, brain_id: str) -> _BrainResources:
        existing = self._brains.get(brain_id)
        if existing is not None:
            return existing
        root = self._cfg.brain_root / _sanitise_brain_id(brain_id)
        _ensure_dir(root)
        conn = _open_index(root / "search.sqlite")
        resource = _BrainResources(brain_id=brain_id, root=root, conn=conn)
        self._brains[brain_id] = resource
        return resource

    async def _ollama_available(self) -> bool:
        if self._ollama_ok is None:
            self._ollama_ok = await _ollama_reachable(self._cfg.ollama_base_url)
        return self._ollama_ok

    # --- primitives -----------------------------------------------------

    def _write_note(
        self,
        brain: _BrainResources,
        *,
        relative_path: str,
        content: bytes,
        title: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        target = brain.root / relative_path
        _ensure_dir(target.parent)
        target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        chunk_id = f"{brain.brain_id}:{digest[:16]}:0"
        _upsert_chunk(
            brain.conn,
            chunk_id=chunk_id,
            path=relative_path,
            ordinal=0,
            title=title,
            content=content.decode("utf-8", errors="ignore")[:32_000],
            metadata={"path": relative_path, "brain_id": brain.brain_id, **metadata},
        )
        return {
            "document_id": f"{brain.brain_id}:{digest[:16]}",
            "chunk_id": chunk_id,
            "hash": digest,
            "path": relative_path,
            "byte_size": len(content),
        }

    # --- tool methods ---------------------------------------------------

    async def remember(self, args: RememberArgs) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        title = _derive_title(args.content, args.title)
        slug = _slug_from_title(title)
        rel_path = args.path or f"memory/global/{slug}.md"
        frontmatter = _build_frontmatter(
            {
                "title": title,
                "scope": "global",
                "type": "user",
                "created": _now_iso(),
                "modified": _now_iso(),
                **({"tags": list(args.tags)} if args.tags else {}),
            }
        )
        body = f"{frontmatter}\n\n{args.content.rstrip()}\n"
        content_bytes = body.encode("utf-8")
        metadata: dict[str, Any] = {}
        if args.tags:
            metadata["tags"] = ",".join(args.tags)
        written = self._write_note(
            brain,
            relative_path=rel_path,
            content=content_bytes,
            title=title,
            metadata=metadata,
        )
        return {
            "id": written["document_id"],
            "brain_id": brain_id,
            "title": title,
            "path": written["path"],
            "source": "ingest",
            "content_type": "text/markdown",
            "byte_size": written["byte_size"],
            "checksum_sha256": written["hash"],
            "metadata": metadata,
            "commit_sha": written["hash"][:12],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "deleted_at": None,
            "chunk_count": 1,
            "embedded_count": 0,
            "reused": False,
        }

    async def search(self, args: SearchArgs) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        top_k = args.top_k or 10
        started = time.perf_counter()
        hits = _search_index(brain.conn, args.query, top_k)
        took_ms = int((time.perf_counter() - started) * 1000)
        return {
            "query": args.query,
            "brain_id": brain_id,
            "hits": [
                {
                    "score": hit["score"],
                    "path": hit["path"],
                    "content": hit["content"],
                    "title": hit["title"],
                    "chunk_id": hit["chunk_id"],
                    "metadata": hit["metadata"],
                }
                for hit in hits
            ],
            "took_ms": took_ms,
        }

    async def recall(self, args: RecallArgs) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        top_k = args.top_k or 5
        hits = _search_index(brain.conn, args.query, top_k)
        return {
            "query": args.query,
            "brain_id": brain_id,
            "session_id": args.session_id,
            "chunks": [
                {
                    "score": hit["score"],
                    "path": hit["path"],
                    "content": hit["content"],
                    "title": hit["title"],
                    "chunk_id": hit["chunk_id"],
                }
                for hit in hits
            ],
        }

    async def ask(
        self, args: AskArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        top_k = args.top_k or 8
        hits = _search_index(brain.conn, args.query, top_k)
        if progress is not None:
            await progress(0.0, "retrieved")
        answer_lines = [f"Results for '{args.query}':"]
        for idx, hit in enumerate(hits, start=1):
            snippet = (hit["content"] or "")[:240]
            answer_lines.append(f"[{idx}] {hit['path']}: {snippet}")
        if not hits:
            answer_lines.append("_no retrieval hits_")
        answer = "\n".join(answer_lines)
        if progress is not None:
            await progress(1.0, "answered")
        citations = [
            {
                "type": "citation",
                "chunk_id": hit["chunk_id"],
                "document_id": hit["path"],
                "answer_start": 0,
                "answer_end": 0,
                "quote": (hit["content"] or "")[:200],
            }
            for hit in hits[:5]
        ]
        retrieved = [
            {
                "chunk_id": hit["chunk_id"],
                "document_id": hit["path"],
                "score": hit["score"],
                "preview": (hit["content"] or "")[:512],
            }
            for hit in hits
        ]
        return {"answer": answer, "citations": citations, "retrieved": retrieved}

    async def ingest_file(
        self, args: IngestFileArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        abs_path = Path(args.path).expanduser().resolve()
        if not abs_path.is_file():
            raise FileNotFoundError(f"memory_ingest_file: not a regular file: {abs_path}")
        size = abs_path.stat().st_size
        if size > FILE_LIMIT_BYTES:
            raise ValueError("file_too_large: 25 MiB limit exceeded")
        data = abs_path.read_bytes()
        if progress is not None:
            await progress(0.0, "read")
        mime = _mime_for(str(abs_path), args.as_)
        digest = hashlib.sha256(data).hexdigest()
        stored_path = f"raw/documents/{digest}{abs_path.suffix.lower() or '.bin'}"
        target = brain.root / stored_path
        _ensure_dir(target.parent)
        target.write_bytes(data)
        title = abs_path.name
        _upsert_chunk(
            brain.conn,
            chunk_id=f"{brain_id}:{digest[:16]}:0",
            path=stored_path,
            ordinal=0,
            title=title,
            content=data.decode("utf-8", errors="ignore")[:32_000],
            metadata={
                "path": stored_path,
                "brain_id": brain_id,
                "mime": mime,
                "source": str(abs_path),
            },
        )
        if progress is not None:
            await progress(1.0, "indexed")
        return {
            "status": "completed",
            "document_id": f"{brain_id}:{digest[:16]}",
            "path": stored_path,
            "hash": digest,
            "chunk_count": 1,
            "embedded_count": 0,
            "duration_ms": 0,
            "reused": False,
        }

    async def ingest_url(
        self, args: IngestUrlArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        brain = self._open_brain(brain_id)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(args.url)
            resp.raise_for_status()
            body = resp.content
            mime = (resp.headers.get("content-type") or "text/plain").split(";")[0].strip()
        if len(body) > URL_FETCH_LIMIT_BYTES:
            raise ValueError("memory_ingest_url: body exceeds 5 MiB fallback limit")
        if progress is not None:
            await progress(0.0, "fetched")
        digest = hashlib.sha256(body).hexdigest()
        stored_path = f"raw/documents/{digest}.txt"
        target = brain.root / stored_path
        _ensure_dir(target.parent)
        target.write_bytes(body)
        _upsert_chunk(
            brain.conn,
            chunk_id=f"{brain_id}:{digest[:16]}:0",
            path=stored_path,
            ordinal=0,
            title=args.url,
            content=body.decode("utf-8", errors="ignore")[:32_000],
            metadata={
                "path": stored_path,
                "brain_id": brain_id,
                "source_url": args.url,
                "mime": mime,
            },
        )
        if progress is not None:
            await progress(1.0, "indexed")
        return {
            "path": "fallback",
            "document": {
                "id": f"{brain_id}:{digest[:16]}",
                "brain_id": brain_id,
                "title": args.url,
                "path": stored_path,
                "source": "ingest",
                "content_type": mime,
                "byte_size": len(body),
                "checksum_sha256": digest,
                "metadata": {"source_url": args.url},
                "commit_sha": digest[:12],
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "deleted_at": None,
            },
        }

    async def extract(
        self, args: ExtractArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        if progress is not None:
            await progress(0.0, "extracting")
        if args.session_id:
            created: list[dict[str, Any]] = []
            for msg in args.messages:
                created.append(
                    {
                        "id": f"msg_{uuid.uuid4().hex}",
                        "session_id": args.session_id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": _now_iso(),
                    }
                )
            if progress is not None:
                await progress(1.0, f"session {len(created)}")
            return {"mode": "session", "messages": created}
        brain = self._open_brain(brain_id)
        transcript_text = "\n\n".join(
            f"[{msg.role}] {msg.content}" for msg in args.messages
        )
        title = f"Transcript {_now_iso()}"
        slug = _slug_from_title(title)
        rel_path = f"transcripts/{slug}.md"
        body_bytes = transcript_text.encode("utf-8")
        written = self._write_note(
            brain,
            relative_path=rel_path,
            content=body_bytes,
            title=title,
            metadata={"source": "extract"},
        )
        if progress is not None:
            await progress(1.0, "transcript")
        return {
            "mode": "transcript",
            "document": {
                "id": written["document_id"],
                "brain_id": brain_id,
                "title": title,
                "path": written["path"],
                "source": "extract",
                "content_type": "text/markdown",
                "byte_size": written["byte_size"],
                "checksum_sha256": written["hash"],
                "metadata": {},
                "commit_sha": written["hash"][:12],
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "deleted_at": None,
            },
        }

    async def reflect(
        self, args: ReflectArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        if progress is not None:
            await progress(0.0, "reflecting")
        # The local SDK does not yet surface a reflection pipeline. Report
        # a graceful no-op so MCP clients see a well-formed payload.
        if progress is not None:
            await progress(1.0, "done")
        return {
            "reflection_status": "no_result",
            "reflection_attempted": False,
            "session_id": args.session_id,
            "ended_at": _now_iso(),
        }

    async def consolidate(
        self, args: ConsolidateArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        await self._ensure_bootstrap()
        brain_id = self._resolve_brain(args.brain)
        if progress is not None:
            await progress(0.0, "consolidating")
        if progress is not None:
            await progress(1.0, "done")
        return {
            "result": {
                "status": "not_implemented",
                "brain_id": brain_id,
                "message": "local consolidate is a no-op in this wrapper",
            }
        }

    async def create_brain(self, args: CreateBrainArgs) -> dict[str, Any]:
        await self._ensure_bootstrap()
        slug = args.slug or _slug_from_title(args.name)
        brain_dir = self._cfg.brain_root / _sanitise_brain_id(slug)
        _ensure_dir(brain_dir)
        config_path = brain_dir / "config.json"
        payload = {
            "version": 1,
            "name": args.name,
            "slug": slug,
            "visibility": args.visibility or "private",
            "createdAt": _now_iso(),
        }
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "id": slug,
            "slug": slug,
            "name": args.name,
            "visibility": payload["visibility"],
            "created_at": payload["createdAt"],
        }

    async def list_brains(self) -> dict[str, Any]:
        await self._ensure_bootstrap()
        if not self._cfg.brain_root.exists():
            return {"items": []}
        items: list[dict[str, Any]] = []
        for entry in sorted(self._cfg.brain_root.iterdir()):
            if not entry.is_dir():
                continue
            config_path = entry / "config.json"
            parsed: dict[str, Any] = {}
            if config_path.exists():
                try:
                    parsed = json.loads(config_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    parsed = {}
            items.append(
                {
                    "id": parsed.get("slug") or entry.name,
                    "slug": parsed.get("slug") or entry.name,
                    "name": parsed.get("name") or entry.name,
                    "visibility": parsed.get("visibility") or "private",
                    "created_at": parsed.get("createdAt"),
                }
            )
        return {"items": items}

    async def close(self) -> None:
        for brain in self._brains.values():
            brain.close()
        self._brains.clear()


# ---------------------------------------------------------------------------
# Hosted mode — httpx against the platform REST API
# ---------------------------------------------------------------------------


class HostedMemoryClient:
    """HTTP implementation of :class:`MemoryClient`."""

    mode: Literal["local", "hosted"] = "hosted"

    def __init__(self, cfg: HostedConfig) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.endpoint.rstrip("/"),
            headers={
                "authorization": f"Bearer {cfg.token}",
                "accept": "application/json",
            },
            timeout=60.0,
        )

    def _resolve_brain(self, override: str | None) -> str:
        if override:
            return override
        if self._cfg.default_brain:
            return self._cfg.default_brain
        raise ValueError(
            "memory-mcp: brain id required in hosted mode; set JB_BRAIN or pass `brain`"
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: dict[str, Any] | None = None,
        files: Any | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        headers: dict[str, str] = {}
        if json_body is not None:
            headers["content-type"] = "application/json"
        resp = await self._client.request(
            method,
            path,
            json=json_body,
            params=params,
            files=files,
            data=data,
            headers=headers,
        )
        if resp.status_code >= 400:
            text = resp.text[:256]
            raise RuntimeError(
                f"memory-mcp: hosted {path} failed {resp.status_code}: {text}"
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    async def remember(self, args: RememberArgs) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        payload: dict[str, Any] = {
            "content": args.content,
        }
        if args.title is not None:
            payload["title"] = args.title
        if args.path is not None:
            payload["path"] = args.path
        if args.tags:
            payload["metadata"] = {"tags": ",".join(args.tags)}
        result = await self._request(
            "POST",
            f"/v1/brains/{brain}/documents",
            json_body=payload,
        )
        return result or {}

    async def search(self, args: SearchArgs) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        params: dict[str, Any] = {"q": args.query}
        if args.top_k is not None:
            params["top_k"] = args.top_k
        if args.scope is not None:
            params["scope"] = args.scope
        if args.sort is not None:
            params["sort"] = args.sort
        result = await self._request(
            "GET", f"/v1/brains/{brain}/search", params=params
        )
        return result or {}

    async def recall(self, args: RecallArgs) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        payload: dict[str, Any] = {"query": args.query}
        if args.scope is not None:
            payload["scope"] = args.scope
        if args.session_id is not None:
            payload["session_id"] = args.session_id
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        result = await self._request(
            "POST", f"/v1/brains/{brain}/recall", json_body=payload
        )
        return result or {}

    async def ask(
        self, args: AskArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        payload: dict[str, Any] = {"query": args.query}
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        # TODO: switch to SSE once the Python hosted transport lands.
        result = await self._request(
            "POST", f"/v1/brains/{brain}/ask", json_body=payload
        )
        return result or {}

    async def ingest_file(
        self, args: IngestFileArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        abs_path = Path(args.path).expanduser().resolve()
        if not abs_path.is_file():
            raise FileNotFoundError(
                f"memory_ingest_file: not a regular file: {abs_path}"
            )
        size = abs_path.stat().st_size
        if size > FILE_LIMIT_BYTES:
            raise ValueError("file_too_large: 25 MiB limit exceeded")
        mime = _mime_for(str(abs_path), args.as_)
        with abs_path.open("rb") as fh:
            files = {"file": (abs_path.name, fh.read(), mime)}
        result = await self._request(
            "POST",
            f"/v1/brains/{brain}/documents/ingest/file",
            files=files,
        )
        return result or {}

    async def ingest_url(
        self, args: IngestUrlArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        result = await self._request(
            "POST",
            f"/v1/brains/{brain}/documents/ingest/url",
            json_body={"url": args.url},
        )
        return {"path": "server", "result": result or {}}

    async def extract(
        self, args: ExtractArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        if args.session_id:
            results: list[Any] = []
            for idx, msg in enumerate(args.messages):
                metadata: dict[str, Any] = {"skip_extract": idx != len(args.messages) - 1}
                if args.actor_id:
                    metadata["actor_id"] = args.actor_id
                payload = {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": metadata,
                }
                result = await self._request(
                    "POST",
                    f"/v1/brains/{brain}/sessions/{args.session_id}/messages",
                    json_body=payload,
                )
                results.append(result)
            return {"mode": "session", "messages": results}
        doc = await self._request(
            "POST",
            f"/v1/brains/{brain}/documents",
            json_body={
                "title": f"Transcript {_now_iso()}",
                "content": "\n\n".join(
                    f"[{msg.role}] {msg.content}" for msg in args.messages
                ),
                "source": "extract",
            },
        )
        return {"mode": "transcript", "document": doc or {}}

    async def reflect(
        self, args: ReflectArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        result = await self._request(
            "POST",
            f"/v1/brains/{brain}/sessions/{args.session_id}/close",
            json_body={},
        )
        return result or {}

    async def consolidate(
        self, args: ConsolidateArgs, progress: ProgressEmitter | None = None
    ) -> dict[str, Any]:
        brain = self._resolve_brain(args.brain)
        result = await self._request(
            "POST",
            f"/v1/brains/{brain}/consolidate",
            json_body={},
        )
        return {"result": result}

    async def create_brain(self, args: CreateBrainArgs) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": args.name}
        if args.slug is not None:
            payload["slug"] = args.slug
        payload["visibility"] = args.visibility or "private"
        result = await self._request("POST", "/v1/brains", json_body=payload)
        return result or {}

    async def list_brains(self) -> dict[str, Any]:
        result = await self._request("GET", "/v1/brains")
        return result or {"items": []}

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_memory_client(cfg: ConfigMode) -> MemoryClient:
    """Build a :class:`MemoryClient` wired to the resolved config."""
    if isinstance(cfg, HostedConfig):
        return HostedMemoryClient(cfg)
    return LocalMemoryClient(cfg)

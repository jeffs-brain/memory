# SPDX-License-Identifier: Apache-2.0
"""HTTP-backed `Store` matching `spec/PROTOCOL.md`.

Uses `httpx.AsyncClient`. Handles Problem+JSON error parsing, bearer
authentication, exponential-backoff retries with `Retry-After` honour,
client-side body size caps, and SSE-based event subscription.
"""

from __future__ import annotations

import asyncio
import base64
import email.utils
import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import quote

import httpx

from ..errors import (
    ErrBadGateway,
    ErrConflict,
    ErrForbidden,
    ErrInternal,
    ErrInvalidPath,
    ErrNotFound,
    ErrPayloadTooLarge,
    ErrRateLimited,
    ErrReadOnly,
    ErrTimeout,
    ErrUnauthorized,
    ErrUnsupportedMedia,
    ErrValidation,
    StoreError,
)
from ..path import BrainPath, is_generated, validate_path
from . import (
    Batch,
    BatchOptions,
    ChangeEvent,
    ChangeKind,
    FileInfo,
    ListOpts,
    Store,
)

log = logging.getLogger(__name__)

_SINGLE_BODY_LIMIT = 2 * 1024 * 1024
_BATCH_BODY_LIMIT = 8 * 1024 * 1024
_BATCH_OP_LIMIT = 1024
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_UA = "jeffs-brain-memory-py/0.1 (+HttpStore)"
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 502, 503, 504}

_CODE_TO_EXC: dict[str, type[StoreError]] = {
    "not_found": ErrNotFound,
    "conflict": ErrConflict,
    "unauthorized": ErrUnauthorized,
    "forbidden": ErrForbidden,
    "validation_error": ErrValidation,
    "payload_too_large": ErrPayloadTooLarge,
    "unsupported_media_type": ErrUnsupportedMedia,
    "rate_limited": ErrRateLimited,
    "internal_error": ErrInternal,
    "bad_gateway": ErrBadGateway,
    "timeout": ErrTimeout,
}


def _status_to_exc(status: int) -> type[StoreError]:
    if status == 404:
        return ErrNotFound
    if status == 400:
        return ErrInvalidPath
    if status == 409:
        return ErrConflict
    if status == 401:
        return ErrUnauthorized
    if status == 403:
        return ErrForbidden
    if status == 413:
        return ErrPayloadTooLarge
    if status == 415:
        return ErrUnsupportedMedia
    if status == 429:
        return ErrRateLimited
    if status == 500:
        return ErrInternal
    if status == 502:
        return ErrBadGateway
    if status == 504:
        return ErrTimeout
    return StoreError


class HttpStore(Store):
    """Store that talks to a remote `memory serve` daemon."""

    def __init__(
        self,
        base_url: str,
        brain_id: str,
        *,
        token: str | None = None,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT,
        user_agent: str = _DEFAULT_UA,
        max_retries: int = _MAX_RETRIES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not brain_id:
            raise ValueError("HttpStore: brain_id is required")
        self.base_url = base_url.rstrip("/")
        self.brain_id = brain_id
        self.token = token or api_key
        self.user_agent = user_agent
        self.max_retries = max(0, max_retries)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._closed = False
        self._sinks: dict[int, Callable[[ChangeEvent], None]] = {}
        self._next_id = 0
        self._sse_task: asyncio.Task[None] | None = None
        self._event_queues: list[asyncio.Queue[ChangeEvent]] = []

    # --- URL helpers ----------------------------------------------------

    def _brain_path(self, suffix: str) -> str:
        return f"/v1/brains/{quote(self.brain_id, safe='')}{suffix}"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self, *, accept: str = "application/json", content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"accept": accept, "user-agent": self.user_agent}
        if content_type:
            headers["content-type"] = content_type
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return headers

    # --- low-level request pipeline -------------------------------------

    async def _do(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: bytes | None = None,
        json_body: Any = None,
        accept: str = "application/json",
        content_type: str | None = None,
    ) -> httpx.Response:
        if self._closed:
            raise ErrReadOnly("HttpStore: closed")
        if json_body is not None and body is None:
            body = json.dumps(json_body).encode()
            if content_type is None:
                content_type = "application/json"
        headers = self._headers(accept=accept, content_type=content_type)
        attempt = 0
        last_response: httpx.Response | None = None
        while True:
            response = await self._client.request(
                method,
                self._url(path),
                params=params,
                content=body,
                headers=headers,
            )
            last_response = response
            if response.status_code not in _RETRY_STATUSES or attempt >= self.max_retries:
                return response
            wait = _retry_backoff(response, attempt)
            attempt += 1
            await asyncio.sleep(wait)

    # --- read side ------------------------------------------------------

    async def read(self, path: BrainPath) -> bytes:
        validate_path(str(path))
        response = await self._do(
            "GET",
            self._brain_path("/documents/read"),
            params={"path": str(path)},
            accept="application/octet-stream",
        )
        if response.status_code != 200:
            raise _response_error(response)
        return response.content

    async def exists(self, path: BrainPath) -> bool:
        validate_path(str(path))
        response = await self._do(
            "HEAD",
            self._brain_path("/documents"),
            params={"path": str(path)},
        )
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        raise _response_error(response)

    async def stat(self, path: BrainPath) -> FileInfo:
        validate_path(str(path))
        response = await self._do(
            "GET",
            self._brain_path("/documents/stat"),
            params={"path": str(path)},
        )
        if response.status_code != 200:
            raise _response_error(response)
        raw = response.json()
        return _file_info_from_raw(raw)

    async def list(
        self,
        dir: BrainPath | str = "",
        opts: ListOpts | None = None,
    ) -> list[FileInfo]:
        opts = opts or ListOpts()
        params: dict[str, str] = {
            "dir": str(dir),
            "recursive": "true" if opts.recursive else "false",
            "include_generated": "true" if opts.include_generated else "false",
        }
        if opts.glob:
            params["glob"] = opts.glob
        response = await self._do(
            "GET",
            self._brain_path("/documents"),
            params=params,
        )
        if response.status_code != 200:
            raise _response_error(response)
        body = response.json()
        return [_file_info_from_raw(it) for it in body.get("items", [])]

    # --- write side -----------------------------------------------------

    async def write(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        if len(content) > _SINGLE_BODY_LIMIT:
            raise ErrPayloadTooLarge(f"HttpStore: write {path}: exceeds {_SINGLE_BODY_LIMIT} bytes")
        response = await self._do(
            "PUT",
            self._brain_path("/documents"),
            params={"path": str(path)},
            body=content,
            content_type="application/octet-stream",
        )
        if response.status_code not in (200, 204):
            raise _response_error(response)

    async def append(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        if len(content) > _SINGLE_BODY_LIMIT:
            raise ErrPayloadTooLarge(
                f"HttpStore: append {path}: exceeds {_SINGLE_BODY_LIMIT} bytes"
            )
        response = await self._do(
            "POST",
            self._brain_path("/documents/append"),
            params={"path": str(path)},
            body=content,
            content_type="application/octet-stream",
        )
        if response.status_code not in (200, 204):
            raise _response_error(response)

    async def delete(self, path: BrainPath) -> None:
        validate_path(str(path))
        response = await self._do(
            "DELETE",
            self._brain_path("/documents"),
            params={"path": str(path)},
        )
        if response.status_code not in (200, 204):
            raise _response_error(response)

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        validate_path(str(src))
        validate_path(str(dst))
        response = await self._do(
            "POST",
            self._brain_path("/documents/rename"),
            json_body={"from": str(src), "to": str(dst)},
            content_type="application/json",
        )
        if response.status_code not in (200, 204):
            raise _response_error(response)

    # --- batch ----------------------------------------------------------

    async def batch(
        self,
        fn: Callable[[Batch], Awaitable[None]] | Callable[[Batch], None],
        opts: BatchOptions | None = None,
    ) -> None:
        opts = opts or BatchOptions()
        b = _HttpBatch(self)
        result = fn(b)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        await b.commit(opts)

    # --- subscribe / close ----------------------------------------------

    def subscribe(self, sink: Callable[[ChangeEvent], None]) -> Callable[[], None]:
        self._next_id += 1
        sink_id = self._next_id
        self._sinks[sink_id] = sink
        if self._sse_task is None:
            self._sse_task = asyncio.create_task(self._run_sse())

        def unsubscribe() -> None:
            self._sinks.pop(sink_id, None)
            if not self._sinks and self._sse_task is not None:
                self._sse_task.cancel()
                self._sse_task = None

        return unsubscribe

    def events(self) -> AsyncIterator[ChangeEvent]:
        queue: asyncio.Queue[ChangeEvent] = asyncio.Queue()
        self._event_queues.append(queue)
        if self._sse_task is None:
            self._sse_task = asyncio.create_task(self._run_sse())

        async def iterator() -> AsyncIterator[ChangeEvent]:
            try:
                while True:
                    evt = await queue.get()
                    yield evt
            finally:
                try:
                    self._event_queues.remove(queue)
                except ValueError:
                    pass

        return iterator()

    async def _run_sse(self) -> None:
        url = self._url(self._brain_path("/events"))
        headers = self._headers(accept="text/event-stream")
        try:
            async with self._client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    log.warning("HttpStore: SSE attach failed: %s", response.status_code)
                    return
                event_name = ""
                data_lines: list[str] = []
                async for raw_line in response.aiter_lines():
                    if raw_line == "":
                        if event_name or data_lines:
                            self._dispatch_sse(event_name, "\n".join(data_lines))
                        event_name = ""
                        data_lines = []
                        continue
                    if raw_line.startswith(":"):
                        continue
                    field, _, value = raw_line.partition(":")
                    value = value.lstrip(" ")
                    if field == "event":
                        event_name = value
                    elif field == "data":
                        data_lines.append(value)
        except (httpx.HTTPError, asyncio.CancelledError):
            return

    def _dispatch_sse(self, event: str, data: str) -> None:
        if event != "change" or not data:
            return
        try:
            raw = json.loads(data)
        except json.JSONDecodeError:
            return
        try:
            when = datetime.fromisoformat(raw.get("when", "").replace("Z", "+00:00"))
        except (ValueError, TypeError):
            when = datetime.now(timezone.utc)
        try:
            kind = ChangeKind(raw.get("kind", ""))
        except ValueError:
            return
        evt = ChangeEvent(
            kind=kind,
            path=BrainPath(raw.get("path", "")),
            when=when,
            old_path=BrainPath(raw["old_path"]) if raw.get("old_path") else None,
            reason=raw.get("reason"),
        )
        for sink in list(self._sinks.values()):
            try:
                sink(evt)
            except Exception:
                pass
        for queue in list(self._event_queues):
            queue.put_nowait(evt)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sse_task is not None:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sse_task = None
        self._sinks.clear()
        if self._owns_client:
            await self._client.aclose()

    def local_path(self, path: BrainPath) -> str | None:
        return None


# ---------- wire helpers -------------------------------------------------


def _file_info_from_raw(raw: dict[str, Any]) -> FileInfo:
    mtime: datetime | None = None
    m = raw.get("mtime")
    if m:
        try:
            mtime = datetime.fromisoformat(m.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            mtime = None
    return FileInfo(
        path=BrainPath(raw.get("path", "")),
        size=int(raw.get("size", 0)),
        mtime=mtime,
        is_dir=bool(raw.get("is_dir", False)),
    )


def _response_error(response: httpx.Response) -> StoreError:
    status = response.status_code
    title = ""
    detail = ""
    code = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            title = str(body.get("title", ""))
            detail = str(body.get("detail", ""))
            code = str(body.get("code", ""))
            status = int(body.get("status", status))
    except (json.JSONDecodeError, ValueError):
        pass
    exc_cls: type[StoreError] = _CODE_TO_EXC.get(code) or _status_to_exc(status)
    message = f"HttpStore: {status} {title or ''}"
    if detail:
        message += f": {detail}"
    return exc_cls(message)


def _retry_backoff(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            seconds = int(retry_after)
            return max(0.0, float(seconds)) + random.random() * 0.1
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(retry_after)
                now = datetime.now(timezone.utc)
                delta = (parsed.astimezone(timezone.utc) - now).total_seconds()
                return max(0.0, delta) + random.random() * 0.1
            except (ValueError, TypeError):
                pass
    base = min(4.0, 0.1 * (1 << attempt))
    return base + random.random() * 0.1


# ---------- batch ---------------------------------------------------------


class _HttpBatch(Batch):
    """Buffers mutations and sends them as a batch-ops POST on commit."""

    def __init__(self, store: HttpStore) -> None:
        self.store = store
        self.journal: list[dict[str, Any]] = []

    def _replay(self, path: BrainPath) -> tuple[str, bytes | None]:
        """Return (state, content) for path. state: untouched | present | deleted."""
        touched = False
        present = False
        content: bytes = b""
        for op in self.journal:
            kind = op["kind"]
            if kind == "write" and op["path"] == path:
                present = True
                content = op["content"]
                touched = True
            elif kind == "append" and op["path"] == path:
                base_content = content if present else b""
                content = base_content + op["content"]
                present = True
                touched = True
            elif kind == "delete" and op["path"] == path:
                present = False
                content = b""
                touched = True
            elif kind == "rename":
                if op["src"] == path:
                    present = False
                    content = b""
                    touched = True
                elif op["path"] == path:
                    present = True
                    touched = True
        if not touched:
            return ("untouched", None)
        return ("present" if present else "deleted", content if present else None)

    async def read(self, path: BrainPath) -> bytes:
        validate_path(str(path))
        state, content = self._replay(path)
        if state == "present":
            assert content is not None
            return bytes(content)
        if state == "deleted":
            raise ErrNotFound(f"HttpStore: read {path}: not found")
        return await self.store.read(path)

    async def exists(self, path: BrainPath) -> bool:
        validate_path(str(path))
        state, _ = self._replay(path)
        if state == "present":
            return True
        if state == "deleted":
            return False
        return await self.store.exists(path)

    async def stat(self, path: BrainPath) -> FileInfo:
        validate_path(str(path))
        state, content = self._replay(path)
        if state == "present":
            assert content is not None
            return FileInfo(
                path=path,
                size=len(content),
                mtime=datetime.now(timezone.utc),
                is_dir=False,
            )
        if state == "deleted":
            raise ErrNotFound(f"HttpStore: stat {path}: not found")
        return await self.store.stat(path)

    async def list(
        self, dir: BrainPath | str = "", opts: ListOpts | None = None
    ) -> list[FileInfo]:
        opts = opts or ListOpts()
        base = await self.store.list(dir, opts)
        by_path: dict[BrainPath, FileInfo] = {fi.path: fi for fi in base}
        touched: set[BrainPath] = set()
        for op in self.journal:
            if op["kind"] == "rename":
                touched.add(op["src"])
                touched.add(op["path"])
            else:
                touched.add(op["path"])
        for p in touched:
            from ._util import path_under

            if not path_under(str(p), str(dir), opts.recursive):
                continue
            state, content = self._replay(p)
            if state == "present":
                if not opts.include_generated and is_generated(p):
                    by_path.pop(p, None)
                    continue
                if opts.glob:
                    import fnmatch

                    if not fnmatch.fnmatchcase(str(p).rsplit("/", 1)[-1], opts.glob):
                        continue
                assert content is not None
                by_path[p] = FileInfo(
                    path=p,
                    size=len(content),
                    mtime=datetime.now(timezone.utc),
                    is_dir=False,
                )
            elif state == "deleted":
                by_path.pop(p, None)
        out = list(by_path.values())
        out.sort(key=lambda fi: fi.path)
        return out

    async def write(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        self.journal.append(
            {"kind": "write", "path": BrainPath(str(path)), "content": bytes(content), "src": None}
        )

    async def append(self, path: BrainPath, content: bytes) -> None:
        validate_path(str(path))
        state, pending = self._replay(path)
        base: bytes = b""
        if state == "present":
            assert pending is not None
            base = pending
        elif state == "deleted":
            base = b""
        else:
            try:
                base = await self.store.read(path)
            except ErrNotFound:
                base = b""
        self.journal.append(
            {
                "kind": "write",
                "path": BrainPath(str(path)),
                "content": base + bytes(content),
                "src": None,
            }
        )

    async def delete(self, path: BrainPath) -> None:
        validate_path(str(path))
        state, _ = self._replay(path)
        if state == "present":
            self.journal.append({"kind": "delete", "path": BrainPath(str(path)), "src": None})
            return
        if state == "deleted":
            raise ErrNotFound(f"HttpStore: delete {path}: not found")
        if not await self.store.exists(path):
            raise ErrNotFound(f"HttpStore: delete {path}: not found")
        self.journal.append({"kind": "delete", "path": BrainPath(str(path)), "src": None})

    async def rename(self, src: BrainPath, dst: BrainPath) -> None:
        validate_path(str(src))
        validate_path(str(dst))
        state, pending = self._replay(src)
        payload: bytes
        if state == "present":
            assert pending is not None
            payload = pending
        elif state == "deleted":
            raise ErrNotFound(f"HttpStore: rename {src}: not found")
        else:
            payload = await self.store.read(src)
        self.journal.append(
            {
                "kind": "write",
                "path": BrainPath(str(dst)),
                "content": payload,
                "src": None,
            }
        )
        self.journal.append({"kind": "delete", "path": BrainPath(str(src)), "src": None})

    async def commit(self, opts: BatchOptions) -> None:
        if not self.journal:
            return
        if len(self.journal) > _BATCH_OP_LIMIT:
            raise ErrPayloadTooLarge(
                f"HttpStore: batch has {len(self.journal)} ops (cap {_BATCH_OP_LIMIT})"
            )
        ops_wire: list[dict[str, Any]] = []
        total = 0
        for op in self.journal:
            content = op.get("content")
            if isinstance(content, (bytes, bytearray)):
                total += len(content)
                if total > _BATCH_BODY_LIMIT:
                    raise ErrPayloadTooLarge(
                        f"HttpStore: batch payload exceeds {_BATCH_BODY_LIMIT} bytes"
                    )
            if op["kind"] == "write":
                ops_wire.append(
                    {
                        "type": "write",
                        "path": str(op["path"]),
                        "content_base64": base64.b64encode(content or b"").decode(),
                    }
                )
            elif op["kind"] == "append":
                ops_wire.append(
                    {
                        "type": "append",
                        "path": str(op["path"]),
                        "content_base64": base64.b64encode(content or b"").decode(),
                    }
                )
            elif op["kind"] == "delete":
                ops_wire.append({"type": "delete", "path": str(op["path"])})
            elif op["kind"] == "rename":
                ops_wire.append(
                    {
                        "type": "rename",
                        "path": str(op["src"]),
                        "to": str(op["path"]),
                    }
                )
        payload: dict[str, Any] = {"ops": ops_wire}
        if opts.reason:
            payload["reason"] = opts.reason
        if opts.message:
            payload["message"] = opts.message
        if opts.author:
            payload["author"] = opts.author
        if opts.email:
            payload["email"] = opts.email
        response = await self.store._do(
            "POST",
            self.store._brain_path("/documents/batch-ops"),
            json_body=payload,
            content_type="application/json",
        )
        if response.status_code != 200:
            raise _response_error(response)

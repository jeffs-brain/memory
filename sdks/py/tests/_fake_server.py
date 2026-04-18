# SPDX-License-Identifier: Apache-2.0
"""Minimal Starlette server mirroring `spec/PROTOCOL.md` for tests."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from jeffs_brain_memory.errors import ErrInvalidPath
from jeffs_brain_memory.path import BrainPath, validate_path
from jeffs_brain_memory.store.mem import MemStore


def _problem(status: int, code: str, title: str, detail: str = "") -> JSONResponse:
    payload: dict[str, Any] = {"status": status, "title": title, "code": code}
    if detail:
        payload["detail"] = detail
    return JSONResponse(payload, status_code=status, media_type="application/problem+json")


def _require_path(request: Request) -> str:
    p = request.query_params.get("path", "")
    try:
        return validate_path(p)
    except ErrInvalidPath as exc:
        raise _ProblemError(400, "validation_error", "Invalid path", str(exc)) from exc


class _ProblemError(Exception):
    def __init__(self, status: int, code: str, title: str, detail: str = "") -> None:
        super().__init__(detail or title)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail


def _brain(request: Request) -> MemStore:
    brain_id = request.path_params["brain_id"]
    stores: dict[str, MemStore] = request.app.state.stores
    return stores.setdefault(brain_id, MemStore())


def _broker(request: Request) -> "_EventBroker":
    brain_id = request.path_params["brain_id"]
    brokers: dict[str, _EventBroker] = request.app.state.brokers
    broker = brokers.get(brain_id)
    if broker is None:
        broker = _EventBroker()
        brokers[brain_id] = broker
        store = _brain(request)

        def on_change(evt):
            broker.publish(evt)

        store.subscribe(on_change)
    return broker


class _EventBroker:
    def __init__(self) -> None:
        self.queues: list[asyncio.Queue[Any]] = []

    def attach(self) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue()
        self.queues.append(q)
        return q

    def detach(self, q: asyncio.Queue[Any]) -> None:
        try:
            self.queues.remove(q)
        except ValueError:
            pass

    def publish(self, evt) -> None:
        payload = {
            "kind": evt.kind.value,
            "path": str(evt.path),
            "when": evt.when.isoformat(),
        }
        if evt.old_path:
            payload["old_path"] = str(evt.old_path)
        if evt.reason:
            payload["reason"] = evt.reason
        for q in list(self.queues):
            q.put_nowait(payload)


async def read_doc(request: Request) -> Response:
    try:
        path = _require_path(request)
    except _ProblemError as err:
        return _problem(err.status, err.code, err.title, err.detail)
    try:
        content = await _brain(request).read(BrainPath(path))
    except Exception:
        return _problem(404, "not_found", "Not Found")
    return Response(content, media_type="application/octet-stream")


async def put_doc(request: Request) -> Response:
    try:
        path = _require_path(request)
    except _ProblemError as err:
        return _problem(err.status, err.code, err.title, err.detail)
    body = await request.body()
    await _brain(request).write(BrainPath(path), body)
    return Response(status_code=204)


async def delete_doc(request: Request) -> Response:
    try:
        path = _require_path(request)
    except _ProblemError as err:
        return _problem(err.status, err.code, err.title, err.detail)
    try:
        await _brain(request).delete(BrainPath(path))
    except Exception:
        return _problem(404, "not_found", "Not Found")
    return Response(status_code=204)


async def append_doc(request: Request) -> Response:
    try:
        path = _require_path(request)
    except _ProblemError as err:
        return _problem(err.status, err.code, err.title, err.detail)
    body = await request.body()
    await _brain(request).append(BrainPath(path), body)
    return Response(status_code=204)


async def rename_doc(request: Request) -> Response:
    body = await request.json()
    src = body.get("from", "")
    dst = body.get("to", "")
    try:
        validate_path(src)
        validate_path(dst)
    except ErrInvalidPath as exc:
        return _problem(400, "validation_error", "Invalid path", str(exc))
    try:
        await _brain(request).rename(BrainPath(src), BrainPath(dst))
    except Exception:
        return _problem(404, "not_found", "Not Found")
    return Response(status_code=204)


async def stat_doc(request: Request) -> Response:
    try:
        path = _require_path(request)
    except _ProblemError as err:
        return _problem(err.status, err.code, err.title, err.detail)
    try:
        info = await _brain(request).stat(BrainPath(path))
    except Exception:
        return _problem(404, "not_found", "Not Found")
    return JSONResponse(
        {
            "path": str(info.path),
            "size": info.size,
            "mtime": info.mtime.isoformat() if info.mtime else "",
            "is_dir": info.is_dir,
        }
    )


async def list_docs(request: Request) -> Response:
    from jeffs_brain_memory.store import ListOpts

    qp = request.query_params
    # A HEAD request with a `path` query param is an existence probe.
    if request.method == "HEAD":
        try:
            path = _require_path(request)
        except _ProblemError as err:
            return _problem(err.status, err.code, err.title, err.detail)
        if await _brain(request).exists(BrainPath(path)):
            return Response(status_code=200)
        return Response(status_code=404)
    if "path" in qp and "dir" not in qp:
        # Some clients use GET /documents?path=... as the existence check.
        try:
            path = _require_path(request)
        except _ProblemError as err:
            return _problem(err.status, err.code, err.title, err.detail)
        if await _brain(request).exists(BrainPath(path)):
            return Response(status_code=200)
        return Response(status_code=404)
    opts = ListOpts(
        recursive=qp.get("recursive", "false") == "true",
        include_generated=qp.get("include_generated", "false") == "true",
        glob=qp.get("glob") or None,
    )
    items = await _brain(request).list(qp.get("dir", ""), opts)
    payload = {
        "items": [
            {
                "path": str(fi.path),
                "size": fi.size,
                "mtime": fi.mtime.isoformat() if fi.mtime else "",
                "is_dir": fi.is_dir,
            }
            for fi in items
        ]
    }
    return JSONResponse(payload)


async def batch_ops(request: Request) -> Response:
    from jeffs_brain_memory.store import BatchOptions

    body = await request.json()
    ops = body.get("ops") or []
    reason = body.get("reason")
    store = _brain(request)
    materialised: list[tuple[str, str, bytes, str | None]] = []
    for op in ops:
        kind = op["type"]
        content = base64.b64decode(op["content_base64"]) if "content_base64" in op else b""
        src = op.get("to")
        materialised.append((kind, op["path"], content, src))

    async def do_work(b):
        for kind, path, content, to_path in materialised:
            if kind == "write":
                await b.write(BrainPath(path), content)
            elif kind == "append":
                await b.append(BrainPath(path), content)
            elif kind == "delete":
                await b.delete(BrainPath(path))
            elif kind == "rename":
                await b.rename(BrainPath(path), BrainPath(to_path or ""))

    try:
        await store.batch(do_work, BatchOptions(reason=reason))
    except Exception as exc:
        return _problem(400, "validation_error", "Batch failed", str(exc))
    return JSONResponse({"committed": len(ops)})


async def events(request: Request) -> StreamingResponse:
    broker = _broker(request)
    queue = broker.attach()

    async def gen():
        # Initial ready frame.
        yield "event: ready\ndata: {}\n\n"
        try:
            while True:
                payload = await queue.get()
                yield f"event: change\ndata: {json.dumps(payload)}\n\n"
        finally:
            broker.detach(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


def build_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/v1/brains/{brain_id}/documents/read", read_doc, methods=["GET"]),
            Route("/v1/brains/{brain_id}/documents/stat", stat_doc, methods=["GET"]),
            Route("/v1/brains/{brain_id}/documents/append", append_doc, methods=["POST"]),
            Route("/v1/brains/{brain_id}/documents/rename", rename_doc, methods=["POST"]),
            Route("/v1/brains/{brain_id}/documents/batch-ops", batch_ops, methods=["POST"]),
            Route("/v1/brains/{brain_id}/documents", list_docs, methods=["GET", "HEAD"]),
            Route("/v1/brains/{brain_id}/documents", put_doc, methods=["PUT"]),
            Route("/v1/brains/{brain_id}/documents", delete_doc, methods=["DELETE"]),
            Route("/v1/brains/{brain_id}/events", events, methods=["GET"]),
        ]
    )
    app.state.stores = {}
    app.state.brokers = {}
    return app

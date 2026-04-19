# SPDX-License-Identifier: Apache-2.0
"""Document CRUD handlers (read / stat / list / put / append / delete / rename / batch)."""

from __future__ import annotations

import base64
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ...errors import ErrInvalidPath
from ..daemon import StoreNotFound
from ..problem import (
    internal_error,
    not_found,
    payload_too_large,
    validation_error,
)
from ._shared import (
    decode_json_body,
    ok_json,
    read_body_limited,
    resolve_brain,
    wrap_invalid_path,
)

# Body size caps from spec/PROTOCOL.md.
DOC_BODY_LIMIT = 2 * 1024 * 1024
BATCH_BODY_LIMIT = 8 * 1024 * 1024
BATCH_OP_LIMIT = 1024


def _bool_query(value: str | None) -> bool:
    return (value or "").lower() == "true"


def _file_info_to_wire(fi: Any) -> dict[str, Any]:
    return {
        "path": fi.path,
        "size": fi.size,
        "mtime": fi.mtime.isoformat(),
        "is_dir": fi.is_dir,
    }


async def doc_read(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    try:
        data = await br.store.read(path)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except StoreNotFound:
        return not_found(f"not found: {path}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return Response(
        content=data,
        status_code=200,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


async def doc_stat(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    try:
        info = await br.store.stat(path)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except StoreNotFound:
        return not_found(f"not found: {path}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return ok_json(_file_info_to_wire(info))


async def doc_head(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    try:
        exists = await br.store.exists(path)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    if not exists:
        return Response(status_code=404)
    return Response(status_code=200, headers={"Cache-Control": "no-store"})


async def doc_list(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    q = request.query_params
    directory = q.get("dir") or ""
    recursive = _bool_query(q.get("recursive"))
    include_generated = _bool_query(q.get("include_generated"))
    glob = q.get("glob") or None
    try:
        entries = await br.store.list(
            directory,
            recursive=recursive,
            glob=glob,
            include_generated=include_generated,
        )
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return ok_json({"items": [_file_info_to_wire(e) for e in entries]})


async def doc_put(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    body = await read_body_limited(request, DOC_BODY_LIMIT)
    if isinstance(body, Response):
        return body
    try:
        await br.store.write(path, body)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return Response(status_code=204)


async def doc_append(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    body = await read_body_limited(request, DOC_BODY_LIMIT)
    if isinstance(body, Response):
        return body
    try:
        await br.store.append(path, body)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return Response(status_code=204)


async def doc_delete(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    path = request.query_params.get("path") or ""
    try:
        await br.store.delete(path)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except StoreNotFound:
        return not_found(f"not found: {path}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return Response(status_code=204)


async def doc_rename(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 64 * 1024)
    if isinstance(body, Response):
        return body
    src = body.get("from")
    dst = body.get("to")
    if not isinstance(src, str) or not isinstance(dst, str):
        return validation_error("from and to required")
    try:
        await br.store.rename(src, dst)
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except StoreNotFound:
        return not_found(f"not found: {src}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return Response(status_code=204)


async def doc_batch(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    # Accept 2x BATCH_BODY_LIMIT to absorb base64 expansion + JSON
    # framing; we reject after the decoded payload exceeds the real cap.
    raw = await read_body_limited(request, BATCH_BODY_LIMIT * 2)
    if isinstance(raw, Response):
        return raw
    if not raw:
        return validation_error("empty body")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return validation_error(f"invalid JSON: {exc}")
    if not isinstance(body, dict):
        return validation_error("expected JSON object")
    ops = body.get("ops")
    if not isinstance(ops, list):
        return validation_error("ops required")
    if len(ops) > BATCH_OP_LIMIT:
        return payload_too_large(f"ops length exceeds {BATCH_OP_LIMIT}")

    decoded_size = 0
    prepared: list[dict[str, Any]] = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            return validation_error(f"op at index {i} must be an object")
        prep = dict(op)
        content_b64 = op.get("content_base64")
        if isinstance(content_b64, str) and content_b64:
            try:
                decoded = base64.b64decode(content_b64, validate=True)
            except (ValueError, Exception) as exc:  # noqa: BLE001
                return validation_error(f"invalid base64 at op {i}: {exc}")
            decoded_size += len(decoded)
            if decoded_size > BATCH_BODY_LIMIT:
                return payload_too_large(
                    "batch payload exceeds 8 MiB after decode"
                )
            prep["_decoded"] = decoded
        else:
            prep["_decoded"] = b""
        prepared.append(prep)

    try:
        committed = await br.store.batch(prepared, reason=body.get("reason"))
    except ErrInvalidPath as exc:
        return wrap_invalid_path(exc)
    except ValueError as exc:
        return validation_error(str(exc))
    except StoreNotFound:
        return not_found("batch: path not found")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return ok_json({"committed": committed})


async def doc_list_or_head(request: Request) -> Response:
    """Dispatch GET vs HEAD since Starlette registers both methods on
    the same route."""
    if request.method == "HEAD":
        return await doc_head(request)
    return await doc_list(request)

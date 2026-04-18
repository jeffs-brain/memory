# SPDX-License-Identifier: Apache-2.0
"""Shared helpers used across handler modules."""

from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ...errors import ErrInvalidPath
from ..daemon import BrainNotFound, BrainResources, Daemon
from ..problem import (
    internal_error,
    not_found,
    payload_too_large,
    problem_response,
    validation_error,
)


def get_daemon(request: Request) -> Daemon:
    daemon: Daemon | None = request.app.state.daemon  # type: ignore[attr-defined]
    if daemon is None:
        raise RuntimeError("daemon not configured on app.state")
    return daemon


async def resolve_brain(request: Request) -> BrainResources | Response:
    """Look up the brain referenced by `{brain_id}` in the path.

    Returns a Problem+JSON response when the brain is missing; otherwise
    the cached resources. Handlers short-circuit on Response.
    """
    daemon = get_daemon(request)
    brain_id = request.path_params.get("brain_id") or ""
    if not brain_id:
        return validation_error("missing brainId")
    try:
        return await daemon.brains.get(brain_id)
    except BrainNotFound:
        return not_found(f"brain not found: {brain_id}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))


async def read_body_limited(request: Request, limit: int) -> bytes | Response:
    """Stream the body up to `limit` bytes; return 413 on overflow."""
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        if not chunk:
            continue
        size += len(chunk)
        if size > limit:
            return payload_too_large(f"body exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


async def decode_json_body(
    request: Request,
    limit: int,
) -> dict[str, Any] | Response:
    body = await read_body_limited(request, limit)
    if isinstance(body, Response):
        return body
    if not body:
        return validation_error("empty body")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return validation_error(f"invalid JSON: {exc}")
    if not isinstance(parsed, dict):
        return validation_error("expected JSON object")
    return parsed


def ok_json(payload: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status)


def wrap_invalid_path(exc: ErrInvalidPath) -> JSONResponse:
    return validation_error(str(exc))


__all__ = [
    "decode_json_body",
    "get_daemon",
    "ok_json",
    "read_body_limited",
    "resolve_brain",
    "wrap_invalid_path",
]

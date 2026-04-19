# SPDX-License-Identifier: Apache-2.0
"""Brain-management handlers: list / create / get / delete."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..daemon import BrainConflict, BrainNotFound
from ..problem import (
    conflict,
    internal_error,
    not_found,
    precondition_required,
    validation_error,
)
from ._shared import decode_json_body, get_daemon, ok_json


async def list_brains(request: Request) -> Response:
    daemon = get_daemon(request)
    try:
        ids = await daemon.brains.list()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return ok_json({"items": [{"brainId": i} for i in ids]})


async def get_brain(request: Request) -> Response:
    daemon = get_daemon(request)
    brain_id = request.path_params.get("brain_id") or ""
    if not brain_id:
        return validation_error("missing brainId")
    if not daemon.brains._exists_on_disk(brain_id):  # noqa: SLF001 - small helper
        return not_found(f"brain not found: {brain_id}")
    return ok_json({"brainId": brain_id})


async def create_brain(request: Request) -> Response:
    daemon = get_daemon(request)
    body = await decode_json_body(request, 64 * 1024)
    if isinstance(body, Response):
        return body
    brain_id = body.get("brainId")
    if not isinstance(brain_id, str) or not brain_id:
        return validation_error("brainId required")
    description = body.get("description") or ""
    try:
        await daemon.brains.create(brain_id)
    except BrainConflict:
        return conflict(f"brain already exists: {brain_id}")
    except ValueError as exc:
        return validation_error(str(exc))
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    response = {"brainId": brain_id}
    if description:
        response["description"] = description
    return ok_json(response, status=201)


async def delete_brain(request: Request) -> Response:
    daemon = get_daemon(request)
    brain_id = request.path_params.get("brain_id") or ""
    if not brain_id:
        return validation_error("missing brainId")
    if request.headers.get("X-Confirm-Delete") != "yes":
        return precondition_required(
            "delete brain requires X-Confirm-Delete: yes header"
        )
    try:
        await daemon.brains.delete(brain_id)
    except BrainNotFound:
        return not_found(f"brain not found: {brain_id}")
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return JSONResponse(None, status_code=204)

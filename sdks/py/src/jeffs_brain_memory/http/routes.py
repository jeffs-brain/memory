# SPDX-License-Identifier: Apache-2.0
"""Route handlers matching `spec/PROTOCOL.md`.

All real endpoints return 501 Problem+JSON during the scaffold phase.
`/healthz` is the only fully implemented endpoint.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .problem import not_implemented


async def healthz(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def documents_read(_: Request) -> Response:
    return not_implemented("GET /v1/brains/{brainId}/documents/read")


async def documents_head(_: Request) -> Response:
    return not_implemented("HEAD /v1/brains/{brainId}/documents")


async def documents_stat(_: Request) -> Response:
    return not_implemented("GET /v1/brains/{brainId}/documents/stat")


async def documents_list(_: Request) -> Response:
    return not_implemented("GET /v1/brains/{brainId}/documents")


async def documents_put(_: Request) -> Response:
    return not_implemented("PUT /v1/brains/{brainId}/documents")


async def documents_append(_: Request) -> Response:
    return not_implemented("POST /v1/brains/{brainId}/documents/append")


async def documents_delete(_: Request) -> Response:
    return not_implemented("DELETE /v1/brains/{brainId}/documents")


async def documents_rename(_: Request) -> Response:
    return not_implemented("POST /v1/brains/{brainId}/documents/rename")


async def documents_batch_ops(_: Request) -> Response:
    return not_implemented("POST /v1/brains/{brainId}/documents/batch-ops")


async def events_sse(_: Request) -> Response:
    return not_implemented("GET /v1/brains/{brainId}/events")


def build_routes() -> list[Route]:
    """Route table matching `spec/PROTOCOL.md`."""
    prefix = "/v1/brains/{brain_id}"
    return [
        Route("/healthz", healthz, methods=["GET"]),
        Route(f"{prefix}/documents/read", documents_read, methods=["GET"]),
        Route(f"{prefix}/documents/stat", documents_stat, methods=["GET"]),
        Route(f"{prefix}/documents/append", documents_append, methods=["POST"]),
        Route(f"{prefix}/documents/rename", documents_rename, methods=["POST"]),
        Route(f"{prefix}/documents/batch-ops", documents_batch_ops, methods=["POST"]),
        Route(f"{prefix}/documents", documents_head, methods=["HEAD"]),
        Route(f"{prefix}/documents", documents_list, methods=["GET"]),
        Route(f"{prefix}/documents", documents_put, methods=["PUT"]),
        Route(f"{prefix}/documents", documents_delete, methods=["DELETE"]),
        Route(f"{prefix}/events", events_sse, methods=["GET"]),
    ]

# SPDX-License-Identifier: Apache-2.0
"""Ingest handlers that delegate to the knowledge base for chunking + index."""

from __future__ import annotations

import base64

from starlette.requests import Request
from starlette.responses import Response

from ...knowledge import IngestRequest
from ..problem import internal_error, validation_error
from ._shared import decode_json_body, ok_json, resolve_brain


async def ingest_file(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 8 * 1024 * 1024)
    if isinstance(body, Response):
        return body

    path = body.get("path")
    if not isinstance(path, str) or not path:
        return validation_error("path required")

    title = body.get("title") or ""
    tags_raw = body.get("tags") or []
    content_type = body.get("contentType") or ""
    content_b64 = body.get("contentBase64")

    ireq = IngestRequest(
        brain_id=br.id,
        path=path,
        content_type=content_type,
        title=title if isinstance(title, str) else "",
        tags=list(tags_raw) if isinstance(tags_raw, list) else [],
    )

    if isinstance(content_b64, str) and content_b64:
        try:
            raw = base64.b64decode(content_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            return validation_error(f"invalid contentBase64: {exc}")
        ireq.content = raw

    try:
        resp = await br.knowledge_base.ingest(ireq)
    except FileNotFoundError as exc:
        return validation_error(str(exc))
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))

    return ok_json(
        {
            "documentId": str(resp.document_id),
            "path": str(resp.path),
            "chunkCount": resp.chunk_count,
            "bytes": resp.bytes,
            "tookMs": resp.took_ms,
        }
    )


async def ingest_url(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 64 * 1024)
    if isinstance(body, Response):
        return body
    url = body.get("url")
    if not isinstance(url, str) or not url:
        return validation_error("url required")
    try:
        resp = await br.knowledge_base.ingest_url(url)
    except Exception as exc:  # noqa: BLE001
        return internal_error(f"fetch {url}: {exc}")
    return ok_json(
        {
            "documentId": str(resp.document_id),
            "path": str(resp.path),
            "chunkCount": resp.chunk_count,
            "bytes": resp.bytes,
            "tookMs": resp.took_ms,
            "source": url,
        }
    )

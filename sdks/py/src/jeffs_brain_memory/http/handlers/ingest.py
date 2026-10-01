# SPDX-License-Identifier: Apache-2.0
"""Ingest handlers that delegate to the knowledge base for chunking + index."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ...knowledge import IngestRequest, IngestResponse, InvalidContentError
from ...net import FetchFailedError, UnsafeUrlError
from ..ingest_root import IngestPathInvalidError, PathIngestRefusedError, resolve_ingest_path
from ..problem import bad_gateway, forbidden, internal_error, validation_error
from ._shared import decode_json_body, get_daemon, ok_json, resolve_brain


def _ingest_error(exc: Exception) -> Response:
    """Map an ingest failure to Problem+JSON. Matches Go's
    ``writeIngestError``."""
    if isinstance(exc, PathIngestRefusedError):
        return forbidden(str(exc))
    if isinstance(exc, (UnsafeUrlError, InvalidContentError, IngestPathInvalidError)):
        return validation_error(str(exc))
    if isinstance(exc, FetchFailedError):
        return bad_gateway(str(exc))
    return internal_error(str(exc))


def _ingest_payload(resp: IngestResponse) -> dict[str, Any]:
    return {
        "documentId": str(resp.document_id),
        "path": str(resp.path),
        "chunkCount": resp.chunk_count,
        "bytes": resp.bytes,
        "tookMs": resp.took_ms,
    }


async def ingest_file(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 8 * 1024 * 1024)
    if isinstance(body, Response):
        return body

    path_raw = body.get("path")
    path = path_raw.strip() if isinstance(path_raw, str) else ""
    title = body.get("title") or ""
    tags_raw = body.get("tags") or []
    content_type = body.get("contentType") or ""
    content_b64 = body.get("contentBase64")

    ireq = IngestRequest(
        brain_id=br.id,
        path=path,
        content_type=content_type if isinstance(content_type, str) else "",
        title=title if isinstance(title, str) else "",
        tags=[t for t in tags_raw if isinstance(t, str)] if isinstance(tags_raw, list) else [],
    )

    if isinstance(content_b64, str) and content_b64:
        # Inline bytes: `path` is only a name hint for content-type
        # detection and the stored title, never read from disk.
        try:
            ireq.content = base64.b64decode(content_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            return validation_error(f"invalid contentBase64: {exc}")
    elif not path:
        return validation_error("path or contentBase64 required")
    else:
        try:
            ireq.path = str(resolve_ingest_path(get_daemon(request).ingest_root, path))
        except (PathIngestRefusedError, IngestPathInvalidError) as exc:
            return _ingest_error(exc)

    try:
        resp = await br.knowledge_base.ingest(ireq)
    except Exception as exc:  # noqa: BLE001
        return _ingest_error(exc)
    return ok_json(_ingest_payload(resp))


async def ingest_url(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 64 * 1024)
    if isinstance(body, Response):
        return body
    url = body.get("url")
    if not isinstance(url, str) or not url.strip():
        return validation_error("url required")
    try:
        resp = await br.knowledge_base.ingest_url(url)
    except Exception as exc:  # noqa: BLE001
        return _ingest_error(exc)
    return ok_json({**_ingest_payload(resp), "source": url})

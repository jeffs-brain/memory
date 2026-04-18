# SPDX-License-Identifier: Apache-2.0
"""Problem+JSON error responses per `spec/PROTOCOL.md`."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse


def problem_response(
    *,
    status: int,
    title: str,
    detail: str | None = None,
    code: str | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a Problem+JSON response."""
    body: dict[str, Any] = {"status": status, "title": title}
    if detail is not None:
        body["detail"] = detail
    if code is not None:
        body["code"] = code
    if extra:
        body.update(extra)
    return JSONResponse(
        body,
        status_code=status,
        media_type="application/problem+json",
    )


def not_implemented(endpoint: str) -> JSONResponse:
    """Scaffold response for unimplemented endpoints."""
    return problem_response(
        status=501,
        title="Not Implemented",
        detail=f"{endpoint}: scaffold only, implementation pending",
        code="not_implemented",
    )

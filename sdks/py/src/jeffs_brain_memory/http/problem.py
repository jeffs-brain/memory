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


def validation_error(detail: str) -> JSONResponse:
    return problem_response(
        status=400,
        title="Validation Error",
        detail=detail,
        code="validation_error",
    )


def not_found(detail: str) -> JSONResponse:
    return problem_response(
        status=404,
        title="Not Found",
        detail=detail,
        code="not_found",
    )


def payload_too_large(detail: str) -> JSONResponse:
    return problem_response(
        status=413,
        title="Payload Too Large",
        detail=detail,
        code="payload_too_large",
    )


def unauthorized(detail: str = "missing or invalid Authorization header") -> JSONResponse:
    return problem_response(
        status=401,
        title="Unauthorized",
        detail=detail,
        code="unauthorized",
    )


def forbidden(detail: str = "forbidden") -> JSONResponse:
    return problem_response(
        status=403,
        title="Forbidden",
        detail=detail,
        code="forbidden",
    )


def conflict(detail: str) -> JSONResponse:
    return problem_response(
        status=409,
        title="Conflict",
        detail=detail,
        code="conflict",
    )


def internal_error(detail: str) -> JSONResponse:
    return problem_response(
        status=500,
        title="Internal Server Error",
        detail=detail,
        code="internal_error",
    )


def precondition_required(detail: str) -> JSONResponse:
    return problem_response(
        status=428,
        title="Precondition Required",
        detail=detail,
        code="confirmation_required",
    )


def not_implemented(endpoint: str) -> JSONResponse:
    """Scaffold response for unimplemented endpoints."""
    return problem_response(
        status=501,
        title="Not Implemented",
        detail=f"{endpoint}: scaffold only, implementation pending",
        code="not_implemented",
    )

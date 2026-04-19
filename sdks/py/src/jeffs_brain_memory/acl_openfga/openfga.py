# SPDX-License-Identifier: Apache-2.0
"""OpenFGA HTTP adapter.

Talks to an OpenFGA server (or any API-compatible proxy) over plain HTTP
via `httpx.AsyncClient`. Implements the ACL `Provider` surface: `check`,
`write`, `read`.

The adapter is stateless and only depends on `httpx` (already a hard
dep of this SDK). Pass `client=httpx.AsyncClient(transport=...)` to
inject a mock transport in tests.

Endpoints (per OpenFGA HTTP API):
  POST {api_url}/stores/{store_id}/check
  POST {api_url}/stores/{store_id}/write
  POST {api_url}/stores/{store_id}/read

The OpenFGA model the adapter speaks lives canonically at
`spec/openfga/schema.fga`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, get_args

import httpx

from ..acl.types import (
    Action,
    CheckResult,
    Provider,
    ReadTuplesQuery,
    Resource,
    ResourceType,
    Subject,
    SubjectKind,
    Tuple,
    WriteTuplesRequest,
    allow,
    deny,
)

__all__ = [
    "OpenFgaHTTPError",
    "OpenFgaOptions",
    "OpenFgaRequestError",
    "create_openfga_provider",
    "decode_resource",
    "decode_subject",
    "encode_resource",
    "encode_subject",
]


_ACTION_TO_RELATION: dict[Action, str] = {
    "read": "reader",
    "write": "writer",
    "delete": "can_delete",
    "admin": "admin",
    "export": "can_export",
}

_SUBJECT_KINDS: frozenset[str] = frozenset(get_args(SubjectKind))
_RESOURCE_TYPES: frozenset[str] = frozenset(get_args(ResourceType))


@dataclass(slots=True)
class OpenFgaOptions:
    api_url: str
    store_id: str
    model_id: str | None = None
    token: str | None = None
    client: httpx.AsyncClient | None = None


class OpenFgaHTTPError(Exception):
    """Raised when the OpenFGA server returns a non-2xx response."""

    def __init__(self, status: int, endpoint: str, body: str) -> None:
        self.status = status
        self.endpoint = endpoint
        self.body = body
        snippet = body[:200]
        super().__init__(f"openfga: {endpoint} failed: HTTP {status}: {snippet}")


class OpenFgaRequestError(Exception):
    """Raised when the underlying HTTP transport fails."""

    def __init__(self, endpoint: str, cause: BaseException) -> None:
        self.endpoint = endpoint
        self.cause = cause
        super().__init__(f"openfga: {endpoint} request failed: {cause}")


def encode_subject(s: Subject) -> str:
    return f"{s.kind}:{s.id}"


def encode_resource(r: Resource) -> str:
    return f"{r.type}:{r.id}"


def decode_subject(s: str) -> Subject | None:
    idx = s.find(":")
    if idx == -1:
        return None
    kind = s[:idx]
    rest = s[idx + 1 :]
    if kind not in _SUBJECT_KINDS:
        return None
    return Subject(kind=cast(SubjectKind, kind), id=rest)


def decode_resource(s: str) -> Resource | None:
    idx = s.find(":")
    if idx == -1:
        return None
    type_str = s[:idx]
    rest = s[idx + 1 :]
    if type_str not in _RESOURCE_TYPES:
        return None
    return Resource(type=cast(ResourceType, type_str), id=rest)


def _tuple_to_key(t: Tuple) -> dict[str, str]:
    return {
        "user": encode_subject(t.subject),
        "relation": t.relation,
        "object": encode_resource(t.resource),
    }


class _OpenFgaProvider:
    """OpenFGA-backed `Provider`.

    Ownership rule for the underlying `httpx.AsyncClient`:

      - If the caller passes `OpenFgaOptions.client`, the caller owns its
        lifecycle. `close()` on this provider will NOT close that client.
      - If `OpenFgaOptions.client` is `None`, the adapter constructs its
        own `httpx.AsyncClient`. `close()` will then `await client.aclose()`
        exactly once. Subsequent calls to `close()` are no-ops.
    """

    name: str = "openfga"

    def __init__(self, opts: OpenFgaOptions) -> None:
        self._opts = opts
        self._base_url = opts.api_url.rstrip("/")
        self._store_path = f"{self._base_url}/stores/{opts.store_id}"
        # Track ownership so we close clients we created, not ones passed in.
        if opts.client is not None:
            self._client = opts.client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        self._closed = False

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"content-type": "application/json"}
        if self._opts.token is not None:
            h["authorization"] = f"Bearer {self._opts.token}"
        return h

    async def _post(self, endpoint: str, body: dict[str, Any]) -> Any:
        url = f"{self._store_path}{endpoint}"
        try:
            response = await self._client.post(
                url, json=body, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise OpenFgaRequestError(endpoint, exc) from exc
        if response.status_code < 200 or response.status_code >= 300:
            try:
                text = response.text
            except Exception:
                text = "<no body>"
            raise OpenFgaHTTPError(response.status_code, endpoint, text)
        try:
            return response.json()
        except Exception as exc:
            raise OpenFgaRequestError(endpoint, exc) from exc

    async def check(
        self, subject: Subject, action: Action, resource: Resource
    ) -> CheckResult:
        body: dict[str, Any] = {
            "tuple_key": {
                "user": encode_subject(subject),
                "relation": _ACTION_TO_RELATION[action],
                "object": encode_resource(resource),
            }
        }
        if self._opts.model_id is not None:
            body["authorization_model_id"] = self._opts.model_id
        json_response = await self._post("/check", body)
        if not isinstance(json_response, dict):
            return deny("openfga: malformed response")
        if json_response.get("allowed") is True:
            resolution = json_response.get("resolution")
            return allow(resolution if isinstance(resolution, str) else None)
        resolution = json_response.get("resolution")
        reason = resolution if isinstance(resolution, str) else "openfga: not allowed"
        return deny(reason)

    async def write(self, request: WriteTuplesRequest) -> None:
        if not request.writes and not request.deletes:
            return
        body: dict[str, Any] = {}
        if request.writes:
            body["writes"] = {"tuple_keys": [_tuple_to_key(t) for t in request.writes]}
        if request.deletes:
            body["deletes"] = {"tuple_keys": [_tuple_to_key(t) for t in request.deletes]}
        if self._opts.model_id is not None:
            body["authorization_model_id"] = self._opts.model_id
        await self._post("/write", body)

    async def read(self, query: ReadTuplesQuery) -> list[Tuple]:
        tuple_key: dict[str, str] = {}
        if query.subject is not None:
            tuple_key["user"] = encode_subject(query.subject)
        if query.relation is not None:
            tuple_key["relation"] = query.relation
        if query.resource is not None:
            tuple_key["object"] = encode_resource(query.resource)
        json_response = await self._post("/read", {"tuple_key": tuple_key})
        out: list[Tuple] = []
        if not isinstance(json_response, dict):
            return out
        rows = json_response.get("tuples")
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, dict):
                continue
            user = key.get("user")
            relation = key.get("relation")
            obj = key.get("object")
            if not isinstance(user, str) or not isinstance(relation, str) or not isinstance(obj, str):
                continue
            subject = decode_subject(user)
            resource = decode_resource(obj)
            if subject is None or resource is None:
                continue
            out.append(Tuple(subject=subject, relation=relation, resource=resource))
        return out

    async def close(self) -> None:
        """Release the internal `httpx.AsyncClient` if the adapter constructed it.

        Idempotent: calling `close()` more than once is safe. When the
        caller supplied the client via `OpenFgaOptions.client`, this is a
        no-op so the caller can keep using their client.
        """
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()


def create_openfga_provider(opts: OpenFgaOptions) -> Provider:
    """Build an `httpx`-backed provider that speaks the OpenFGA HTTP API."""
    return _OpenFgaProvider(opts)

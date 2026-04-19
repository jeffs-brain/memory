# SPDX-License-Identifier: Apache-2.0
"""Simple RBAC adapter.

Stores `Tuple` rows in memory (or in an injected `TupleStore`) and
evaluates `check` against a fixed hierarchy:

  workspace -> brain -> collection -> document

Roles collapse to `admin`, `writer`, `reader` at every level, with
inheritance `admin -> writer -> reader`. Granting `admin` on a workspace
implicitly grants `admin` on every brain/collection/document that names
that workspace as its parent. A matching `deny` tuple at a lower level
overrides the parent grant.

Parent relationships are themselves expressed as tuples (the parent
encoded as a service-shaped `Subject`) so the store stays uniform.

Semantics:
  - Only `admin`, `writer`, `reader`, `parent` and `deny:<role>` are
    accepted as relations on `write`. Anything else raises
    `UnsupportedRelationError`.
  - `deny:<role>` tuples take precedence over any grant reachable via
    inheritance, but only at the level the deny is attached to and below.
"""

from __future__ import annotations

from typing import Literal, Protocol, cast, get_args, runtime_checkable

from .types import (
    Action,
    CheckResult,
    Provider,
    ReadTuplesQuery,
    Resource,
    ResourceType,
    Subject,
    Tuple,
    UnsupportedRelationError,
    WriteTuplesRequest,
    allow,
    deny,
    resource_key,
    subject_key,
    tuple_key,
)

__all__ = [
    "InMemoryTupleStore",
    "Role",
    "TupleStore",
    "create_rbac_provider",
    "deny_tuple",
    "grant_tuple",
    "parent_subject",
    "parent_tuple",
]


Role = Literal["admin", "writer", "reader"]

_ROLES: tuple[Role, ...] = ("admin", "writer", "reader")
_ROLE_RANK: dict[Role, int] = {"admin": 3, "writer": 2, "reader": 1}
_DENY_PREFIX = "deny:"

_ACTION_TO_MIN_ROLE: dict[Action, Role] = {
    "read": "reader",
    "write": "writer",
    "delete": "admin",
    "admin": "admin",
    "export": "reader",
}

_RESOURCE_TYPES: frozenset[str] = frozenset(get_args(ResourceType))


def _is_role(s: str) -> bool:
    return s == "admin" or s == "writer" or s == "reader"


@runtime_checkable
class TupleStore(Protocol):
    """Minimal tuple-persistence surface."""

    def add(self, tuple_: Tuple) -> None: ...

    def remove(self, tuple_: Tuple) -> None: ...

    def list(self, query: ReadTuplesQuery) -> list[Tuple]: ...


class InMemoryTupleStore:
    """Default `TupleStore` backed by a dict keyed by tuple key."""

    def __init__(self) -> None:
        self._rows: dict[str, Tuple] = {}

    def add(self, tuple_: Tuple) -> None:
        self._rows[tuple_key(tuple_)] = tuple_

    def remove(self, tuple_: Tuple) -> None:
        self._rows.pop(tuple_key(tuple_), None)

    def list(self, query: ReadTuplesQuery) -> list[Tuple]:
        out: list[Tuple] = []
        for t in self._rows.values():
            if query.subject is not None and subject_key(query.subject) != subject_key(
                t.subject
            ):
                continue
            if query.relation is not None and query.relation != t.relation:
                continue
            if query.resource is not None and resource_key(query.resource) != resource_key(
                t.resource
            ):
                continue
            out.append(t)
        return out


def parent_subject(parent: Resource) -> Subject:
    """Encode a parent resource as a service-shaped pseudo-`Subject`."""
    return Subject(kind="service", id=f"{parent.type}:{parent.id}")


def parent_tuple(child: Resource, parent: Resource) -> Tuple:
    return Tuple(subject=parent_subject(parent), relation="parent", resource=child)


def grant_tuple(subject: Subject, role: Role, resource: Resource) -> Tuple:
    return Tuple(subject=subject, relation=role, resource=resource)


def deny_tuple(subject: Subject, role: Role, resource: Resource) -> Tuple:
    return Tuple(subject=subject, relation=f"{_DENY_PREFIX}{role}", resource=resource)


def _decode_parent_subject(s: Subject) -> Resource | None:
    if s.kind != "service":
        return None
    idx = s.id.find(":")
    if idx == -1:
        return None
    type_str = s.id[:idx]
    id_str = s.id[idx + 1 :]
    if type_str not in _RESOURCE_TYPES:
        return None
    return Resource(type=cast(ResourceType, type_str), id=id_str)


def _assert_supported_relation(relation: str) -> None:
    if relation == "parent":
        return
    if relation.startswith(_DENY_PREFIX):
        rest = relation[len(_DENY_PREFIX) :]
        if _is_role(rest):
            return
    if _is_role(relation):
        return
    raise UnsupportedRelationError(f"rbac: unsupported relation: {relation}")


class _RbacProvider:
    name: str = "rbac"

    def __init__(self, store: TupleStore) -> None:
        self._store = store

    def _walk_ancestors(self, resource: Resource) -> list[Resource]:
        chain: list[Resource] = [resource]
        seen: set[str] = {resource_key(resource)}
        cursor: Resource | None = resource
        while cursor is not None:
            parents = self._store.list(
                ReadTuplesQuery(resource=cursor, relation="parent")
            )
            if not parents:
                break
            parent = _decode_parent_subject(parents[0].subject)
            if parent is None:
                break
            key = resource_key(parent)
            if key in seen:
                break
            seen.add(key)
            chain.append(parent)
            cursor = parent
        return chain

    def _highest_granted_role(
        self, subject: Subject, chain: list[Resource]
    ) -> Role | None:
        best: Role | None = None
        for resource in chain:
            for role in _ROLES:
                found = self._store.list(
                    ReadTuplesQuery(subject=subject, relation=role, resource=resource)
                )
                if found:
                    if best is None or _ROLE_RANK[role] > _ROLE_RANK[best]:
                        best = role
        return best

    def _denied_roles_at(self, subject: Subject, resource: Resource) -> set[Role]:
        denied: set[Role] = set()
        for role in _ROLES:
            found = self._store.list(
                ReadTuplesQuery(
                    subject=subject,
                    relation=f"{_DENY_PREFIX}{role}",
                    resource=resource,
                )
            )
            if found:
                denied.add(role)
        return denied

    async def check(
        self, subject: Subject, action: Action, resource: Resource
    ) -> CheckResult:
        required = _ACTION_TO_MIN_ROLE[action]
        chain = self._walk_ancestors(resource)

        # A deny tuple anywhere from the target resource upwards that
        # covers the required role blocks the request outright.
        for node in chain:
            for role in self._denied_roles_at(subject, node):
                if _ROLE_RANK[role] >= _ROLE_RANK[required]:
                    return deny(f"deny:{role} on {resource_key(node)}")

        granted = self._highest_granted_role(subject, chain)
        if granted is None:
            return deny(f"no grant for {subject_key(subject)} on {resource_key(resource)}")
        if _ROLE_RANK[granted] >= _ROLE_RANK[required]:
            return allow(f"{granted} grant satisfies {required}")
        return deny(f"{granted} insufficient for {required}")

    async def write(self, request: WriteTuplesRequest) -> None:
        for t in request.writes:
            _assert_supported_relation(t.relation)
        for t in request.deletes:
            _assert_supported_relation(t.relation)
        for t in request.writes:
            self._store.add(t)
        for t in request.deletes:
            self._store.remove(t)

    async def read(self, query: ReadTuplesQuery) -> list[Tuple]:
        return self._store.list(query)

    async def close(self) -> None:
        """No-op: the in-memory tuple store holds no releasable state."""
        return None


def create_rbac_provider(store: TupleStore | None = None) -> Provider:
    """Build a fresh in-process RBAC provider."""
    return _RbacProvider(store if store is not None else InMemoryTupleStore())

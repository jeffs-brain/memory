# SPDX-License-Identifier: Apache-2.0
"""ACL primitives for `jeffs_brain_memory`.

Defines the `Provider` protocol used by the memory core to authorise reads
and writes against a `Store`. Two in-box adapters live alongside this
module:

  - `rbac` - a simple role-based adapter for small installs.
  - `acl_openfga` (sibling top-level package) - an OpenFGA HTTP adapter
    for production use.

The OpenFGA authorisation model the adapter speaks lives canonically at
`spec/openfga/schema.fga`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from ..errors import ErrForbidden

__all__ = [
    "Action",
    "AccessControlError",
    "CheckResult",
    "ForbiddenError",
    "Provider",
    "ReadTuplesQuery",
    "Resource",
    "ResourceType",
    "Subject",
    "SubjectKind",
    "Tuple",
    "UnsupportedRelationError",
    "WriteTuplesRequest",
    "allow",
    "deny",
    "resource_key",
    "subject_key",
    "tuple_key",
]


ResourceType = Literal["workspace", "brain", "collection", "document"]
SubjectKind = Literal["user", "api_key", "service"]
Action = Literal["read", "write", "delete", "admin", "export"]


@dataclass(frozen=True, slots=True)
class Resource:
    type: ResourceType
    id: str


@dataclass(frozen=True, slots=True)
class Subject:
    kind: SubjectKind
    id: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Tuple:
    """A single Zanzibar/OpenFGA-style relationship tuple."""

    subject: Subject
    relation: str
    resource: Resource


@dataclass(slots=True)
class WriteTuplesRequest:
    writes: list[Tuple] = field(default_factory=list)
    deletes: list[Tuple] = field(default_factory=list)


@dataclass(slots=True)
class ReadTuplesQuery:
    subject: Subject | None = None
    relation: str | None = None
    resource: Resource | None = None


@runtime_checkable
class Provider(Protocol):
    """Pluggable authorisation provider.

    Adapters MUST implement every method. `write` and `read` may no-op
    where the underlying authoriser does not expose tuple persistence,
    but the methods themselves remain part of the surface so callers can
    program against a single shape.
    """

    name: str

    async def check(
        self, subject: Subject, action: Action, resource: Resource
    ) -> CheckResult: ...

    async def write(self, request: WriteTuplesRequest) -> None: ...

    async def read(self, query: ReadTuplesQuery) -> list[Tuple]: ...

    async def close(self) -> None:
        """Release any resources held by the provider (network clients, file handles, etc.).

        Adapters that own nothing release-worthy implement this as a no-op.
        """
        ...


def allow(reason: str | None = None) -> CheckResult:
    return CheckResult(allowed=True, reason=reason)


def deny(reason: str | None = None) -> CheckResult:
    return CheckResult(allowed=False, reason=reason)


def subject_key(s: Subject) -> str:
    return f"{s.kind}:{s.id}"


def resource_key(r: Resource) -> str:
    return f"{r.type}:{r.id}"


def tuple_key(t: Tuple) -> str:
    return f"{subject_key(t.subject)}|{t.relation}|{resource_key(t.resource)}"


class AccessControlError(Exception):
    """Base class for ACL failures raised by this package."""


class ForbiddenError(AccessControlError, ErrForbidden):
    """Raised when an authorisation check denies a request.

    Inherits from both `AccessControlError` and the framework-wide
    `ErrForbidden` so existing `except ErrForbidden:` handlers keep
    catching denied calls without modification.
    """

    def __init__(
        self,
        subject: Subject,
        action: Action,
        resource: Resource,
        reason: str | None = None,
    ) -> None:
        self.subject = subject
        self.action = action
        self.resource = resource
        self.reason = reason
        suffix = f" ({reason})" if reason else ""
        message = (
            f"acl: forbidden: {subject.kind}:{subject.id} {action} "
            f"{resource.type}:{resource.id}{suffix}"
        )
        super().__init__(message)


class UnsupportedRelationError(AccessControlError):
    """Raised by the RBAC provider when given an unknown relation."""

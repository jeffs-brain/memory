# SPDX-License-Identifier: Apache-2.0
"""Canonical store exceptions.

Mirrors `brain/errors.go` in the Go reference. Every concrete store maps
its backend's failure modes onto one of these classes so callers can match
with `except ErrNotFound` regardless of whether the backend was the
filesystem, git, in-memory, or a remote HTTP daemon.
"""

from __future__ import annotations

__all__ = [
    "StoreError",
    "ErrNotFound",
    "ErrConflict",
    "ErrReadOnly",
    "ErrInvalidPath",
    "ErrSchemaVersion",
    "ErrUnauthorized",
    "ErrForbidden",
    "ErrValidation",
    "ErrPayloadTooLarge",
    "ErrUnsupportedMedia",
    "ErrRateLimited",
    "ErrInternal",
    "ErrBadGateway",
    "ErrTimeout",
]


class StoreError(Exception):
    """Base class for every store error raised by this package."""


class ErrNotFound(StoreError):
    """Raised when the requested path does not exist."""


class ErrConflict(StoreError):
    """Raised when concurrent activity prevents a safe commit."""


class ErrReadOnly(StoreError):
    """Raised when the store is in a read-only state (e.g. after close)."""


class ErrInvalidPath(StoreError):
    """Raised when a path violates the canonical shape rules."""


class ErrSchemaVersion(StoreError):
    """Raised when the brain's schema version is newer than the runtime."""


class ErrUnauthorized(StoreError):
    """Maps to HTTP 401. Missing or unparseable Authorization header."""


class ErrForbidden(StoreError):
    """Maps to HTTP 403. Authenticated principal lacks the required scope."""


class ErrValidation(StoreError):
    """Maps to HTTP 400 for non-path schema violations."""


class ErrPayloadTooLarge(StoreError):
    """Maps to HTTP 413. Body exceeds the wire or client-side cap."""


class ErrUnsupportedMedia(StoreError):
    """Maps to HTTP 415."""


class ErrRateLimited(StoreError):
    """Maps to HTTP 429. Retry budget exhausted after backoff."""


class ErrInternal(StoreError):
    """Maps to HTTP 500. Unhandled server-side failure."""


class ErrBadGateway(StoreError):
    """Maps to HTTP 502. Upstream store returned an unusable response."""


class ErrTimeout(StoreError):
    """Maps to HTTP 504. Server-side deadline exceeded."""

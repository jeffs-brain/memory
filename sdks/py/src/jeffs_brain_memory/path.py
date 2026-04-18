# SPDX-License-Identifier: Apache-2.0
"""Canonical path and brain-scoping types.

The wire contract (see `spec/PROTOCOL.md` and `spec/STORAGE.md`) requires
POSIX-style, relative, canonical paths with no traversal segments, NUL
bytes, or backslashes. This module mirrors the rules in the Go
`brain.ValidatePath` helper.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import NewType

from .errors import ErrInvalidPath

BrainPath = NewType("BrainPath", str)
"""Canonical, relative, POSIX-style path within a brain."""

ChunkID = NewType("ChunkID", str)
"""Opaque identifier for a retrieved or stored chunk."""

DocumentID = NewType("DocumentID", str)
"""Opaque identifier for a stored document."""


@dataclass(frozen=True, slots=True)
class BrainScope:
    """Identifies the brain a request operates on."""

    brain_id: str


def clean_posix(path: str) -> str:
    """Return the canonical POSIX form of a path string.

    Delegates to `posixpath.normpath` but keeps the relative shape (no
    leading slash, no trailing slash beyond the root marker). Callers
    should pass the result through `validate_path` if they want strict
    compliance.
    """
    cleaned = posixpath.normpath(path)
    if cleaned == ".":
        return ""
    return cleaned


def validate_path(path: str) -> BrainPath:
    """Validate and brand a string as a `BrainPath`.

    Raises `ErrInvalidPath` when the path contains backslashes, null bytes,
    a leading or trailing slash, a `..` component, or is otherwise
    non-canonical. Empty strings are rejected.
    """
    if path == "":
        raise ErrInvalidPath("brain: invalid path: empty path")
    if "\\" in path:
        raise ErrInvalidPath(f"brain: invalid path: contains backslash: {path}")
    if "\x00" in path:
        raise ErrInvalidPath("brain: invalid path: contains null byte")
    if path.startswith("/"):
        raise ErrInvalidPath(f"brain: invalid path: leading slash: {path}")
    if path.endswith("/"):
        raise ErrInvalidPath(f"brain: invalid path: trailing slash: {path}")
    cleaned = posixpath.normpath(path)
    if cleaned != path:
        raise ErrInvalidPath(f"brain: invalid path: non-canonical: {path}")
    for part in path.split("/"):
        if part == "..":
            raise ErrInvalidPath(f"brain: invalid path: contains ..: {path}")
    return BrainPath(path)


def is_generated(path: str) -> bool:
    """Report whether the last path component starts with an underscore."""
    base = posixpath.basename(path)
    return base.startswith("_")

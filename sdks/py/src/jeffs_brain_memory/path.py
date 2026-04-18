# SPDX-License-Identifier: Apache-2.0
"""Canonical path and brain-scoping types.

The wire contract (see `spec/PROTOCOL.md` and `spec/STORAGE.md`) requires
POSIX-style, relative, canonical paths with no traversal segments, NUL
bytes, or backslashes. Validation helpers land in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

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
    """Return the canonical POSIX form of `path`. Stub."""
    raise NotImplementedError("path.clean_posix: implement in Phase 4a")


def validate_path(path: str) -> BrainPath:
    """Validate and brand a string as a `BrainPath`. Stub."""
    raise NotImplementedError("path.validate_path: implement in Phase 4a")

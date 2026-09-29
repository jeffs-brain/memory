# SPDX-License-Identifier: Apache-2.0
"""Confinement for ``POST /ingest/file`` requests that name a
server-side path instead of sending ``contentBase64``.

Path ingest is disabled unless the daemon was started with an ingest
root, and a permitted path must resolve, after symlinks, inside that
root. Mirrors ``go/cmd/memory/handler_ingest.go`` and
``sdks/ts/memory/src/http/ingest-root.ts``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "IngestPathInvalidError",
    "PathIngestRefusedError",
    "resolve_ingest_path",
    "resolve_ingest_root",
]


class PathIngestRefusedError(Exception):
    """The request named a path the daemon refuses to read. Maps to 403."""


class IngestPathInvalidError(Exception):
    """The named path is missing or is not a regular file. Maps to 400."""


def resolve_ingest_root(root: str | Path | None) -> Path | None:
    """Validate the configured ingest root and return its absolute,
    symlink-free form. Blank input disables path ingest and yields
    ``None``. Raises when the root is missing or not a directory."""
    text = str(root).strip() if root is not None else ""
    if not text:
        return None
    real = Path(text).resolve(strict=True)
    if not real.is_dir():
        raise NotADirectoryError(f"ingest root: {real} is not a directory")
    return real


def _outside_root() -> PathIngestRefusedError:
    return PathIngestRefusedError("path resolves outside the configured ingest root")


def resolve_ingest_path(root: Path | None, requested: str) -> Path:
    """Resolve ``requested`` inside ``root``.

    A relative path is taken relative to the root. Containment is
    checked on the normalised path first, so a path outside the root is
    refused without revealing whether it exists, and again after
    resolving symlinks, so a link inside the root cannot point the read
    elsewhere. Returns the real path of a regular file inside the root.
    """
    if root is None:
        raise PathIngestRefusedError(
            "server-side path ingest is disabled; send contentBase64 or start "
            "the daemon with an ingest root"
        )
    abs_root = Path(os.path.abspath(root))
    real_root = abs_root.resolve(strict=True)
    candidate = Path(os.path.normpath(os.path.join(abs_root, requested)))
    if not (candidate.is_relative_to(abs_root) or candidate.is_relative_to(real_root)):
        raise _outside_root()
    try:
        real = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise IngestPathInvalidError("file not found") from exc
    except OSError as exc:
        raise _outside_root() from exc
    if not real.is_relative_to(real_root):
        raise _outside_root()
    if not real.is_file():
        raise IngestPathInvalidError("path is not a regular file")
    return real

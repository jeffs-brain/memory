# SPDX-License-Identifier: Apache-2.0
"""In-memory store stand-in for the knowledge test suite.

The real ``jeffs_brain_memory.store.MemStore`` is still a stub that
raises ``NotImplementedError`` everywhere, so this module provides a
minimal implementation of the surface the knowledge package touches:
``read``, ``write``, and ``list`` with prefix filtering plus
``include_generated``. It lives in ``tests/`` so the library package
never picks up test-only code.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _Entry:
    path: str
    size: int = 0
    is_dir: bool = False


class KnowledgeTestStore:
    """Minimal async store for knowledge tests."""

    def __init__(self) -> None:
        self._docs: dict[str, bytes] = {}

    # --- write side ------------------------------------------------------

    async def write(self, path: str, content: bytes) -> None:
        self._docs[str(path)] = bytes(content)

    async def delete(self, path: str) -> None:
        self._docs.pop(str(path), None)

    # --- read side -------------------------------------------------------

    async def read(self, path: str) -> bytes:
        key = str(path)
        if key not in self._docs:
            raise FileNotFoundError(path)
        return self._docs[key]

    async def exists(self, path: str) -> bool:
        return str(path) in self._docs

    async def list(self, dir: str = "", opts: Any = None) -> list[_Entry]:
        prefix = str(dir)
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        recursive = getattr(opts, "recursive", False) if opts is not None else False
        include_generated = (
            getattr(opts, "include_generated", False) if opts is not None else False
        )

        out: list[_Entry] = []
        for key in sorted(self._docs.keys()):
            if prefix and not (key == prefix.rstrip("/") or key.startswith(prefix)):
                continue
            rel = key[len(prefix) :] if prefix else key
            if not include_generated and posixpath.basename(rel).startswith("_"):
                continue
            if not recursive and "/" in rel:
                # Non-recursive: collapse everything below the first
                # segment into a virtual directory entry.
                segment = rel.split("/", 1)[0]
                dir_path = prefix + segment if prefix else segment
                if not any(entry.path == dir_path for entry in out):
                    out.append(_Entry(path=dir_path, is_dir=True))
                continue
            out.append(_Entry(path=key, size=len(self._docs[key])))
        return out

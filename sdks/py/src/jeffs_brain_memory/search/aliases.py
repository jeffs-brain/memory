# SPDX-License-Identifier: Apache-2.0
"""Alias table for the query parser.

The ``AliasTable`` is an in-memory contract mirrored from
``spec/QUERY-DSL.md`` (Alias tables section). Keys are lowercase
surface tokens; values are ordered lists of alternative surface
strings. Persistence is reserved for a future spec revision: the
v1.0 SDK exposes a stub :meth:`AliasTable.save` that raises.

Expansion rules (normative):

- Only bare ``term`` tokens are expanded. Phrases and prefixes pass
  through unchanged.
- Alternatives are lowercased and split on runs of non-(letter|digit)
  characters. Single-word survivors become bare term tokens;
  multi-word survivors become phrase tokens.
- Deduplication keys on ``(kind, text)`` so overlapping alternatives
  collapse to a single emitted token.
- When the original token carried an explicit operator, that operator
  is attached to the first emitted alternative.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable

__all__ = [
    "AliasTable",
    "load",
    "save",
]


class AliasTable:
    """Runtime-only alias map, thread-safe for concurrent readers.

    Writes are rare (construction or ``load``) so the lock is a
    :class:`threading.RLock` rather than an ``RWLock``. The underlying
    ``dict`` is treated as immutable by :meth:`expand`; mutation
    goes through :meth:`set` which snapshots the existing state.
    """

    __slots__ = ("_entries", "_lock")

    def __init__(self, entries: dict[str, list[str]] | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, list[str]] = {}
        if entries:
            for key, values in entries.items():
                self.set(key, values)

    @classmethod
    def from_entries(cls, entries: dict[str, Iterable[str]]) -> AliasTable:
        """Build an :class:`AliasTable` from a plain mapping.

        Keys and values are lowercased and trimmed. Duplicate values
        inside the same entry are collapsed preserving first occurrence.
        Entries that reduce to zero non-empty alternatives are dropped.
        """
        table = cls()
        for raw_key, raw_vals in entries.items():
            table.set(raw_key, list(raw_vals))
        return table

    def set(self, key: str, values: Iterable[str]) -> None:
        """Upsert ``key`` with the cleaned, deduplicated alternatives."""
        trigger = key.strip().lower()
        if not trigger:
            return
        seen: set[str] = set()
        cleaned: list[str] = []
        for raw in values:
            norm = raw.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned.append(norm)
        with self._lock:
            if cleaned:
                self._entries[trigger] = cleaned
            else:
                self._entries.pop(trigger, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            return key.strip().lower() in self._entries

    def expand(self, token: str) -> list[str]:
        """Return alternatives for ``token`` or ``[token]`` on a miss.

        Lookup is case-insensitive. The returned list is a fresh copy
        so callers may mutate it freely.
        """
        key = token.strip().lower()
        if not key:
            return [token]
        with self._lock:
            hits = self._entries.get(key)
            if not hits:
                return [token]
            return list(hits)


def load(path: Path | str) -> AliasTable:
    """Load an alias table from a JSON file.

    Accepted shape is ``{key: [alternative, ...]}`` matching the
    reserved persisted form in ``spec/QUERY-DSL.md``. Missing files
    return an empty table so callers may treat absence as "no aliases
    configured". Malformed JSON raises :class:`json.JSONDecodeError`.
    """
    target = Path(path)
    if not target.exists():
        return AliasTable()
    data = target.read_text(encoding="utf-8").strip()
    if not data:
        return AliasTable()
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError(f"alias file {target} is not a JSON object")
    entries: dict[str, Iterable[str]] = {}
    for key, value in parsed.items():
        if not isinstance(value, list):
            raise ValueError(f"alias entry {key!r} is not a list")
        entries[str(key)] = [str(v) for v in value]
    return AliasTable.from_entries(entries)


def save(path: Path | str, table: AliasTable) -> None:
    """Persist ``table`` to ``path``.

    Stub: persistence is reserved per ``spec/QUERY-DSL.md``. Raises
    :class:`NotImplementedError` to surface the contract unambiguously.
    Callers that need on-disk alias tables should track the spec and
    wait for the normative section.
    """
    raise NotImplementedError("alias persistence is reserved; see spec/QUERY-DSL.md")

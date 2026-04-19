# SPDX-License-Identifier: Apache-2.0
"""YAML frontmatter parsing for indexed markdown files.

Accepts the canonical ``---\\n...\\n---`` block and the line-scan
fallback used by the Go SDK's port for malformed files. Mirrors the
subset of memory and wiki frontmatter fields the search layer
consumes during indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

__all__ = [
    "MemoryFrontmatter",
    "WikiFrontmatter",
    "parse_memory_frontmatter",
    "parse_wiki_frontmatter",
]


@dataclass(slots=True)
class MemoryFrontmatter:
    """Subset of memory frontmatter the indexer cares about."""

    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    session_id: str = ""
    session_date: str = ""
    observed_on: str = ""
    modified: str = ""
    superseded_by: str = ""


@dataclass(slots=True)
class WikiFrontmatter:
    """Subset of wiki article frontmatter the indexer cares about."""

    title: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    session_id: str = ""
    session_date: str = ""
    observed_on: str = ""
    modified: str = ""


def _split_frontmatter(content: str) -> tuple[list[str], str] | None:
    """Split a markdown file into frontmatter lines and the body.

    Returns ``None`` when the file does not open with ``---``, mirroring
    the Go SDK's behaviour of treating non-frontmatter files as pure
    body.
    """
    lines = content.split("\n")
    if len(lines) < 2 or lines[0].strip() != "---":
        return None
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return None
    body = "\n".join(lines[close_idx + 1 :]).strip()
    return lines[1:close_idx], body


def _coerce_str(value: Any) -> str:
    """Return ``value`` rendered as a string suitable for indexing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_tags(value: Any) -> list[str]:
    """Normalise ``value`` into a list of tag strings.

    Accepts a list (YAML sequence), a comma-separated scalar, or a
    single scalar. Empty entries are dropped.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _line_scan_kv(lines: list[str]) -> dict[str, Any]:
    """Fallback parser when YAML parsing fails or is skipped.

    Mirrors the Go ``parseFrontmatterKV`` helper: split on the first
    colon, strip optional surrounding quotes, capture dash-prefixed
    continuations as list items for the most recent key.
    """
    out: dict[str, Any] = {}
    current_list_key = ""
    for raw in lines:
        trimmed = raw.strip()
        if current_list_key and trimmed.startswith("- "):
            val = trimmed[2:].strip()
            existing = out.setdefault(current_list_key, [])
            if isinstance(existing, list):
                existing.append(val)
            continue
        if ":" not in raw:
            current_list_key = ""
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not value:
            current_list_key = key
            out.setdefault(key, [])
            continue
        current_list_key = ""
        out[key] = value
    return out


def _parse_block(content: str) -> tuple[dict[str, Any], str]:
    """Return ``(fields, body)`` from ``content``.

    Prefers a real YAML parse; falls back to a line-scan for
    malformed or ambiguous blocks so a broken file does not kill
    indexing of the surrounding corpus.
    """
    split = _split_frontmatter(content)
    if split is None:
        return {}, content
    lines, body = split
    raw = "\n".join(lines)
    parsed: dict[str, Any] | None = None
    try:
        loaded = yaml.safe_load(raw) if raw.strip() else {}
    except yaml.YAMLError:
        loaded = None
    if isinstance(loaded, dict):
        parsed = {str(k): v for k, v in loaded.items()}
    if parsed is None:
        parsed = _line_scan_kv(lines)
    return parsed, body


def parse_memory_frontmatter(content: str) -> tuple[MemoryFrontmatter, str]:
    """Parse a memory markdown file.

    Returns ``(frontmatter, body)``. Missing fields remain at their
    dataclass defaults so callers may feed the struct straight into
    the FTS index.
    """
    fields, body = _parse_block(content)
    fm = MemoryFrontmatter(
        name=_coerce_str(fields.get("name")),
        description=_coerce_str(fields.get("description")),
        tags=_coerce_tags(fields.get("tags")),
        session_id=_coerce_str(fields.get("session_id")),
        session_date=_coerce_str(fields.get("session_date")),
        observed_on=_coerce_str(fields.get("observed_on")),
        modified=_coerce_str(fields.get("modified")),
        superseded_by=_coerce_str(fields.get("superseded_by")),
    )
    return fm, body


def parse_wiki_frontmatter(content: str) -> tuple[WikiFrontmatter, str]:
    """Parse a wiki markdown file.

    Behaves like :func:`parse_memory_frontmatter` but maps the
    ``title`` / ``summary`` fields instead of memory's ``name`` /
    ``description``.
    """
    fields, body = _parse_block(content)
    fm = WikiFrontmatter(
        title=_coerce_str(fields.get("title")),
        summary=_coerce_str(fields.get("summary")),
        tags=_coerce_tags(fields.get("tags")),
        session_id=_coerce_str(fields.get("session_id")),
        session_date=_coerce_str(fields.get("session_date")),
        observed_on=_coerce_str(fields.get("observed_on")),
        modified=_coerce_str(fields.get("modified")),
    )
    return fm, body

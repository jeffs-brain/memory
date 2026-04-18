# SPDX-License-Identifier: Apache-2.0
"""YAML frontmatter parsing with a line-scan fallback.

Ported from ``sdks/go/knowledge/frontmatter.go`` and the original
``jeff/apps/jeff/internal/knowledge/frontmatter.go``. The parser stays
lenient about quoting and list shapes that hand-authored markdown
commonly uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

__all__ = ["Frontmatter", "parse_frontmatter"]


@dataclass(slots=True)
class Frontmatter:
    """Fields extracted from a YAML header.

    Covers both the article shape (``title``/``summary``/``tags``) and
    the memory-scope shape (``name``/``description``). Extra fields that
    the YAML path captures land in ``extra`` so downstream callers can
    inspect them without parsing the header a second time.
    """

    title: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source: str = ""
    source_type: str = ""
    created: str = ""
    modified: str = ""
    ingested: str = ""
    name: str = ""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def has_anything(self) -> bool:
        """Report whether any field carries content."""
        return bool(
            self.title
            or self.summary
            or self.source
            or self.source_type
            or self.created
            or self.modified
            or self.ingested
            or self.name
            or self.description
            or self.tags
            or self.sources
        )


# Keys lifted into the top-level Frontmatter fields. Anything else the
# YAML parse yields drops into ``extra`` so downstream tooling still sees
# the data.
_KNOWN_KEYS = {
    "title",
    "summary",
    "tags",
    "sources",
    "source",
    "source_type",
    "created",
    "modified",
    "ingested",
    "name",
    "description",
}


def parse_frontmatter(content: str) -> tuple[Frontmatter, str]:
    """Extract the YAML frontmatter block and return the remainder.

    Accepts the familiar ``---\\n<yaml>\\n---\\n<body>`` layout. When the
    content has no frontmatter block the returned :class:`Frontmatter` is
    empty and ``body`` is the entire input. Malformed YAML falls through
    to a line-by-line scan so the caller always receives a usable record.
    """

    lines = content.splitlines()
    if len(lines) < 2 or lines[0].strip() != "---":
        return Frontmatter(), content

    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return Frontmatter(), content

    header = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :]).strip()

    try:
        parsed = yaml.safe_load(header)
    except yaml.YAMLError:
        parsed = None

    if isinstance(parsed, dict):
        fm = _frontmatter_from_dict(parsed)
        if fm.has_anything() or _all_keys_empty(parsed):
            return fm, body

    fm = _parse_line_scan(lines[1:close_idx])
    return fm, body


def _all_keys_empty(parsed: dict[str, Any]) -> bool:
    """Report whether ``parsed`` has no useful values at all.

    When YAML parses a header whose values are every-field-empty we still
    want the YAML result rather than the line-scan fallback (the line
    scan would produce the same empty result anyway).
    """
    return not any(bool(v) for v in parsed.values())


def _frontmatter_from_dict(raw: dict[str, Any]) -> Frontmatter:
    """Project a parsed YAML dict onto :class:`Frontmatter`."""
    fm = Frontmatter()
    for key, val in raw.items():
        key_l = key.lower() if isinstance(key, str) else str(key).lower()
        if key_l == "title":
            fm.title = _as_string(val)
        elif key_l == "summary":
            fm.summary = _as_string(val)
        elif key_l == "source":
            fm.source = _as_string(val)
        elif key_l == "source_type":
            fm.source_type = _as_string(val)
        elif key_l == "created":
            fm.created = _as_string(val)
        elif key_l == "modified":
            fm.modified = _as_string(val)
        elif key_l == "ingested":
            fm.ingested = _as_string(val)
        elif key_l == "name":
            fm.name = _as_string(val)
        elif key_l == "description":
            fm.description = _as_string(val)
        elif key_l == "tags":
            fm.tags = _as_string_list(val)
        elif key_l == "sources":
            fm.sources = _as_string_list(val)
        else:
            fm.extra[key_l] = val
    return fm


def _as_string(val: Any) -> str:
    """Coerce a scalar YAML value to a string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def _as_string_list(val: Any) -> list[str]:
    """Coerce a YAML list or comma-delimited scalar to a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [_as_string(v) for v in val if _as_string(v) != ""]
    if isinstance(val, str):
        return [part.strip() for part in val.split(",") if part.strip()]
    return [str(val)]


def _parse_line_scan(lines: list[str]) -> Frontmatter:
    """Jeff-style line-scan fallback for malformed YAML blocks."""
    fm = Frontmatter()
    current_list_key: str | None = None

    for line in lines:
        trimmed = line.strip()

        if current_list_key and trimmed.startswith("- "):
            val = _strip_quotes(trimmed[2:].strip())
            if current_list_key == "tags":
                fm.tags.append(val)
            elif current_list_key == "sources":
                fm.sources.append(val)
            continue

        key, val, ok = _parse_kv(line)
        if not ok:
            current_list_key = None
            continue

        if val == "":
            current_list_key = key
            continue
        current_list_key = None

        if key == "title":
            fm.title = val
        elif key == "summary":
            fm.summary = val
        elif key == "source":
            fm.source = val
        elif key == "source_type":
            fm.source_type = val
        elif key == "created":
            fm.created = val
        elif key == "modified":
            fm.modified = val
        elif key == "ingested":
            fm.ingested = val
        elif key == "name":
            fm.name = val
        elif key == "description":
            fm.description = val
        elif key == "tags":
            for tag in val.split(","):
                tag = _strip_quotes(tag.strip())
                if tag:
                    fm.tags.append(tag)
        elif key == "sources":
            for src in val.split(","):
                src = _strip_quotes(src.strip())
                if src:
                    fm.sources.append(src)

    return fm


def _parse_kv(line: str) -> tuple[str, str, bool]:
    """Split ``key: value`` honouring the YAML quoting conventions."""
    idx = line.find(":")
    if idx < 0:
        return "", "", False
    key = line[:idx].strip()
    val = _strip_quotes(line[idx + 1 :].strip())
    return key, val, True


def _strip_quotes(val: str) -> str:
    """Remove a matching pair of single or double quotes from ``val``."""
    if len(val) < 2:
        return val
    first, last = val[0], val[-1]
    if (first == '"' and last == '"') or (first == "'" and last == "'"):
        return val[1:-1]
    return val

# SPDX-License-Identifier: Apache-2.0
"""Topic listing and frontmatter parsing."""

from __future__ import annotations

from .types import Frontmatter, TopicFile

__all__ = ["Frontmatter", "TopicFile", "parse_frontmatter", "parse_kv"]


def parse_frontmatter(content: str) -> tuple[Frontmatter, str]:
    """Extract YAML frontmatter delimited by `---` lines."""
    lines = content.split("\n")
    if len(lines) < 2 or lines[0].strip() != "---":
        return Frontmatter(), content

    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return Frontmatter(), content

    fm = Frontmatter()
    current_list_key = ""

    for line in lines[1:close_idx]:
        trimmed = line.strip()

        if current_list_key and trimmed.startswith("- "):
            val = trimmed[1:].strip()
            if current_list_key == "tags":
                fm.tags.append(val)
            continue

        kv = parse_kv(line)
        if kv is None:
            current_list_key = ""
            continue

        key, val = kv
        if val == "":
            current_list_key = key
            continue
        current_list_key = ""

        if key == "name":
            fm.name = val
        elif key == "description":
            fm.description = val
        elif key == "type":
            fm.type = val
        elif key == "created":
            fm.created = val
        elif key == "modified":
            fm.modified = val
        elif key == "confidence":
            fm.confidence = val
        elif key == "source":
            fm.source = val
        elif key == "supersedes":
            fm.supersedes = val
        elif key == "superseded_by":
            fm.superseded_by = val
        elif key == "tags":
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                for tag in inner.split(","):
                    tag = tag.strip()
                    if tag:
                        fm.tags.append(tag)
            else:
                for tag in val.split(","):
                    tag = tag.strip()
                    if tag:
                        fm.tags.append(tag)

    remaining = lines[close_idx + 1 :]
    body = "\n".join(remaining).strip()
    return fm, body


def parse_kv(line: str) -> tuple[str, str] | None:
    """Split a `key: value` line; returns ``None`` without a colon."""
    idx = line.find(":")
    if idx < 0:
        return None
    key = line[:idx].strip()
    val = line[idx + 1 :].strip()
    if len(val) >= 2:
        if (val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'"):
            val = val[1:-1]
    return key, val

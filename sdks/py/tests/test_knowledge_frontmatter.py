# SPDX-License-Identifier: Apache-2.0
"""Frontmatter parser tests."""

from __future__ import annotations

from jeffs_brain_memory.knowledge.frontmatter import parse_frontmatter


def test_basic_yaml_path() -> None:
    content = '---\ntitle: "Hello"\ntags: [one, two]\nsummary: "short"\n---\nbody\n'
    fm, body = parse_frontmatter(content)
    assert fm.title == "Hello"
    assert fm.summary == "short"
    assert fm.tags == ["one", "two"]
    assert body == "body"


def test_list_form_bullets() -> None:
    content = '---\ntitle: "With list"\ntags:\n  - alpha\n  - beta\nsources:\n  - a\n  - b\n---\nbody\n'
    fm, _ = parse_frontmatter(content)
    assert fm.tags == ["alpha", "beta"]
    assert fm.sources == ["a", "b"]


def test_no_header_returns_content_untouched() -> None:
    fm, body = parse_frontmatter("plain body\n")
    assert fm.title == ""
    assert "plain body" in body


def test_memory_scope_shape() -> None:
    content = '---\nname: "foo"\ndescription: "bar"\n---\nbody'
    fm, _ = parse_frontmatter(content)
    assert fm.name == "foo"
    assert fm.description == "bar"


def test_line_scan_fallback_for_malformed_yaml() -> None:
    content = "---\ntitle: Broken\ntags:\n  - one\n  - two\nsummary: test\n---\nbody"
    fm, body = parse_frontmatter(content)
    assert fm.title == "Broken"
    assert fm.summary == "test"
    assert fm.tags == ["one", "two"]
    assert body == "body"


def test_unterminated_frontmatter_returns_full_content() -> None:
    content = "---\ntitle: never-closes\nbody goes here\n"
    fm, body = parse_frontmatter(content)
    assert fm.title == ""
    assert content == body


def test_source_type_and_ingested_round_trip() -> None:
    content = '---\ntitle: T\nsource_type: pdf\ningested: "2026-01-02T03:04:05Z"\n---\nbody'
    fm, _ = parse_frontmatter(content)
    assert fm.source_type == "pdf"
    assert fm.ingested.startswith("2026-01-02")

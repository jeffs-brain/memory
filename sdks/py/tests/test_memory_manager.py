# SPDX-License-Identifier: Apache-2.0
"""Core MemoryManager tests: frontmatter, indexes, topics, prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeffs_brain_memory.memory import (
    MemoryManager,
    MemStore,
    memory_global_index,
    memory_global_topic,
    memory_project_index,
    memory_project_topic,
    parse_frontmatter,
    parse_kv,
    project_slug,
    set_slug_map_for_test,
)


@pytest.fixture
def slug_map_isolated(tmp_path: Path):
    restore = set_slug_map_for_test(str(tmp_path / "slug-map.yaml"))
    yield
    restore()


@pytest.fixture
def mem():
    store = MemStore()
    return MemoryManager(store), store


def write(store: MemStore, path: str, content: str) -> None:
    store.write(path, content.encode("utf-8"))


# ---- ParseFrontmatter ----


def test_parse_frontmatter_complete():
    content = """---
name: Architecture
description: System architecture notes
type: project
---

Some body content here."""
    fm, body = parse_frontmatter(content)
    assert fm.name == "Architecture"
    assert fm.description == "System architecture notes"
    assert fm.type == "project"
    assert body == "Some body content here."


def test_parse_frontmatter_quoted_values():
    content = """---
name: "My Topic"
description: 'A description'
type: user
---

Body."""
    fm, _ = parse_frontmatter(content)
    assert fm.name == "My Topic"
    assert fm.description == "A description"
    assert fm.type == "user"


def test_parse_frontmatter_no_frontmatter():
    content = "Just some plain text."
    fm, body = parse_frontmatter(content)
    assert fm.name == "" and fm.description == "" and fm.type == ""
    assert body == content


def test_parse_frontmatter_unclosed_block():
    content = """---
name: Broken
type: project
No closing delimiter here."""
    fm, body = parse_frontmatter(content)
    assert fm.name == ""
    assert body == content


def test_parse_frontmatter_empty_body():
    content = """---
name: Empty
type: reference
---"""
    fm, body = parse_frontmatter(content)
    assert fm.name == "Empty"
    assert fm.type == "reference"
    assert body == ""


def test_parse_kv_simple():
    kv = parse_kv("name: test")
    assert kv == ("name", "test")


def test_parse_kv_no_colon():
    assert parse_kv("no colon here") is None


def test_parse_kv_double_quoted():
    kv = parse_kv('name: "quoted value"')
    assert kv is not None
    assert kv[1] == "quoted value"


def test_parse_kv_single_quoted():
    kv = parse_kv("name: 'single quoted'")
    assert kv is not None
    assert kv[1] == "single quoted"


def test_parse_kv_empty_value():
    kv = parse_kv("name:")
    assert kv == ("name", "")


# ---- LoadProjectIndex / LoadGlobalIndex ----


def test_load_project_index_missing(mem, slug_map_isolated):
    m, _ = mem
    assert m.load_project_index("/some/project") == ""


def test_load_project_index_reads_content(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    content = "# My Memory\n\nSome important notes."
    write(store, memory_project_index(slug), content)
    assert m.load_project_index(path) == content


def test_load_project_index_caps_at_200_lines(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    raw = "\n".join(["line"] * 250)
    write(store, memory_project_index(slug), raw)
    out = m.load_project_index(path)
    assert "[...truncated]" in out
    assert len(out.split("\n")) == 201


# ---- ListProjectTopics ----


def test_list_project_topics_empty(mem, slug_map_isolated):
    m, _ = mem
    assert m.list_project_topics("/no/such/project") == []


def test_list_project_topics_skips_memory_md(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    write(store, memory_project_index(slug), "# Index")
    write(
        store,
        memory_project_topic(slug, "architecture"),
        """---
name: Architecture
description: System design
type: project
---

Content here.""",
    )
    topics = m.list_project_topics(path)
    assert len(topics) == 1
    assert topics[0].name == "Architecture"
    assert topics[0].type == "project"
    assert topics[0].description == "System design"


def test_list_project_topics_fallback_filename(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    write(store, memory_project_topic(slug, "notes"), "Just plain text.")
    topics = m.list_project_topics(path)
    assert len(topics) == 1
    assert topics[0].name == "notes"


# ---- ReadTopic ----


def test_read_topic_success(mem, slug_map_isolated):
    m, store = mem
    expected = "# Test\n\nContent."
    write(store, memory_global_topic("test"), expected)
    assert m.read_topic(memory_global_topic("test")) == expected


def test_read_topic_not_found(mem, slug_map_isolated):
    m, _ = mem
    with pytest.raises(Exception):
        m.read_topic(memory_global_topic("nope"))


# ---- BuildMemoryPromptFor ----


def test_build_memory_prompt_no_index(mem, slug_map_isolated):
    m, _ = mem
    assert m.build_memory_prompt_for("/no/such/project") == ""


def test_build_memory_prompt_project_only(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    write(store, memory_project_index(slug), "- Architecture notes")
    out = m.build_memory_prompt_for(path)
    assert "# Project Memory" in out
    assert "# Global Memory" not in out


def test_build_memory_prompt_global_only(mem, slug_map_isolated):
    m, store = mem
    write(store, memory_global_index(), "- User prefers British English")
    out = m.build_memory_prompt_for("/example/project")
    assert "# Global Memory" in out
    assert "British English" in out
    assert "# Project Memory" not in out


def test_build_memory_prompt_both_scopes(mem, slug_map_isolated):
    m, store = mem
    path = "/example/project"
    slug = project_slug(path)
    write(store, memory_global_index(), "- User prefers British English")
    write(store, memory_project_index(slug), "- Architecture notes")
    out = m.build_memory_prompt_for(path)
    assert out.index("# Global Memory") < out.index("# Project Memory")


# ---- ProjectSlug ----


def test_project_slug_deterministic(tmp_path, slug_map_isolated):
    a = project_slug(str(tmp_path))
    b = project_slug(str(tmp_path))
    assert a == b


def test_project_slug_different_paths(tmp_path, slug_map_isolated):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert project_slug(str(a)) != project_slug(str(b))

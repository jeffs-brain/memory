# SPDX-License-Identifier: Apache-2.0
"""Wikilink extraction and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from jeffs_brain_memory.memory import (
    MemoryManager,
    MemStore,
    extract_wikilinks,
    memory_global_topic,
    memory_project_topic,
    normalise_topic,
    project_slug,
    resolve_all_wikilinks,
    resolve_wikilink,
    set_slug_map_for_test,
)


@pytest.fixture
def iso(tmp_path: Path):
    restore = set_slug_map_for_test(str(tmp_path / "slug-map.yaml"))
    yield
    restore()


def test_extract_wikilinks_multiple():
    links = extract_wikilinks(
        "See [[architecture]] and [[deployment]] and [[tooling]]."
    )
    assert links == ["architecture", "deployment", "tooling"]


def test_extract_wikilinks_with_display():
    links = extract_wikilinks("See [[auth-migration|Auth Migration]] today.")
    assert links == ["auth-migration|Auth Migration"]


def test_extract_wikilinks_none():
    assert extract_wikilinks("plain text") == []


def test_extract_wikilinks_global_prefix():
    assert extract_wikilinks("[[global:coding-style]]") == ["global:coding-style"]


def test_normalise_topic_spaces_to_hyphens():
    assert normalise_topic("Some Topic Name") == "some-topic-name"


def test_normalise_topic_trims_whitespace():
    assert normalise_topic("  padded  ") == "padded"


def test_normalise_topic_empty():
    assert normalise_topic("") == ""


def test_resolve_project_before_global(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    store.write(memory_project_topic(slug, "architecture"), b"project body")
    store.write(memory_global_topic("architecture"), b"global body")
    resolved = resolve_wikilink(mem, "architecture", "/x")
    assert resolved == memory_project_topic(slug, "architecture")


def test_resolve_global_bypass(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    store.write(memory_project_topic(slug, "style"), b"p")
    store.write(memory_global_topic("style"), b"g")
    resolved = resolve_wikilink(mem, "global:style", "/x")
    assert resolved == memory_global_topic("style")


def test_resolve_falls_back_to_global(iso):
    store = MemStore()
    mem = MemoryManager(store)
    store.write(memory_global_topic("user-prefs"), b"g")
    resolved = resolve_wikilink(mem, "user-prefs", "/x")
    assert resolved == memory_global_topic("user-prefs")


def test_resolve_missing_returns_empty(iso):
    store = MemStore()
    mem = MemoryManager(store)
    assert resolve_wikilink(mem, "nope", "/x") == ""


def test_resolve_all_dedup(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    store.write(memory_project_topic(slug, "auth"), b"body")
    paths = resolve_all_wikilinks(mem, "See [[auth]] and [[auth]] again.", "/x")
    assert len(paths) == 1


def test_resolve_all_skips_missing(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    store.write(memory_project_topic(slug, "exists"), b"body")
    paths = resolve_all_wikilinks(mem, "See [[exists]] and [[missing]].", "/x")
    assert len(paths) == 1

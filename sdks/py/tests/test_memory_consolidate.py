# SPDX-License-Identifier: Apache-2.0
"""Consolidation: index rebuild, staleness, heuristic reinforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jeffs_brain_memory.memory import (
    Consolidator,
    MemoryManager,
    MemStore,
    memory_global_index,
    memory_global_topic,
    memory_project_index,
    memory_project_topic,
    project_slug,
    set_slug_map_for_test,
)


@pytest.fixture
def iso(tmp_path: Path):
    restore = set_slug_map_for_test(str(tmp_path / "slug-map.yaml"))
    yield
    restore()


def write(store, path, content):
    store.write(path, content.encode("utf-8"))


@pytest.mark.asyncio
async def test_new_consolidator(iso):
    store = MemStore()
    mem = MemoryManager(store)
    c = Consolidator(None, "", mem)
    assert c is not None


@pytest.mark.asyncio
async def test_run_quick_empty(iso):
    store = MemStore()
    mem = MemoryManager(store)
    c = Consolidator(None, "", mem)
    report = await c.run_quick()
    assert report is not None
    assert report.duration_seconds >= 0


@pytest.mark.asyncio
async def test_run_quick_rebuilds_index(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/example/project")
    write(
        store,
        memory_project_topic(slug, "architecture"),
        """---
name: Architecture Overview
description: High-level system architecture
type: project
---

microservices.
""",
    )
    write(
        store,
        memory_project_topic(slug, "testing-patterns"),
        """---
name: Testing Patterns
description: How we write tests
type: reference
---

tests.
""",
    )
    write(store, memory_project_index(slug), "- [Old Entry](old.md) -- outdated\n")
    c = Consolidator(None, "", mem)
    report = await c.run_quick()
    assert report.indexes_rebuilt >= 1
    data = store.read(memory_project_index(slug)).decode("utf-8")
    assert "architecture.md" in data
    assert "testing-patterns.md" in data
    assert "old.md" not in data
    assert "Architecture Overview" in data


@pytest.mark.asyncio
async def test_staleness_detects_old_files(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/x")
    old = (datetime.now(timezone.utc) - timedelta(days=200)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    write(
        store,
        memory_project_topic(slug, "stale"),
        f"---\nname: Stale\nmodified: {old}\n---\n\nOld.\n",
    )
    c = Consolidator(None, "", mem)
    report = await c.run_quick()
    assert report.stale_memories_flagged >= 1


@pytest.mark.asyncio
async def test_run_quick_reinforces_heuristics(iso):
    store = MemStore()
    mem = MemoryManager(store)
    slug = project_slug("/y")
    # Heuristic file with 4+ observations (high confidence expected) but
    # tagged low.
    body = """---
name: "testing: r"
description: "rule"
type: feedback
modified: 2026-01-01T00:00:00Z
confidence: low
source: reflection
tags:
  - heuristic
  - testing
---

## R

body

## R2

body

## R3

body

## R4

body
"""
    write(store, memory_project_topic(slug, "heuristic-testing-x"), body)
    c = Consolidator(None, "", mem)
    report = await c.run_quick()
    data = store.read(memory_project_topic(slug, "heuristic-testing-x")).decode("utf-8")
    assert "confidence: high" in data
    assert report.heuristics_updated >= 1


@pytest.mark.asyncio
async def test_run_full_without_provider_skips_dedup(iso):
    store = MemStore()
    mem = MemoryManager(store)
    c = Consolidator(None, "", mem)
    report = await c.run_full()
    assert any("deduplication skipped" in e for e in report.errors)


@pytest.mark.asyncio
async def test_in_progress_guard(iso):
    store = MemStore()
    mem = MemoryManager(store)
    c = Consolidator(None, "", mem)
    # Force the guard.
    c._in_progress = True
    with pytest.raises(RuntimeError):
        await c.run_quick()

# SPDX-License-Identifier: Apache-2.0
"""Buffer append/render/compact tests."""

from __future__ import annotations

from datetime import datetime, timezone

from jeffs_brain_memory.memory import MemStore, memory_buffer_global, memory_buffer_project
from jeffs_brain_memory.memory.buffer import (
    Buffer,
    Config,
    Observation,
    Scope,
    ScopeKind,
    default_config,
    format_observation,
)


def make_buffer(store=None, *, cfg=None, scope=None):
    store = store or MemStore()
    cfg = cfg or default_config()
    scope = scope or Scope(kind=ScopeKind.GLOBAL)
    return Buffer(store, scope, cfg), store


def test_append_and_render_round_trip():
    buf, _ = make_buffer()
    buf.append(
        Observation(
            at=datetime(2026, 4, 14, 10, 30, 0, tzinfo=timezone.utc),
            intent="edit",
            entities=["brain/paths.go"],
            outcome="ok",
            summary="Added buffer path helpers",
        )
    )
    out = buf.render()
    assert "Added buffer path helpers" in out
    assert "(edit)" in out
    assert "{brain/paths.go}" in out
    assert "[ok]" not in out


def test_render_empty_buffer():
    buf, _ = make_buffer()
    assert buf.render() == ""


def test_token_count():
    store = MemStore()
    buf = Buffer(store, Scope(kind=ScopeKind.GLOBAL), default_config())
    assert buf.token_count() == 0
    store.write(memory_buffer_global(), b"abcd" * 100)
    assert buf.token_count() == 100


def test_needs_compaction_triggers():
    cfg = Config(
        token_budget=10, compact_threshold=100, keep_recent_percent=50, max_observation_len=160
    )
    store = MemStore()
    buf = Buffer(store, Scope(kind=ScopeKind.GLOBAL), cfg)
    assert not buf.needs_compaction()
    store.write(memory_buffer_global(), b"a" * 44)
    assert buf.needs_compaction()


def test_compact_keeps_recent():
    cfg = Config(
        token_budget=8192,
        compact_threshold=100,
        keep_recent_percent=50,
        max_observation_len=160,
    )
    store = MemStore()
    buf = Buffer(store, Scope(kind=ScopeKind.GLOBAL), cfg)
    now = datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        buf.append(
            Observation(
                at=now.replace(minute=i), intent="chat", summary="line " + "x" * 5
            )
        )
    removed = buf.compact()
    assert removed == 5
    lines = buf.render().strip().split("\n")
    assert len(lines) == 5


def test_compact_empty_buffer_noop():
    buf, _ = make_buffer()
    assert buf.compact() == 0


def test_summary_truncation():
    cfg = Config(
        token_budget=8192,
        compact_threshold=100,
        keep_recent_percent=50,
        max_observation_len=20,
    )
    store = MemStore()
    buf = Buffer(store, Scope(kind=ScopeKind.GLOBAL), cfg)
    buf.append(
        Observation(
            at=datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc),
            intent="chat",
            summary="z" * 50,
        )
    )
    out = buf.render()
    assert "z" * 50 not in out
    assert "z" * 20 in out


def test_global_vs_project_paths():
    store = MemStore()
    gbuf = Buffer(store, Scope(kind=ScopeKind.GLOBAL), default_config())
    pbuf = Buffer(store, Scope(kind=ScopeKind.PROJECT, slug="my-project"), default_config())
    at = datetime(2026, 4, 14, 9, 0, 0, tzinfo=timezone.utc)
    gbuf.append(Observation(at=at, intent="plan", summary="global obs"))
    pbuf.append(Observation(at=at, intent="plan", summary="project obs"))
    gc = gbuf.render()
    pc = pbuf.render()
    assert "global obs" in gc and "project obs" not in gc
    assert "project obs" in pc and "global obs" not in pc
    assert store.exists(memory_buffer_global())
    assert store.exists(memory_buffer_project("my-project"))


def test_format_observation_shapes():
    at = datetime(2026, 4, 14, 14, 30, 45, tzinfo=timezone.utc)
    assert (
        format_observation(at, "edit", "error", "x", ["paths.go", "store.go"])
        == "- [14:30:45] (edit) [error] x {paths.go, store.go}"
    )
    assert format_observation(at, "read", "ok", "x", []) == "- [14:30:45] (read) x"
    assert format_observation(at, "", "", "bare", []) == "- [14:30:45] bare"

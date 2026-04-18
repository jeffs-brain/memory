# SPDX-License-Identifier: Apache-2.0
"""SlugMap persistence and lookup."""

from __future__ import annotations

from pathlib import Path

from jeffs_brain_memory.memory import SlugMap


def test_lookup_miss(tmp_path: Path):
    sm = SlugMap(str(tmp_path / "slug-map.yaml"))
    assert sm.lookup("/nope") is None


def test_register_and_lookup(tmp_path: Path):
    sm = SlugMap(str(tmp_path / "slug-map.yaml"))
    sm.register("/home/alex/code/jeff", "jaythegeek-jeff")
    assert sm.lookup("/home/alex/code/jeff") == "jaythegeek-jeff"


def test_persists_across_load_save(tmp_path: Path):
    path = tmp_path / "slug-map.yaml"
    sm1 = SlugMap(str(path))
    sm1.register("/a", "slug-a")
    sm1.register("/b", "slug-b")

    sm2 = SlugMap(str(path))
    sm2.load()
    assert sm2.lookup("/a") == "slug-a"
    assert sm2.lookup("/b") == "slug-b"


def test_overwrite_existing(tmp_path: Path):
    sm = SlugMap(str(tmp_path / "slug-map.yaml"))
    sm.register("/proj", "old")
    sm.register("/proj", "new")
    assert sm.lookup("/proj") == "new"


def test_load_missing_file(tmp_path: Path):
    sm = SlugMap(str(tmp_path / "no" / "slug-map.yaml"))
    sm.load()
    assert sm.lookup("/anything") is None


def test_multiple_projects_same_slug(tmp_path: Path):
    sm = SlugMap(str(tmp_path / "slug-map.yaml"))
    sm.register("/alice/jeff", "jaythegeek-jeff")
    sm.register("/bob/jeff", "jaythegeek-jeff")
    assert sm.lookup("/alice/jeff") == "jaythegeek-jeff"
    assert sm.lookup("/bob/jeff") == "jaythegeek-jeff"

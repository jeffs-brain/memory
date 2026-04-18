// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"path/filepath"
	"testing"
)

func TestSlugMap_LookupMiss(t *testing.T) {
	sm := NewSlugMap(filepath.Join(t.TempDir(), "slug-map.yaml"))
	_, ok := sm.Lookup("/nonexistent/path")
	if ok {
		t.Error("expected miss for unknown path")
	}
}

func TestSlugMap_RegisterAndLookup(t *testing.T) {
	sm := NewSlugMap(filepath.Join(t.TempDir(), "slug-map.yaml"))

	if err := sm.Register("/home/alex/code/jeff", "jaythegeek-jeff"); err != nil {
		t.Fatal(err)
	}

	slug, ok := sm.Lookup("/home/alex/code/jeff")
	if !ok {
		t.Fatal("expected hit after Register")
	}
	if slug != "jaythegeek-jeff" {
		t.Errorf("slug = %q, want %q", slug, "jaythegeek-jeff")
	}
}

func TestSlugMap_PersistsAcrossLoadSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "slug-map.yaml")

	sm1 := NewSlugMap(path)
	if err := sm1.Register("/home/alex/code/jeff", "jaythegeek-jeff"); err != nil {
		t.Fatal(err)
	}
	if err := sm1.Register("/home/alex/projects/jeff", "jaythegeek-jeff"); err != nil {
		t.Fatal(err)
	}

	sm2 := NewSlugMap(path)
	sm2.Load()

	slug, ok := sm2.Lookup("/home/alex/code/jeff")
	if !ok || slug != "jaythegeek-jeff" {
		t.Errorf("after reload: /home/alex/code/jeff → %q (ok=%v)", slug, ok)
	}

	slug, ok = sm2.Lookup("/home/alex/projects/jeff")
	if !ok || slug != "jaythegeek-jeff" {
		t.Errorf("after reload: /home/alex/projects/jeff → %q (ok=%v)", slug, ok)
	}
}

func TestSlugMap_OverwriteExisting(t *testing.T) {
	sm := NewSlugMap(filepath.Join(t.TempDir(), "slug-map.yaml"))

	if err := sm.Register("/project", "old-slug"); err != nil {
		t.Fatal(err)
	}
	if err := sm.Register("/project", "new-slug"); err != nil {
		t.Fatal(err)
	}

	slug, ok := sm.Lookup("/project")
	if !ok || slug != "new-slug" {
		t.Errorf("slug = %q (ok=%v), want %q", slug, ok, "new-slug")
	}
}

func TestSlugMap_LoadMissingFile(t *testing.T) {
	sm := NewSlugMap(filepath.Join(t.TempDir(), "does-not-exist", "slug-map.yaml"))
	sm.Load()

	_, ok := sm.Lookup("/anything")
	if ok {
		t.Error("expected miss after loading missing file")
	}
}

func TestSlugMap_MultipleProjectsSameSlug(t *testing.T) {
	sm := NewSlugMap(filepath.Join(t.TempDir(), "slug-map.yaml"))

	if err := sm.Register("/home/alice/jeff", "jaythegeek-jeff"); err != nil {
		t.Fatal(err)
	}
	if err := sm.Register("/home/bob/jeff", "jaythegeek-jeff"); err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{"/home/alice/jeff", "/home/bob/jeff"} {
		slug, ok := sm.Lookup(path)
		if !ok || slug != "jaythegeek-jeff" {
			t.Errorf("Lookup(%q) = %q (ok=%v), want %q", path, slug, ok, "jaythegeek-jeff")
		}
	}
}

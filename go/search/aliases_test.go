// SPDX-License-Identifier: Apache-2.0

package search

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// TestLoadAliasMap_Success loads the committed testdata fixture and
// asserts every trigger lines up with its expected expansion. The
// loader is responsible for lowercasing both keys and values and for
// deduplicating repeats.
func TestLoadAliasMap_Success(t *testing.T) {
	m, err := LoadAliasMap(filepath.Join("testdata", "aliases.yaml"))
	if err != nil {
		t.Fatalf("LoadAliasMap: unexpected error: %v", err)
	}
	if m == nil {
		t.Fatal("LoadAliasMap: returned nil map")
	}

	cases := []struct {
		token string
		want  []string
	}{
		{"e-volt", []string{"e-volt", "nova-e-volt", "nova-evolt"}},
		{"E-Volt", []string{"e-volt", "nova-e-volt", "nova-evolt"}},
		{"evolt", []string{"e-volt", "nova-evolt"}},
		{"zenco", []string{"zenco", "zenco-group"}},
		{"hill", []string{"mill"}},
		{"brook", []string{"mill-brook", "brook"}},
	}

	for _, tc := range cases {
		got := m.Expand(tc.token)
		sorted := append([]string(nil), got...)
		sort.Strings(sorted)
		want := append([]string(nil), tc.want...)
		sort.Strings(want)
		if !stringSlicesEqual(sorted, want) {
			t.Errorf("Expand(%q) = %v, want %v", tc.token, got, tc.want)
		}
	}
}

// TestLoadAliasMap_Missing asserts the loader treats an absent file
// as "no aliases configured" rather than an error.
func TestLoadAliasMap_Missing(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "nope.yaml")

	m, err := LoadAliasMap(path)
	if err != nil {
		t.Fatalf("LoadAliasMap missing file: unexpected error: %v", err)
	}
	if m == nil {
		t.Fatal("LoadAliasMap missing file: returned nil map")
	}
	if m.Len() != 0 {
		t.Errorf("LoadAliasMap missing file: Len() = %d, want 0", m.Len())
	}
	if got := m.Expand("zenco"); len(got) != 1 || got[0] != "zenco" {
		t.Errorf("Expand on empty map = %v, want [zenco]", got)
	}
}

// TestLoadAliasMap_Malformed asserts that a syntactically invalid
// YAML file surfaces an error so callers can decide whether to log
// and continue.
func TestLoadAliasMap_Malformed(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "bad.yaml")
	if err := os.WriteFile(path, []byte("aliases: [not, a, map\n"), 0o600); err != nil {
		t.Fatalf("write malformed fixture: %v", err)
	}

	if _, err := LoadAliasMap(path); err == nil {
		t.Fatal("LoadAliasMap malformed: expected error, got nil")
	}
}

// TestAliasMap_Expand covers the lookup semantics directly: hit
// returns alternatives, miss echoes the input, case is ignored, and
// a nil receiver still behaves as a pass-through.
func TestAliasMap_Expand(t *testing.T) {
	m := NewAliasMap()
	m.entries["zenco"] = []string{"zenco", "zenco-group"}
	m.entries["hill"] = []string{"mill"}

	if got := m.Expand("zenco"); len(got) != 2 || got[0] != "zenco" || got[1] != "zenco-group" {
		t.Errorf("Expand(zenco) = %v", got)
	}
	if got := m.Expand("ZENCO"); len(got) != 2 || got[0] != "zenco" || got[1] != "zenco-group" {
		t.Errorf("Expand(ZENCO) case-insensitive failed: %v", got)
	}
	if got := m.Expand("lleverage"); len(got) != 1 || got[0] != "lleverage" {
		t.Errorf("Expand(miss) = %v, want [lleverage]", got)
	}
	if got := m.Expand("  hill  "); len(got) != 1 || got[0] != "mill" {
		t.Errorf("Expand(whitespace) = %v, want [mill]", got)
	}

	var nilMap *AliasMap
	if got := nilMap.Expand("zenco"); len(got) != 1 || got[0] != "zenco" {
		t.Errorf("nil receiver Expand = %v, want [zenco]", got)
	}
}

// TestAliasMapFromEntries exercises the programmatic builder that
// the spec's alias-table contract expects every SDK to expose.
func TestAliasMapFromEntries(t *testing.T) {
	m := AliasMapFromEntries(map[string][]string{
		"  ZENCO  ": {"Zenco Group", "  zenco  ", "zenco group"},
		"empty":     {},
		"":          {"nothing"},
	})
	if m.Len() != 1 {
		t.Errorf("Len() = %d, want 1", m.Len())
	}
	got := m.Expand("zenco")
	want := []string{"zenco group", "zenco"}
	sort.Strings(got)
	sort.Strings(want)
	if !stringSlicesEqual(got, want) {
		t.Errorf("Expand(zenco) = %v, want %v", got, want)
	}
}

// TestDefaultAliasPath asserts the helper honours XDG_CONFIG_HOME
// and falls back to $HOME/.config. Never touches the real home
// directory.
func TestDefaultAliasPath(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "/tmp/xdg-fake")
	if got := DefaultAliasPath(); got != filepath.Join("/tmp/xdg-fake", "jeff", "aliases.yaml") {
		t.Errorf("DefaultAliasPath with XDG = %q", got)
	}

	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("HOME", "/tmp/home-fake")
	if got := DefaultAliasPath(); got != filepath.Join("/tmp/home-fake", ".config", "jeff", "aliases.yaml") {
		t.Errorf("DefaultAliasPath with HOME = %q", got)
	}
}

// stringSlicesEqual is a local helper so the assertions above stay
// readable.
func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

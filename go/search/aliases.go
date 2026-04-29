// SPDX-License-Identifier: Apache-2.0

package search

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// AliasMap is an in-memory entity alias lookup. Keys are lowercase
// trigger tokens; values are the alternatives to emit as an OR
// expansion. A token that is not in the map is returned as-is via
// Expand.
//
// The runtime shape mirrors the contract in spec/QUERY-DSL.md
// (Alias tables section): persistence is reserved for a future spec
// revision; only the in-memory map form is normative. Callers build
// it programmatically — typically by iterating their domain-specific
// entity store — and pass it to the parser via [SetAliasMap].
type AliasMap struct {
	entries map[string][]string
}

// aliasFile mirrors the on-disk YAML shape so yaml.v3 can unmarshal
// into a strongly typed struct before we normalise the entries.
type aliasFile struct {
	Aliases map[string][]string `yaml:"aliases"`
}

// NewAliasMap constructs an empty AliasMap. Use LoadAliasMap to
// populate from YAML or [AliasMapFromEntries] to load from a Go map.
func NewAliasMap() *AliasMap {
	return &AliasMap{entries: map[string][]string{}}
}

// AliasMapFromEntries builds an AliasMap from a Go map. Keys and values
// are lowercased and trimmed; duplicate values inside the same entry
// are collapsed. Empty entries (zero alternatives after cleanup) are
// dropped.
//
// This is the normative construction path for the spec's runtime
// alias table contract: SDKs MUST accept an in-memory shape
// equivalent to ReadonlyMap<string, readonly string[]>. File-backed
// loaders are a convenience layered on top (see [LoadAliasMap]).
func AliasMapFromEntries(entries map[string][]string) *AliasMap {
	out := NewAliasMap()
	for key, vals := range entries {
		trigger := strings.ToLower(strings.TrimSpace(key))
		if trigger == "" {
			continue
		}
		seen := map[string]bool{}
		var cleaned []string
		for _, v := range vals {
			v = strings.ToLower(strings.TrimSpace(v))
			if v == "" || seen[v] {
				continue
			}
			seen[v] = true
			cleaned = append(cleaned, v)
		}
		if len(cleaned) == 0 {
			continue
		}
		out.entries[trigger] = cleaned
	}
	return out
}

// LoadAliasMap reads an alias YAML file and returns the parsed map.
// A missing file returns an empty AliasMap with nil error so callers
// can treat absence as "no aliases configured". Malformed YAML or
// unreadable files return an error; the caller decides whether to
// log and continue.
func LoadAliasMap(path string) (*AliasMap, error) {
	if path == "" {
		return NewAliasMap(), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return NewAliasMap(), nil
		}
		return nil, fmt.Errorf("read alias file %s: %w", path, err)
	}

	if len(strings.TrimSpace(string(data))) == 0 {
		return NewAliasMap(), nil
	}

	var parsed aliasFile
	if err := yaml.Unmarshal(data, &parsed); err != nil {
		return nil, fmt.Errorf("parse alias file %s: %w", path, err)
	}

	return AliasMapFromEntries(parsed.Aliases), nil
}

// Expand returns the alias alternatives for token, or a single-
// element slice containing the token verbatim when no alias matches.
// Lookup is case-insensitive. The returned slice is a fresh copy so
// callers may mutate it without affecting the map.
func (a *AliasMap) Expand(token string) []string {
	if a == nil || len(a.entries) == 0 {
		return []string{token}
	}
	key := strings.ToLower(strings.TrimSpace(token))
	if key == "" {
		return []string{token}
	}
	vals, ok := a.entries[key]
	if !ok || len(vals) == 0 {
		return []string{token}
	}
	out := make([]string, len(vals))
	copy(out, vals)
	return out
}

// Len returns the number of trigger entries in the alias map. Useful
// for tests and for logging how many aliases were loaded.
func (a *AliasMap) Len() int {
	if a == nil {
		return 0
	}
	return len(a.entries)
}

// DefaultAliasPath returns the user-editable alias file location at
// ~/.config/jeff/aliases.yaml. The function honours the XDG_CONFIG_HOME
// environment variable when set, otherwise falls back to $HOME/.config.
// On error it returns an empty string; callers should treat that as
// "no configured path" and skip alias loading.
func DefaultAliasPath() string {
	if xdg := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME")); xdg != "" {
		return filepath.Join(xdg, "jeff", "aliases.yaml")
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".config", "jeff", "aliases.yaml")
}

// aliasMapState is the package-wide alias map consulted by ParseQuery.
// Guarded by aliasMapMu so SetAliasMap is safe for concurrent use.
var (
	aliasMapMu    sync.RWMutex
	aliasMapState *AliasMap
)

// SetAliasMap registers a package-wide alias map consulted by
// [ParseQuery]. Passing nil disables aliasing. Safe for concurrent
// use.
func SetAliasMap(m *AliasMap) {
	aliasMapMu.Lock()
	defer aliasMapMu.Unlock()
	aliasMapState = m
}

// getAliasMap returns the currently registered alias map, or nil when
// none has been installed. The returned pointer is safe to call Expand
// on even if nil.
func getAliasMap() *AliasMap {
	aliasMapMu.RLock()
	defer aliasMapMu.RUnlock()
	return aliasMapState
}

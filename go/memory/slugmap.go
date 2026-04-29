// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

// SlugMap maps local project paths to canonical slugs. Each machine keeps
// its own map because filesystem paths are inherently machine-specific;
// only the resulting canonical slug travels with the brain store.
type SlugMap struct {
	mu      sync.Mutex
	entries map[string]string
	path    string
}

// slugMapPath returns the default location for the machine-local slug
// map: ~/.local/state/jeffs-brain/slug-map.yaml.
func slugMapPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".local", "state", "jeffs-brain", "slug-map.yaml")
}

// NewSlugMap creates a SlugMap backed by the given file path. Pass an
// empty string to use the default location.
func NewSlugMap(path string) *SlugMap {
	if path == "" {
		path = slugMapPath()
	}
	return &SlugMap{
		entries: make(map[string]string),
		path:    path,
	}
}

// Load reads the slug map from disk. Missing or unreadable files are
// silently ignored.
func (m *SlugMap) Load() {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, err := os.ReadFile(m.path)
	if err != nil {
		return
	}

	parsed := make(map[string]string)
	if err := yaml.Unmarshal(data, &parsed); err != nil {
		return
	}
	m.entries = parsed
}

// Save writes the slug map to disk, creating parent directories as
// needed.
func (m *SlugMap) Save() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.saveLocked()
}

func (m *SlugMap) saveLocked() error {
	if err := os.MkdirAll(filepath.Dir(m.path), 0o755); err != nil {
		return err
	}

	data, err := yaml.Marshal(m.entries)
	if err != nil {
		return err
	}
	return os.WriteFile(m.path, data, 0o644)
}

// Lookup returns the canonical slug for the given project path.
func (m *SlugMap) Lookup(projectPath string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	slug, ok := m.entries[projectPath]
	return slug, ok
}

// Register stores a path to slug mapping and persists the map.
func (m *SlugMap) Register(projectPath, slug string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[projectPath] = slug
	return m.saveLocked()
}

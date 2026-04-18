// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"log/slog"
	"net/url"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// slugState holds the process-wide slug map.
var slugState struct {
	once    sync.Once
	slugMap *SlugMap
}

func ensureSlugMap() *SlugMap {
	slugState.once.Do(func() {
		slugState.slugMap = NewSlugMap("")
		slugState.slugMap.Load()
	})
	return slugState.slugMap
}

// SetSlugMapForTest replaces the process-wide slug map with one backed
// by the given file. Returns a cleanup function.
func SetSlugMapForTest(path string) func() {
	prev := slugState.slugMap
	sm := NewSlugMap(path)
	sm.Load()
	slugState.slugMap = sm
	slugState.once.Do(func() {})
	return func() { slugState.slugMap = prev }
}

// ProjectSlug returns a safe directory name derived from the project
// path. It checks the slug map first, then derives a canonical slug
// from the git remote URL, and falls back to a path-based slug when
// there is no remote.
func ProjectSlug(projectPath string) string {
	sm := ensureSlugMap()

	abs, err := filepath.Abs(projectPath)
	if err != nil {
		abs = projectPath
	}

	if slug, ok := sm.Lookup(abs); ok {
		return slug
	}

	root := gitRoot(projectPath)
	canonical := ""
	if root != "" {
		if remoteURL := gitRemoteURL(root); remoteURL != "" {
			canonical = canonicalSlugFromRemote(remoteURL)
		}
	}

	if canonical == "" {
		base := root
		if base == "" {
			base = abs
		}
		canonical = pathBasedSlug(base)
	}

	if err := sm.Register(abs, canonical); err != nil {
		slog.Warn("memory: failed to save slug map", "err", err)
	}

	return canonical
}

// pathBasedSlug produces a slug from the filesystem path.
func pathBasedSlug(root string) string {
	slug := filepath.ToSlash(root)
	slug = strings.ReplaceAll(slug, "/", "-")
	slug = strings.TrimLeft(slug, "-")
	return slug
}

// canonicalSlugFromRemote extracts a canonical slug from a git remote
// URL. Handles SCP-style and URL-style remotes.
func canonicalSlugFromRemote(remoteURL string) string {
	remoteURL = strings.TrimSpace(remoteURL)
	if remoteURL == "" {
		return ""
	}

	var pathPart string

	if idx := strings.Index(remoteURL, "://"); idx < 0 {
		if colonIdx := strings.Index(remoteURL, ":"); colonIdx >= 0 {
			pathPart = remoteURL[colonIdx+1:]
		} else {
			return ""
		}
	} else {
		parsed, err := url.Parse(remoteURL)
		if err != nil || parsed.Path == "" {
			return ""
		}
		pathPart = parsed.Path
	}

	pathPart = strings.TrimSuffix(pathPart, ".git")
	pathPart = strings.Trim(pathPart, "/")

	if pathPart == "" {
		return ""
	}

	parts := strings.Split(pathPart, "/")
	slug := strings.Join(parts, "-")
	slug = strings.ToLower(slug)
	return slug
}

// gitRoot returns the top-level directory of the git repository
// containing projectPath. Returns an empty string if git is unavailable
// or the path is not inside a repository.
func gitRoot(projectPath string) string {
	if projectPath == "" {
		return ""
	}

	abs, err := filepath.Abs(projectPath)
	if err != nil {
		return ""
	}

	cmd := exec.Command("git", "rev-parse", "--show-toplevel")
	cmd.Dir = abs
	out, err := cmd.Output()
	if err != nil {
		return ""
	}

	return strings.TrimSpace(string(out))
}

// gitRemoteURL returns the fetch URL of the origin remote for the git
// repository at repoRoot.
func gitRemoteURL(repoRoot string) string {
	cmd := exec.Command("git", "remote", "get-url", "origin")
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

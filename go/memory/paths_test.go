// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestCanonicalSlugFromRemote(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		{"ssh scp-style", "git@github.com:jaythegeek/jeff.git", "jaythegeek-jeff"},
		{"https with .git", "https://github.com/jaythegeek/jeff.git", "jaythegeek-jeff"},
		{"https without .git", "https://github.com/jaythegeek/jeff", "jaythegeek-jeff"},
		{"ssh:// scheme with .git", "ssh://git@github.com/jaythegeek/jeff.git", "jaythegeek-jeff"},
		{"ssh:// scheme without .git", "ssh://git@github.com/jaythegeek/jeff", "jaythegeek-jeff"},
		{"gitlab nested subgroup", "git@gitlab.com:org/subgroup/repo.git", "org-subgroup-repo"},
		{"https nested subgroup", "https://gitlab.com/org/subgroup/repo.git", "org-subgroup-repo"},
		{"uppercase normalisation", "git@github.com:JayTheGeek/Jeff.git", "jaythegeek-jeff"},
		{"ssh with port", "ssh://git@github.com:2222/jaythegeek/jeff.git", "jaythegeek-jeff"},
		{"trailing whitespace", "  git@github.com:jaythegeek/jeff.git  \n", "jaythegeek-jeff"},
		{"empty string", "", ""},
		{"whitespace only", "   ", ""},
		{"bare hostname no path", "git@github.com:", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := canonicalSlugFromRemote(tt.url)
			if got != tt.want {
				t.Errorf("canonicalSlugFromRemote(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}

func TestProjectSlug_UsesCanonicalWhenGitRemoteAvailable(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	tmp := t.TempDir()
	initGitRepo(t, tmp, "git@github.com:testorg/testrepo.git")

	mapPath := filepath.Join(t.TempDir(), "slug-map.yaml")
	restore := SetSlugMapForTest(mapPath)
	t.Cleanup(restore)

	slug := ProjectSlug(tmp)
	if slug != "testorg-testrepo" {
		t.Errorf("ProjectSlug = %q, want %q", slug, "testorg-testrepo")
	}
}

func TestProjectSlug_FallsBackToPathWhenNoRemote(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	tmp := t.TempDir()

	mapPath := filepath.Join(t.TempDir(), "slug-map.yaml")
	restore := SetSlugMapForTest(mapPath)
	t.Cleanup(restore)

	slug := ProjectSlug(tmp)
	if slug == "" {
		t.Error("slug should not be empty")
	}
	if want := pathBasedSlug(tmp); slug != want {
		t.Errorf("ProjectSlug = %q, want path-based slug %q", slug, want)
	}
}

func TestProjectSlug_CachesInSlugMap(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	tmp := t.TempDir()
	initGitRepo(t, tmp, "git@github.com:cachetest/repo.git")

	mapPath := filepath.Join(t.TempDir(), "slug-map.yaml")
	restore := SetSlugMapForTest(mapPath)
	t.Cleanup(restore)

	slug1 := ProjectSlug(tmp)
	if slug1 != "cachetest-repo" {
		t.Fatalf("first call: slug = %q, want %q", slug1, "cachetest-repo")
	}

	sm := NewSlugMap(mapPath)
	sm.Load()
	abs, _ := filepath.Abs(tmp)
	cached, ok := sm.Lookup(abs)
	if !ok {
		t.Fatal("slug not found in persisted slug map")
	}
	if cached != "cachetest-repo" {
		t.Errorf("cached slug = %q, want %q", cached, "cachetest-repo")
	}
}

func TestProjectSlug_GitRepoWithoutRemoteFallsBackToPath(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	tmp := t.TempDir()

	cmd := exec.Command("git", "init")
	cmd.Dir = tmp
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}

	mapPath := filepath.Join(t.TempDir(), "slug-map.yaml")
	restore := SetSlugMapForTest(mapPath)
	t.Cleanup(restore)

	slug := ProjectSlug(tmp)
	if slug == "" {
		t.Error("slug should not be empty")
	}
	root := gitRoot(tmp)
	if root == "" {
		t.Fatal("expected git root")
	}
	if want := pathBasedSlug(root); slug != want {
		t.Errorf("ProjectSlug = %q, want path-based slug %q", slug, want)
	}
}

func TestPathBasedSlug(t *testing.T) {
	tests := []struct {
		root string
		want string
	}{
		{"/home/alex/code/jeff", "home-alex-code-jeff"},
		{"/home/alex/projects/jeff", "home-alex-projects-jeff"},
		{"/tmp/test", "tmp-test"},
	}
	for _, tt := range tests {
		got := pathBasedSlug(tt.root)
		if got != tt.want {
			t.Errorf("pathBasedSlug(%q) = %q, want %q", tt.root, got, tt.want)
		}
	}
}

// initGitRepo creates a bare git repo in dir with the given origin URL.
func initGitRepo(t *testing.T, dir, remoteURL string) {
	t.Helper()
	for _, args := range [][]string{
		{"init"},
		{"remote", "add", "origin", remoteURL},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(),
			"GIT_CONFIG_NOSYSTEM=1",
			"HOME="+t.TempDir(),
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
}

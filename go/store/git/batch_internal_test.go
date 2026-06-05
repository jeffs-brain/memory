// SPDX-License-Identifier: Apache-2.0

package git

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestResetWorktreeWithGitCLIRestoresTrackedDeletion(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not available on PATH")
	}
	root := t.TempDir()
	runGitInternalTest(t, root, "init", "-b", "main")
	runGitInternalTest(t, root, "config", "user.email", "test@example.com")
	runGitInternalTest(t, root, "config", "user.name", "test")

	path := filepath.Join(root, "wiki", "kept.md")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte("kept"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	runGitInternalTest(t, root, "add", "wiki/kept.md")
	runGitInternalTest(t, root, "commit", "-m", "seed")
	head := runGitInternalTest(t, root, "rev-parse", "HEAD")

	if err := os.Remove(path); err != nil {
		t.Fatalf("remove tracked file: %v", err)
	}
	if err := resetWorktreeWithGitCLI(root, head); err != nil {
		t.Fatalf("resetWorktreeWithGitCLI: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("tracked file was not restored: %v", err)
	}
}

func runGitInternalTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, dir, err, out.String())
	}
	return out.String()
}

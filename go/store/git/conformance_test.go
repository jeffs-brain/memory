// SPDX-License-Identifier: Apache-2.0

package git_test

import (
	"context"
	"os/exec"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/brain/braintest"
	gitstore "github.com/jeffs-brain/memory/go/store/git"
)

// TestGitStoreConformance runs the shared [braintest] contract suite
// against the git-backed [brain.Store]. The gitstore exposes a real
// on-disk working tree (so LocalPath always resolves) and must handle
// batches against a brand-new empty repository correctly, so both
// capability flags are on.
func TestGitStoreConformance(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not available on PATH")
	}

	factory := func(t *testing.T) (brain.Store, func()) {
		t.Helper()
		ctx := context.Background()
		store, err := gitstore.New(ctx, gitstore.Options{
			Root:   t.TempDir(),
			Branch: "main",
		})
		if err != nil {
			t.Fatalf("gitstore.New: %v", err)
		}
		return store, func() { _ = store.Close() }
	}

	braintest.RunContract(t, factory, braintest.Capabilities{
		LocalPathAlwaysExists:  true,
		SupportsEmptyRepoBatch: true,
	})
}

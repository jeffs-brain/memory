// SPDX-License-Identifier: Apache-2.0

package fs_test

import (
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/brain/braintest"
	"github.com/jeffs-brain/memory/go/store/fs"
)

func TestFsStoreConformance(t *testing.T) {
	braintest.RunContract(t, func(t *testing.T) (brain.Store, func()) {
		store, err := fs.New(t.TempDir())
		if err != nil {
			t.Fatalf("fs.New: %v", err)
		}
		return store, func() { _ = store.Close() }
	}, braintest.Capabilities{
		LocalPathAlwaysExists: true,
	})
}

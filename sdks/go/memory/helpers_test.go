// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

// contains is a thin wrapper around [strings.Contains].
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

// newTestMemory returns a Memory backed by an in-memory store.
func newTestMemory(t *testing.T) (*Memory, brain.Store) {
	t.Helper()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	return New(store), store
}

// writeTopic writes a logical path to the store and fails the test on
// error.
func writeTopic(t *testing.T, store brain.Store, p brain.Path, content string) {
	t.Helper()
	if err := store.Write(context.Background(), p, []byte(content)); err != nil {
		t.Fatalf("store.Write %s: %v", p, err)
	}
}

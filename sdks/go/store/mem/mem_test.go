// SPDX-License-Identifier: Apache-2.0

package mem_test

import (
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/brain/braintest"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func TestMemStoreConformance(t *testing.T) {
	braintest.RunContract(t, func(t *testing.T) (brain.Store, func()) {
		store := mem.New()
		return store, func() { _ = store.Close() }
	}, braintest.Capabilities{})
}

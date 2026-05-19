// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

// newMemStore creates a fresh in-memory brain.Store for testing.
func newMemStore() brain.Store {
	return mem.New()
}

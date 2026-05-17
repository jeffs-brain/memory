// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// SyncCursor represents a position in an external service's change
// stream. The Value is opaque to the framework -- each connector
// decides what to store (timestamp, cursor token, change ID, etc.).
type SyncCursor struct {
	// Value is the opaque cursor value.
	Value string `json:"value"`
	// UpdatedAt records when the cursor was last persisted.
	UpdatedAt time.Time `json:"updatedAt"`
	// Metadata holds optional connector-specific cursor context.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// serialisedSyncState is the JSON wire format for sync state persistence.
type serialisedSyncState struct {
	Cursor     SyncCursor `json:"cursor"`
	Generation int64      `json:"generation"`
}

// SyncStateManager persists and retrieves sync cursors per connector
// per brain. Storage uses the brain Store at:
//
//	connector/<name>/<brainID>/sync-state.json
//
// Updates use optimistic concurrency via a generation counter.
//
// Known limitation: the generation counter uses read-then-write without
// compare-and-swap. Two concurrent SetCursor calls may both read
// generation N and write N+1, with the second silently overwriting the
// first. This is acceptable for V1 where a single connector instance
// handles one brain's sync.
type SyncStateManager struct {
	store brain.Store
}

// NewSyncStateManager creates a new sync state manager backed by the
// given brain Store.
func NewSyncStateManager(store brain.Store) *SyncStateManager {
	return &SyncStateManager{store: store}
}

func syncStatePath(connectorName, brainID string) brain.Path {
	return brain.Path(fmt.Sprintf("connector/%s/%s/sync-state.json", connectorName, brainID))
}

// GetCursor retrieves the last sync cursor for the given connector and
// brain. Returns (zero, false, nil) when no cursor exists.
func (m *SyncStateManager) GetCursor(ctx context.Context, connectorName, brainID string) (SyncCursor, bool, error) {
	data, err := m.store.Read(ctx, syncStatePath(connectorName, brainID))
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return SyncCursor{}, false, nil
		}
		return SyncCursor{}, false, fmt.Errorf("connector: sync state read: %w", err)
	}

	var state serialisedSyncState
	if err := json.Unmarshal(data, &state); err != nil {
		return SyncCursor{}, false, fmt.Errorf("connector: sync state decode: %w", err)
	}
	return state.Cursor, true, nil
}

// SetCursor persists a sync cursor. Uses optimistic concurrency: the
// generation counter is incremented on each write.
func (m *SyncStateManager) SetCursor(ctx context.Context, connectorName, brainID string, cursor SyncCursor) error {
	p := syncStatePath(connectorName, brainID)

	// Read the current generation for optimistic concurrency.
	var generation int64
	data, err := m.store.Read(ctx, p)
	if err == nil {
		var existing serialisedSyncState
		if jsonErr := json.Unmarshal(data, &existing); jsonErr == nil {
			generation = existing.Generation
		}
	} else if !errors.Is(err, brain.ErrNotFound) {
		return fmt.Errorf("connector: sync state read for update: %w", err)
	}

	cursor.UpdatedAt = time.Now().UTC()
	state := serialisedSyncState{
		Cursor:     cursor,
		Generation: generation + 1,
	}

	out, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("connector: sync state encode: %w", err)
	}

	if err := m.store.Write(ctx, p, out); err != nil {
		return fmt.Errorf("connector: sync state write: %w", err)
	}
	return nil
}

// ClearCursor removes the sync cursor, forcing a full sync on the next
// run. No error is returned if no cursor exists.
func (m *SyncStateManager) ClearCursor(ctx context.Context, connectorName, brainID string) error {
	err := m.store.Delete(ctx, syncStatePath(connectorName, brainID))
	if err != nil && !errors.Is(err, brain.ErrNotFound) {
		return fmt.Errorf("connector: sync state delete: %w", err)
	}
	return nil
}

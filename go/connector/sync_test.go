// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/connector"
)

func TestSyncStateManager_SetAndGet(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	cursor := connector.SyncCursor{
		Value:    "ts:1234567890.123456",
		Metadata: map[string]string{"channel": "C123"},
	}

	if err := mgr.SetCursor(ctx, "slack", "brain-1", cursor); err != nil {
		t.Fatalf("SetCursor: %v", err)
	}

	loaded, found, err := mgr.GetCursor(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor: %v", err)
	}
	if !found {
		t.Fatal("GetCursor returned found=false after SetCursor")
	}
	if loaded.Value != "ts:1234567890.123456" {
		t.Errorf("Value = %q, want %q", loaded.Value, "ts:1234567890.123456")
	}
	if loaded.UpdatedAt.IsZero() {
		t.Error("UpdatedAt should not be zero")
	}
	if loaded.Metadata["channel"] != "C123" {
		t.Errorf("Metadata[channel] = %q, want %q", loaded.Metadata["channel"], "C123")
	}
}

func TestSyncStateManager_GetNonExistent(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	_, found, err := mgr.GetCursor(ctx, "unknown", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor: %v", err)
	}
	if found {
		t.Error("GetCursor returned found=true for nonexistent cursor")
	}
}

func TestSyncStateManager_ClearCursor(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	cursor := connector.SyncCursor{Value: "cursor-abc"}
	if err := mgr.SetCursor(ctx, "slack", "brain-1", cursor); err != nil {
		t.Fatalf("SetCursor: %v", err)
	}

	if err := mgr.ClearCursor(ctx, "slack", "brain-1"); err != nil {
		t.Fatalf("ClearCursor: %v", err)
	}

	_, found, err := mgr.GetCursor(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor after clear: %v", err)
	}
	if found {
		t.Error("cursor found after ClearCursor")
	}
}

func TestSyncStateManager_MultiConnectorIsolation(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	slackCursor := connector.SyncCursor{Value: "slack-cursor"}
	gdriveCursor := connector.SyncCursor{Value: "gdrive-cursor"}

	if err := mgr.SetCursor(ctx, "slack", "brain-1", slackCursor); err != nil {
		t.Fatalf("SetCursor slack: %v", err)
	}
	if err := mgr.SetCursor(ctx, "gdrive", "brain-1", gdriveCursor); err != nil {
		t.Fatalf("SetCursor gdrive: %v", err)
	}

	loadedSlack, foundSlack, err := mgr.GetCursor(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor slack: %v", err)
	}
	if !foundSlack || loadedSlack.Value != "slack-cursor" {
		t.Errorf("slack cursor: found=%v, value=%q", foundSlack, loadedSlack.Value)
	}

	loadedGdrive, foundGdrive, err := mgr.GetCursor(ctx, "gdrive", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor gdrive: %v", err)
	}
	if !foundGdrive || loadedGdrive.Value != "gdrive-cursor" {
		t.Errorf("gdrive cursor: found=%v, value=%q", foundGdrive, loadedGdrive.Value)
	}
}

func TestSyncStateManager_MultiBrainIsolation(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	cursorA := connector.SyncCursor{Value: "cursor-brain-a"}
	cursorB := connector.SyncCursor{Value: "cursor-brain-b"}

	if err := mgr.SetCursor(ctx, "slack", "brain-a", cursorA); err != nil {
		t.Fatalf("SetCursor brain-a: %v", err)
	}
	if err := mgr.SetCursor(ctx, "slack", "brain-b", cursorB); err != nil {
		t.Fatalf("SetCursor brain-b: %v", err)
	}

	loadedA, foundA, err := mgr.GetCursor(ctx, "slack", "brain-a")
	if err != nil {
		t.Fatalf("GetCursor brain-a: %v", err)
	}
	if !foundA || loadedA.Value != "cursor-brain-a" {
		t.Errorf("brain-a cursor: found=%v, value=%q", foundA, loadedA.Value)
	}

	loadedB, foundB, err := mgr.GetCursor(ctx, "slack", "brain-b")
	if err != nil {
		t.Fatalf("GetCursor brain-b: %v", err)
	}
	if !foundB || loadedB.Value != "cursor-brain-b" {
		t.Errorf("brain-b cursor: found=%v, value=%q", foundB, loadedB.Value)
	}
}

func TestSyncStateManager_GenerationIncrement(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	// Write twice and verify the cursor is updated.
	cursor1 := connector.SyncCursor{Value: "v1"}
	if err := mgr.SetCursor(ctx, "slack", "brain-1", cursor1); err != nil {
		t.Fatalf("SetCursor v1: %v", err)
	}

	cursor2 := connector.SyncCursor{Value: "v2"}
	if err := mgr.SetCursor(ctx, "slack", "brain-1", cursor2); err != nil {
		t.Fatalf("SetCursor v2: %v", err)
	}

	loaded, found, err := mgr.GetCursor(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("GetCursor: %v", err)
	}
	if !found || loaded.Value != "v2" {
		t.Errorf("cursor: found=%v, value=%q, want v2", found, loaded.Value)
	}
}

func TestSyncStateManager_ClearNonExistent(t *testing.T) {
	store := newMemStore()
	mgr := connector.NewSyncStateManager(store)
	ctx := context.Background()

	// Clearing a nonexistent cursor should not return an error.
	if err := mgr.ClearCursor(ctx, "nonexistent", "brain-1"); err != nil {
		t.Fatalf("ClearCursor nonexistent: %v", err)
	}
}

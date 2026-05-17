// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/connector"
)

// newTestTokenStore creates a SecureTokenStore backed by a fresh
// in-memory brain Store.
func newTestTokenStore(t *testing.T) *connector.SecureTokenStore {
	t.Helper()
	store := newMemStore()
	ts, err := connector.NewSecureTokenStore(store, []byte("test-passphrase-at-least-16-bytes"))
	if err != nil {
		t.Fatalf("NewSecureTokenStore: %v", err)
	}
	return ts
}

func TestSecureTokenStore_SaveAndLoad(t *testing.T) {
	ts := newTestTokenStore(t)
	ctx := context.Background()

	token := connector.OAuth2Token{
		AccessToken:  "access-abc",
		RefreshToken: "refresh-def",
		ExpiresAt:    time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC),
		TokenType:    "Bearer",
		Scopes:       []string{"read", "write"},
	}

	if err := ts.Save(ctx, "slack", "brain-1", token); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, found, err := ts.Load(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !found {
		t.Fatal("Load returned found=false after Save")
	}
	if loaded.AccessToken != token.AccessToken {
		t.Errorf("AccessToken = %q, want %q", loaded.AccessToken, token.AccessToken)
	}
	if loaded.RefreshToken != token.RefreshToken {
		t.Errorf("RefreshToken = %q, want %q", loaded.RefreshToken, token.RefreshToken)
	}
	if !loaded.ExpiresAt.Equal(token.ExpiresAt) {
		t.Errorf("ExpiresAt = %v, want %v", loaded.ExpiresAt, token.ExpiresAt)
	}
	if loaded.TokenType != token.TokenType {
		t.Errorf("TokenType = %q, want %q", loaded.TokenType, token.TokenType)
	}
	if len(loaded.Scopes) != len(token.Scopes) {
		t.Errorf("Scopes length = %d, want %d", len(loaded.Scopes), len(token.Scopes))
	}
}

func TestSecureTokenStore_LoadNotFound(t *testing.T) {
	ts := newTestTokenStore(t)
	ctx := context.Background()

	_, found, err := ts.Load(ctx, "nonexistent", "brain-1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if found {
		t.Error("Load returned found=true for nonexistent token")
	}
}

func TestSecureTokenStore_Delete(t *testing.T) {
	ts := newTestTokenStore(t)
	ctx := context.Background()

	token := connector.OAuth2Token{
		AccessToken: "access",
		ExpiresAt:   time.Now().Add(time.Hour),
	}

	if err := ts.Save(ctx, "slack", "brain-1", token); err != nil {
		t.Fatalf("Save: %v", err)
	}

	if err := ts.Delete(ctx, "slack", "brain-1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	_, found, err := ts.Load(ctx, "slack", "brain-1")
	if err != nil {
		t.Fatalf("Load after Delete: %v", err)
	}
	if found {
		t.Error("token still found after Delete")
	}
}

func TestSecureTokenStore_DeleteNonexistent(t *testing.T) {
	ts := newTestTokenStore(t)
	ctx := context.Background()

	// Deleting a nonexistent token should not return an error.
	if err := ts.Delete(ctx, "nonexistent", "brain-1"); err != nil {
		t.Fatalf("Delete nonexistent: %v", err)
	}
}

func TestSecureTokenStore_MultiTenantIsolation(t *testing.T) {
	ts := newTestTokenStore(t)
	ctx := context.Background()

	tokenA := connector.OAuth2Token{
		AccessToken: "access-brain-a",
		ExpiresAt:   time.Now().Add(time.Hour),
	}
	tokenB := connector.OAuth2Token{
		AccessToken: "access-brain-b",
		ExpiresAt:   time.Now().Add(time.Hour),
	}

	if err := ts.Save(ctx, "slack", "brain-a", tokenA); err != nil {
		t.Fatalf("Save brain-a: %v", err)
	}
	if err := ts.Save(ctx, "slack", "brain-b", tokenB); err != nil {
		t.Fatalf("Save brain-b: %v", err)
	}

	loadedA, foundA, err := ts.Load(ctx, "slack", "brain-a")
	if err != nil {
		t.Fatalf("Load brain-a: %v", err)
	}
	if !foundA || loadedA.AccessToken != "access-brain-a" {
		t.Errorf("brain-a token mismatch: found=%v, access=%q", foundA, loadedA.AccessToken)
	}

	loadedB, foundB, err := ts.Load(ctx, "slack", "brain-b")
	if err != nil {
		t.Fatalf("Load brain-b: %v", err)
	}
	if !foundB || loadedB.AccessToken != "access-brain-b" {
		t.Errorf("brain-b token mismatch: found=%v, access=%q", foundB, loadedB.AccessToken)
	}
}

func TestSecureTokenStore_WrongKey(t *testing.T) {
	store := newMemStore()
	ctx := context.Background()

	ts1, err := connector.NewSecureTokenStore(store, []byte("passphrase-one-at-least-16-bytes"))
	if err != nil {
		t.Fatalf("NewSecureTokenStore 1: %v", err)
	}

	token := connector.OAuth2Token{
		AccessToken: "secret-access",
		ExpiresAt:   time.Now().Add(time.Hour),
	}
	if err := ts1.Save(ctx, "slack", "brain-1", token); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load with a different key should fail.
	ts2, err := connector.NewSecureTokenStore(store, []byte("passphrase-two-at-least-16-bytes"))
	if err != nil {
		t.Fatalf("NewSecureTokenStore 2: %v", err)
	}

	_, _, err = ts2.Load(ctx, "slack", "brain-1")
	if err == nil {
		t.Fatal("expected decryption error with wrong key")
	}
}

func TestSecureTokenStore_ShortPassphrase(t *testing.T) {
	store := newMemStore()
	_, err := connector.NewSecureTokenStore(store, []byte("short"))
	if err == nil {
		t.Fatal("expected error for short passphrase")
	}
}

// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/cmd/memory-mcp/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// newInMemoryServer wires a server + client pair via in-memory
// transports so the stdio boundary does not become the focus of the
// test. Returns the client session ready to call the tool surface.
func newInMemoryServer(t *testing.T, home string) *mcp.ClientSession {
	t.Helper()
	t.Setenv("JB_HOME", home)
	t.Setenv("JB_TOKEN", "")
	t.Setenv("JB_BRAIN", "")
	t.Setenv("JB_LLM_PROVIDER", "fake")
	t.Setenv("JB_EMBED_PROVIDER", "fake")

	cfg, err := ResolveConfig(os.Getenv)
	if err != nil {
		t.Fatalf("resolve config: %v", err)
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
	memClient, err := NewMemoryClient(cfg, logger)
	if err != nil {
		t.Fatalf("new memory client: %v", err)
	}
	t.Cleanup(func() { _ = memClient.Close() })

	server := mcp.NewServer(&mcp.Implementation{Name: serverName, Version: serverVersion}, nil)
	tools.Register(server, toolAdapter{client: memClient})

	serverTransport, clientTransport := mcp.NewInMemoryTransports()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "v0.0.1"}, nil)
	session, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

// callTool issues a tools/call request and returns the decoded
// StructuredContent payload.
func callTool(t *testing.T, session *mcp.ClientSession, name string, args map[string]any) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if res.IsError {
		txt := ""
		for _, c := range res.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				txt = tc.Text
			}
		}
		t.Fatalf("%s returned error: %s", name, txt)
	}
	if res.StructuredContent == nil {
		// Fall back to parsing the text content — the SDK occasionally
		// renders structured payloads purely as JSON text.
		for _, c := range res.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				var out map[string]any
				if err := json.Unmarshal([]byte(tc.Text), &out); err != nil {
					t.Fatalf("%s: decode text: %v", name, err)
				}
				return out
			}
		}
		return map[string]any{}
	}
	raw, err := json.Marshal(res.StructuredContent)
	if err != nil {
		t.Fatalf("%s: marshal structured: %v", name, err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%s: decode structured: %v", name, err)
	}
	return out
}

// TestListToolsReturnsElevenTools verifies the tool surface matches the
// canonical spec.
func TestListToolsReturnsElevenTools(t *testing.T) {
	home := t.TempDir()
	session := newInMemoryServer(t, home)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	listed, err := session.ListTools(ctx, nil)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	names := map[string]bool{}
	for _, tool := range listed.Tools {
		names[tool.Name] = true
	}
	expected := []string{
		"memory_remember",
		"memory_recall",
		"memory_search",
		"memory_ask",
		"memory_ingest_file",
		"memory_ingest_url",
		"memory_extract",
		"memory_reflect",
		"memory_consolidate",
		"memory_create_brain",
		"memory_list_brains",
	}
	for _, name := range expected {
		if !names[name] {
			t.Errorf("missing tool: %s", name)
		}
	}
	if len(listed.Tools) != len(expected) {
		t.Errorf("expected %d tools, got %d", len(expected), len(listed.Tools))
	}
}

// TestEmptyListBrains verifies list returns an empty array on a fresh
// JB_HOME.
func TestEmptyListBrains(t *testing.T) {
	home := t.TempDir()
	session := newInMemoryServer(t, home)

	out := callTool(t, session, "memory_list_brains", map[string]any{})
	items, ok := out["items"].([]any)
	if !ok {
		t.Fatalf("memory_list_brains: items missing or wrong type: %#v", out["items"])
	}
	if len(items) != 0 {
		t.Fatalf("expected empty list, got %d", len(items))
	}
}

// TestCreateAndListBrains confirms a created brain round-trips through
// the listing endpoint.
func TestCreateAndListBrains(t *testing.T) {
	home := t.TempDir()
	session := newInMemoryServer(t, home)

	created := callTool(t, session, "memory_create_brain", map[string]any{
		"name": "Test Brain",
		"slug": "test-brain",
	})
	if slug, _ := created["slug"].(string); slug != "test-brain" {
		t.Fatalf("expected slug=test-brain, got %v", created["slug"])
	}

	listed := callTool(t, session, "memory_list_brains", map[string]any{})
	items, _ := listed["items"].([]any)
	if len(items) != 1 {
		t.Fatalf("expected 1 brain, got %d", len(items))
	}
	first, _ := items[0].(map[string]any)
	if slug, _ := first["slug"].(string); slug != "test-brain" {
		t.Fatalf("expected slug=test-brain, got %v", first["slug"])
	}
}

// TestIngestFileAndSearch ingests a tiny markdown fixture and verifies
// a search returns at least one hit.
func TestIngestFileAndSearch(t *testing.T) {
	home := t.TempDir()
	fixture := filepath.Join(t.TempDir(), "note.md")
	body := "# Saturday notes\n\nI ran a 10k in under 55 minutes at the park.\n"
	if err := os.WriteFile(fixture, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	session := newInMemoryServer(t, home)

	_ = callTool(t, session, "memory_create_brain", map[string]any{
		"name": "Default",
		"slug": "default",
	})

	ingest := callTool(t, session, "memory_ingest_file", map[string]any{
		"path":  fixture,
		"brain": "default",
		"as":    "markdown",
	})
	if status, _ := ingest["status"].(string); status == "" {
		t.Fatalf("expected status in ingest response, got %#v", ingest)
	}

	search := callTool(t, session, "memory_search", map[string]any{
		"query": "Saturday",
		"brain": "default",
	})
	hits, _ := search["hits"].([]any)
	if len(hits) == 0 {
		t.Fatalf("expected at least one search hit, got response %#v", search)
	}
}

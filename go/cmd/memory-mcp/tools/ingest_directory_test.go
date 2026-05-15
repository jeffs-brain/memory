// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mockDirMemoryClient is a minimal MemoryClient for testing directory ingest.
type mockDirMemoryClient struct {
	ingestFileFn func(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error)
}

func (m *mockDirMemoryClient) Remember(context.Context, RememberArgs) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Search(context.Context, SearchArgs) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Recall(context.Context, RecallArgs) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Ask(context.Context, AskArgs, ProgressEmitter) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error) {
	if m.ingestFileFn != nil {
		return m.ingestFileFn(ctx, args, progress)
	}
	return map[string]any{
		"status":      "completed",
		"document_id": "doc-" + filepath.Base(args.Path),
		"hash":        "hash-" + filepath.Base(args.Path),
		"byte_size":   100,
	}, nil
}
func (m *mockDirMemoryClient) IngestURL(context.Context, IngestURLArgs, ProgressEmitter) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Extract(context.Context, ExtractArgs, ProgressEmitter) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Reflect(context.Context, ReflectArgs, ProgressEmitter) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) Consolidate(context.Context, ConsolidateArgs, ProgressEmitter) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) CreateBrain(context.Context, CreateBrainArgs) (map[string]any, error) {
	return nil, nil
}
func (m *mockDirMemoryClient) ListBrains(context.Context) (map[string]any, error) { return nil, nil }

func setupDirTestServer(t *testing.T, client MemoryClient) *mcp.ClientSession {
	t.Helper()
	server := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "0.0.0"}, nil)
	registerIngestDirectory(server, client)

	serverTransport, clientTransport := mcp.NewInMemoryTransports()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
		t.Fatalf("server connect: %v", err)
	}
	mcpClient := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "0.0.1"}, nil)
	session, err := mcpClient.Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatalf("client connect: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func callDirTool(t *testing.T, session *mcp.ClientSession, args map[string]any) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "memory_ingest_directory", Arguments: args})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError {
		txt := ""
		for _, c := range res.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				txt = tc.Text
			}
		}
		t.Fatalf("memory_ingest_directory returned error: %s", txt)
	}

	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			var out map[string]any
			if err := json.Unmarshal([]byte(tc.Text), &out); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			return out
		}
	}
	t.Fatal("no text content in response")
	return nil
}

func TestIngestDirectory_EnumeratesAndIngests(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.md", "b.md", "c.md"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("# "+name), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	calls := []string{}
	client := &mockDirMemoryClient{
		ingestFileFn: func(_ context.Context, args IngestFileArgs, _ ProgressEmitter) (map[string]any, error) {
			calls = append(calls, args.Path)
			return map[string]any{
				"status":      "completed",
				"document_id": "doc-" + filepath.Base(args.Path),
				"hash":        "hash-" + filepath.Base(args.Path),
				"byte_size":   100,
			}, nil
		},
	}
	session := setupDirTestServer(t, client)

	payload := callDirTool(t, session, map[string]any{
		"directory": dir,
	})

	if payload["total"] != float64(3) {
		t.Errorf("expected total=3, got %v", payload["total"])
	}
	if payload["succeeded"] != float64(3) {
		t.Errorf("expected succeeded=3, got %v", payload["succeeded"])
	}
	if payload["failed"] != float64(0) {
		t.Errorf("expected failed=0, got %v", payload["failed"])
	}
	if len(calls) != 3 {
		t.Errorf("expected 3 ingest calls, got %d", len(calls))
	}
}

func TestIngestDirectory_MaxFilesOver500Rejected(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.md"), []byte("content"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	client := &mockDirMemoryClient{}
	session := setupDirTestServer(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{
		Name: "memory_ingest_directory",
		Arguments: map[string]any{
			"directory": dir,
			"maxFiles":  501,
		},
	})
	// The MCP SDK may reject via schema validation (error) or via
	// the handler guard (IsError response). Either is acceptable.
	if err != nil {
		return // Schema validation rejected it.
	}
	if !res.IsError {
		t.Fatal("expected error for maxFiles > 500, but got success")
	}
}

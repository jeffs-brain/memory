// SPDX-License-Identifier: Apache-2.0

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mockMemoryClient is a minimal MemoryClient for testing batch ingest.
type mockMemoryClient struct {
	ingestFileFn func(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (*IngestResult, error)
}

func (m *mockMemoryClient) Remember(context.Context, RememberArgs) (*RememberResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Search(context.Context, SearchArgs) (*SearchResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Recall(context.Context, RecallArgs) (*RecallResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Ask(context.Context, AskArgs, ProgressEmitter) (*AskResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (*IngestResult, error) {
	if m.ingestFileFn != nil {
		return m.ingestFileFn(ctx, args, progress)
	}
	return &IngestResult{
		Status:     "completed",
		DocumentID: "doc-" + args.Path,
		Hash:       "hash-" + args.Path,
	}, nil
}
func (m *mockMemoryClient) IngestURL(context.Context, IngestURLArgs, ProgressEmitter) (*IngestURLResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) ExtractAfterIngest(context.Context, ExtractAfterIngestArgs) (*ExtractAfterIngestResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Extract(context.Context, ExtractArgs, ProgressEmitter) (*ExtractResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Reflect(context.Context, ReflectArgs, ProgressEmitter) (*ReflectResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) Consolidate(context.Context, ConsolidateArgs, ProgressEmitter) (*ConsolidateResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) CreateBrain(context.Context, CreateBrainArgs) (*CreateBrainResult, error) {
	return nil, nil
}
func (m *mockMemoryClient) ListBrains(context.Context) (*ListBrainsResult, error) { return nil, nil }

func setupBatchTestServer(t *testing.T, client MemoryClient) *mcp.ClientSession {
	t.Helper()
	server := mcp.NewServer(&mcp.Implementation{Name: "test", Version: "0.0.0"}, nil)
	registerIngestBatch(server, client)

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

func callBatchTool(t *testing.T, session *mcp.ClientSession, args map[string]any) map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "memory_ingest_batch", Arguments: args})
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
		t.Fatalf("memory_ingest_batch returned error: %s", txt)
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

func TestIngestBatch_AllSucceed(t *testing.T) {
	client := &mockMemoryClient{}
	session := setupBatchTestServer(t, client)

	payload := callBatchTool(t, session, map[string]any{
		"files": []map[string]any{
			{"path": "/a.md"},
			{"path": "/b.md"},
			{"path": "/c.md"},
		},
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
	results, ok := payload["results"].([]any)
	if !ok || len(results) != 3 {
		t.Fatalf("expected 3 results, got %v", payload["results"])
	}
	first, ok := results[0].(map[string]any)
	if !ok {
		t.Fatal("expected map result entry")
	}
	if first["status"] != "success" {
		t.Errorf("expected status=success, got %v", first["status"])
	}
}

func TestIngestBatch_PerFileErrorIsolation(t *testing.T) {
	client := &mockMemoryClient{
		ingestFileFn: func(_ context.Context, args IngestFileArgs, _ ProgressEmitter) (*IngestResult, error) {
			if args.Path == "/missing.md" {
				return nil, fmt.Errorf("file not found")
			}
			return &IngestResult{
				Status:     "completed",
				DocumentID: "doc-" + args.Path,
				Hash:       "hash-" + args.Path,
			}, nil
		},
	}
	session := setupBatchTestServer(t, client)

	payload := callBatchTool(t, session, map[string]any{
		"files": []map[string]any{
			{"path": "/a.md"},
			{"path": "/missing.md"},
			{"path": "/c.md"},
		},
	})

	if payload["succeeded"] != float64(2) {
		t.Errorf("expected succeeded=2, got %v", payload["succeeded"])
	}
	if payload["failed"] != float64(1) {
		t.Errorf("expected failed=1, got %v", payload["failed"])
	}

	results := payload["results"].([]any)
	second := results[1].(map[string]any)
	if second["status"] != "error" {
		t.Errorf("expected status=error for file 2, got %v", second["status"])
	}
	if second["error"] != "file not found" {
		t.Errorf("expected error message, got %v", second["error"])
	}
	third := results[2].(map[string]any)
	if third["status"] != "success" {
		t.Errorf("expected status=success for file 3, got %v", third["status"])
	}
}

func TestIngestBatch_DuplicateFiles(t *testing.T) {
	var mu sync.Mutex
	var callCount int
	client := &mockMemoryClient{
		ingestFileFn: func(_ context.Context, args IngestFileArgs, _ ProgressEmitter) (*IngestResult, error) {
			mu.Lock()
			callCount++
			n := callCount
			mu.Unlock()
			return &IngestResult{
				Status:     "completed",
				DocumentID: fmt.Sprintf("doc-%d", n),
				Hash:       fmt.Sprintf("hash-%d", n),
			}, nil
		},
	}
	session := setupBatchTestServer(t, client)

	payload := callBatchTool(t, session, map[string]any{
		"files": []map[string]any{
			{"path": "/same.md"},
			{"path": "/same.md"},
		},
	})

	if payload["total"] != float64(2) {
		t.Errorf("expected total=2, got %v", payload["total"])
	}
	if payload["succeeded"] != float64(2) {
		t.Errorf("expected succeeded=2, got %v", payload["succeeded"])
	}
	if payload["failed"] != float64(0) {
		t.Errorf("expected failed=0, got %v", payload["failed"])
	}
	mu.Lock()
	if callCount != 2 {
		t.Fatalf("expected 2 calls, got %d", callCount)
	}
	mu.Unlock()
}

func TestIngestBatch_BrainAppliedToAll(t *testing.T) {
	var mu sync.Mutex
	brainsSeen := []string{}
	client := &mockMemoryClient{
		ingestFileFn: func(_ context.Context, args IngestFileArgs, _ ProgressEmitter) (*IngestResult, error) {
			mu.Lock()
			brainsSeen = append(brainsSeen, args.Brain)
			mu.Unlock()
			return &IngestResult{Status: "completed", DocumentID: "d", Hash: "h"}, nil
		},
	}
	session := setupBatchTestServer(t, client)

	callBatchTool(t, session, map[string]any{
		"files": []map[string]any{
			{"path": "/a.md"},
			{"path": "/b.md"},
		},
		"brain": "test-brain",
	})

	mu.Lock()
	defer mu.Unlock()
	if len(brainsSeen) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(brainsSeen))
	}
	for i, b := range brainsSeen {
		if b != "test-brain" {
			t.Errorf("call %d: expected brain=test-brain, got %s", i, b)
		}
	}
}

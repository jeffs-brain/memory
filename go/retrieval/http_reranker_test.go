// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func decodeRerankRequest(t *testing.T, r *http.Request) httpRerankRequest {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var parsed httpRerankRequest
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	return parsed
}

func serveHealthyProbe(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodGet {
		return false
	}
	switch r.URL.Path {
	case "/health", "/info":
		w.WriteHeader(http.StatusNoContent)
		return true
	default:
		return false
	}
}

func TestHTTPReranker_ReordersByRelevance(t *testing.T) {
	t.Parallel()
	var lastPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		lastPath = r.URL.Path
		req := decodeRerankRequest(t, r)
		if req.Query != "widgets" {
			t.Errorf("query = %q, want widgets", req.Query)
		}
		if len(req.Documents) != 3 {
			t.Errorf("documents = %d, want 3", len(req.Documents))
		}
		// Score index 2 highest, then 0, then 1.
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(httpRerankResponse{
			Results: []httpRerankHit{
				{Index: 2, RelevanceScore: 0.95},
				{Index: 0, RelevanceScore: 0.6},
				{Index: 1, RelevanceScore: 0.2},
			},
		})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		Model:    "test-model",
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	chunks := makeRerankChunks(3)
	out, err := rr.Rerank(context.Background(), "widgets", chunks)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("out length = %d, want 3", len(out))
	}
	if out[0].ChunkID != "c" {
		t.Errorf("top = %q, want c", out[0].ChunkID)
	}
	if out[0].RerankScore != 0.95 {
		t.Errorf("top score = %v, want 0.95", out[0].RerankScore)
	}
	if lastPath != "/v1/rerank" {
		t.Errorf("server hit path = %q, want /v1/rerank", lastPath)
	}
}

func TestHTTPReranker_ForwardsBearerToken(t *testing.T) {
	t.Parallel()
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if serveHealthyProbe(w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(httpRerankResponse{
			Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
		})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		APIKey:   "sk-test-42",
		Model:    "m",
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if gotAuth != "Bearer sk-test-42" {
		t.Errorf("Authorization = %q, want Bearer sk-test-42", gotAuth)
	}
}

func TestHTTPReranker_ServerErrorSurfaces(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		http.Error(w, "upstream exploded", http.StatusInternalServerError)
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: server.URL + "/v1/rerank"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	_, err = rr.Rerank(context.Background(), "q", makeRerankChunks(2))
	if err == nil {
		t.Fatal("expected 500 to surface as error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %v, want mention of 500", err)
	}
}

func TestHTTPReranker_EmptyCandidates(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be hit for empty candidates")
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: server.URL + "/v1/rerank"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	out, err := rr.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("Rerank empty: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("out = %d, want 0", len(out))
	}
}

func TestHTTPReranker_NoResultsSurfacesError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(httpRerankResponse{Results: nil})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: server.URL + "/v1/rerank"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	_, err = rr.Rerank(context.Background(), "q", makeRerankChunks(2))
	if err == nil {
		t.Fatal("expected no-results response to surface as error")
	}
	if !strings.Contains(err.Error(), "no scored documents") {
		t.Errorf("error = %v, want no-scored message", err)
	}
}

func TestHTTPReranker_DropsOutOfRangeIndexes(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(httpRerankResponse{
			Results: []httpRerankHit{
				{Index: 99, RelevanceScore: 10},
				{Index: 0, RelevanceScore: 1},
				{Index: 1, RelevanceScore: 5},
			},
		})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: server.URL + "/v1/rerank"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	out, err := rr.Rerank(context.Background(), "q", makeRerankChunks(2))
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if out[0].ChunkID != "b" {
		t.Errorf("top = %q, want b", out[0].ChunkID)
	}
}

func TestHTTPReranker_NormaliseEndpoint(t *testing.T) {
	t.Parallel()
	got := normaliseRerankEndpoint("http://localhost:8012")
	if got != "http://localhost:8012/v1/rerank" {
		t.Errorf("normalise base: %q", got)
	}
	got = normaliseRerankEndpoint("http://localhost:8012/")
	if got != "http://localhost:8012/v1/rerank" {
		t.Errorf("normalise trailing slash: %q", got)
	}
	got = normaliseRerankEndpoint("http://localhost:8012/custom/path")
	if got != "http://localhost:8012/custom/path" {
		t.Errorf("normalise custom path changed: %q", got)
	}
}

func TestHTTPReranker_ModelDefaulted(t *testing.T) {
	t.Parallel()
	var captured string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		req := decodeRerankRequest(t, r)
		captured = req.Model
		_ = json.NewEncoder(w).Encode(httpRerankResponse{
			Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
		})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: server.URL + "/v1/rerank"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if captured != "bge-reranker-v2-m3" {
		t.Errorf("default model = %q, want bge-reranker-v2-m3", captured)
	}
}

func TestHTTPReranker_RequiresEndpoint(t *testing.T) {
	t.Parallel()
	if _, err := NewHTTPReranker(HTTPRerankerConfig{}); err == nil {
		t.Fatal("expected error for empty endpoint")
	}
}

func TestHTTPReranker_MaxDocCharsTrimsPayload(t *testing.T) {
	t.Parallel()
	var received []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		req := decodeRerankRequest(t, r)
		received = append(received, req.Documents...)
		_ = json.NewEncoder(w).Encode(httpRerankResponse{
			Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
		})
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint:    server.URL + "/v1/rerank",
		MaxDocChars: 20,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	chunks := []RetrievedChunk{
		{ChunkID: "a", Path: "a.md", Title: strings.Repeat("a", 50)},
	}
	if _, err := rr.Rerank(context.Background(), "q", chunks); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(received) == 0 || len(received[0]) != 20 {
		t.Errorf("document length = %d, want 20", len(received[0]))
	}
}

func TestHTTPReranker_NameReflectsModel(t *testing.T) {
	t.Parallel()
	rr, err := NewHTTPReranker(HTTPRerankerConfig{Endpoint: "http://x/v1/rerank", Model: "m"})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	if rr.Name() != "http:m" {
		t.Errorf("Name = %q, want http:m", rr.Name())
	}
}

func TestHTTPReranker_TimeoutIsRespected(t *testing.T) {
	t.Parallel()
	stop := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveHealthyProbe(w, r) {
			return
		}
		// Block until the client gives up or the test signals stop.
		select {
		case <-stop:
		case <-r.Context().Done():
		}
	}))
	defer func() {
		close(stop)
		server.CloseClientConnections()
		server.Close()
	}()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		Timeout:  50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	start := time.Now()
	_, err = rr.Rerank(context.Background(), "q", makeRerankChunks(1))
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if time.Since(start) > 2*time.Second {
		t.Errorf("timeout not respected, took %v", time.Since(start))
	}
}

func TestHTTPReranker_CachesHealthyProbe(t *testing.T) {
	t.Parallel()

	var healthCount atomic.Int32
	var infoCount atomic.Int32
	var postCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			healthCount.Add(1)
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && r.URL.Path == "/info":
			infoCount.Add(1)
			t.Fatal("probe should not fall back to /info when /health is healthy")
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rerank":
			postCount.Add(1)
			_ = json.NewEncoder(w).Encode(httpRerankResponse{
				Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
			})
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		ProbeTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}

	for i := 0; i < 2; i++ {
		if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err != nil {
			t.Fatalf("Rerank %d: %v", i, err)
		}
	}

	if healthCount.Load() != 1 {
		t.Fatalf("GET /health probe count = %d, want 1", healthCount.Load())
	}
	if infoCount.Load() != 0 {
		t.Fatalf("GET /info probe count = %d, want 0", infoCount.Load())
	}
	if postCount.Load() != 2 {
		t.Fatalf("POST count = %d, want 2", postCount.Load())
	}
}

func TestHTTPReranker_ProbeFallsBackToInfo(t *testing.T) {
	t.Parallel()

	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.Path)
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			http.NotFound(w, r)
		case r.Method == http.MethodGet && r.URL.Path == "/info":
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rerank":
			_ = json.NewEncoder(w).Encode(httpRerankResponse{
				Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
			})
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		ProbeTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}

	if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	want := []string{"GET /health", "GET /info", "POST /v1/rerank"}
	if len(calls) != len(want) {
		t.Fatalf("call count = %d, want %d (%v)", len(calls), len(want), calls)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Fatalf("call %d = %q, want %q (all: %v)", i, calls[i], want[i], calls)
		}
	}
}

func TestHTTPReranker_CachesFailedProbe(t *testing.T) {
	t.Parallel()

	var healthCount atomic.Int32
	var infoCount atomic.Int32
	var postCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			healthCount.Add(1)
			http.NotFound(w, r)
		case r.Method == http.MethodGet && r.URL.Path == "/info":
			infoCount.Add(1)
			http.Error(w, "unhealthy", http.StatusServiceUnavailable)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rerank":
			postCount.Add(1)
			t.Fatal("POST should not run when the cached health probe is failing")
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		ProbeTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}

	for i := 0; i < 2; i++ {
		if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err == nil {
			t.Fatalf("Rerank %d: expected probe error", i)
		} else if !strings.Contains(err.Error(), "503") {
			t.Fatalf("Rerank %d error = %v, want mention of 503", i, err)
		}
	}

	if healthCount.Load() != 1 {
		t.Fatalf("GET /health probe count = %d, want 1", healthCount.Load())
	}
	if infoCount.Load() != 1 {
		t.Fatalf("GET /info probe count = %d, want 1", infoCount.Load())
	}
	if postCount.Load() != 0 {
		t.Fatalf("POST count = %d, want 0", postCount.Load())
	}
}

func TestHTTPReranker_ConcurrencyCap(t *testing.T) {
	var active atomic.Int32
	var maxActive atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && (r.URL.Path == "/health" || r.URL.Path == "/info"):
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && r.URL.Path == "/v1/rerank":
			cur := active.Add(1)
			for {
				prev := maxActive.Load()
				if cur <= prev || maxActive.CompareAndSwap(prev, cur) {
					break
				}
			}
			time.Sleep(40 * time.Millisecond)
			active.Add(-1)
			_ = json.NewEncoder(w).Encode(httpRerankResponse{
				Results: []httpRerankHit{{Index: 0, RelevanceScore: 1}},
			})
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	rr, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		ProbeTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < rerankMaxConcurrent*2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := rr.Rerank(context.Background(), "q", makeRerankChunks(1)); err != nil {
				t.Errorf("Rerank: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := maxActive.Load(); got == 0 || got > rerankMaxConcurrent {
		t.Fatalf("max concurrent POSTs = %d, want 1..%d", got, rerankMaxConcurrent)
	}
}

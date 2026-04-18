// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestHTTPReranker_ReordersByRelevance(t *testing.T) {
	t.Parallel()
	var lastPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

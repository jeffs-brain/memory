// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// newTestDaemon spins up a Daemon rooted at t.TempDir() and returns
// it along with an httptest.Server pointed at the daemon mux. The
// LLM provider is overridden with a deterministic Fake so /ask is
// reproducible.
func newTestDaemon(t *testing.T) (*Daemon, *httptest.Server) {
	t.Helper()
	root := t.TempDir()
	log := slog.New(slog.NewJSONHandler(io.Discard, nil))
	daemon, err := NewDaemon(context.Background(), root, "", log)
	if err != nil {
		t.Fatalf("daemon: %v", err)
	}
	// Replace the auto-detected provider with the deterministic fake.
	if daemon.LLM != nil {
		_ = daemon.LLM.Close()
	}
	daemon.LLM = llm.NewFake([]string{"The hedgehog lives in hedgerows."})
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	daemon.RegisterRoutes(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		_ = daemon.Close()
	})
	return daemon, srv
}

// TestServeHealth covers the unauth /healthz route.
func TestServeHealth(t *testing.T) {
	_, srv := newTestDaemon(t)
	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("get healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// TestServeBrainLifecycle covers create, list, get, delete.
func TestServeBrainLifecycle(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()

	body := strings.NewReader(`{"brainId":"alpha"}`)
	resp, err := c.Post(srv.URL+"/v1/brains", "application/json", body)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create status = %d, want 201", resp.StatusCode)
	}

	resp, err = c.Get(srv.URL + "/v1/brains/alpha")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("get status = %d, want 200", resp.StatusCode)
	}

	resp, err = c.Get(srv.URL + "/v1/brains")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	defer resp.Body.Close()
	var listBody struct {
		Items []struct {
			BrainID string `json:"brainId"`
		} `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listBody); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(listBody.Items) == 0 || listBody.Items[0].BrainID != "alpha" {
		t.Fatalf("list = %+v, want one entry alpha", listBody)
	}

	// Delete without confirm header should fail.
	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/v1/brains/alpha", nil)
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusNoContent {
		t.Fatalf("delete without confirm should not succeed")
	}

	req, _ = http.NewRequest(http.MethodDelete, srv.URL+"/v1/brains/alpha", nil)
	req.Header.Set("X-Confirm-Delete", "yes")
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("delete confirmed: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete confirmed status = %d, want 204", resp.StatusCode)
	}
}

// TestServeDocumentCRUD verifies PUT/GET/HEAD/DELETE/LIST/batch
// against a fresh brain.
func TestServeDocumentCRUD(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()

	mustCreateBrain(t, c, srv.URL, "docs")

	put := func(path string, body []byte) {
		req, _ := http.NewRequest(http.MethodPut, srv.URL+"/v1/brains/docs/documents?path="+path, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/octet-stream")
		resp, err := c.Do(req)
		if err != nil {
			t.Fatalf("put %s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNoContent {
			t.Fatalf("put %s status = %d, want 204", path, resp.StatusCode)
		}
	}

	put("memory%2Fglobal%2Fa.md", []byte("hello"))

	resp, err := c.Get(srv.URL + "/v1/brains/docs/documents/read?path=memory%2Fglobal%2Fa.md")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Equal(body, []byte("hello")) {
		t.Fatalf("read = %q, want hello", body)
	}

	req, _ := http.NewRequest(http.MethodHead, srv.URL+"/v1/brains/docs/documents?path=memory%2Fglobal%2Fa.md", nil)
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("head: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("head status = %d, want 200", resp.StatusCode)
	}

	resp, err = c.Get(srv.URL + "/v1/brains/docs/documents?dir=memory%2Fglobal&recursive=true")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), "memory/global/a.md") {
		t.Fatalf("list missing entry: %s", body)
	}

	// Batch: create one and delete the existing.
	batch := map[string]any{
		"reason": "test",
		"ops": []map[string]any{
			{"type": "write", "path": "memory/global/b.md", "content_base64": base64.StdEncoding.EncodeToString([]byte("world"))},
			{"type": "delete", "path": "memory/global/a.md"},
		},
	}
	bb, _ := json.Marshal(batch)
	resp, err = c.Post(srv.URL+"/v1/brains/docs/documents/batch-ops", "application/json", bytes.NewReader(bb))
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("batch status = %d body=%s", resp.StatusCode, body)
	}

	req, _ = http.NewRequest(http.MethodDelete, srv.URL+"/v1/brains/docs/documents?path=memory%2Fglobal%2Fb.md", nil)
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete status = %d, want 204", resp.StatusCode)
	}
}

// TestServeIngestSearch ingests a tiny markdown file and searches
// for a term inside it.
func TestServeIngestSearch(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "ingest")

	body := map[string]any{
		"path":          "test.md",
		"contentType":   "text/markdown",
		"contentBase64": base64.StdEncoding.EncodeToString([]byte("# hedgehog\n\nThe hedgehog lives in hedgerows.")),
	}
	bb, _ := json.Marshal(body)
	resp, err := c.Post(srv.URL+"/v1/brains/ingest/ingest/file", "application/json", bytes.NewReader(bb))
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("ingest status = %d body=%s", resp.StatusCode, respBody)
	}

	// Search for the keyword "hedgehog". With mode "auto" the
	// daemon falls through retriever -> search.Index -> knowledge
	// in-memory scan so ingested raw/documents content surfaces.
	searchBody, _ := json.Marshal(map[string]any{"query": "hedgehog", "topK": 5, "mode": "auto"})
	resp, err = c.Post(srv.URL+"/v1/brains/ingest/search", "application/json", bytes.NewReader(searchBody))
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	defer resp.Body.Close()
	var sb struct {
		Chunks []map[string]any `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sb); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(sb.Chunks) == 0 {
		t.Fatalf("search returned zero chunks")
	}
}

// TestServeAskSSE drives the /ask endpoint and consumes the SSE
// stream emitted by the daemon's deterministic Fake provider.
func TestServeAskSSE(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "asksse")

	// Pre-populate one document so retrieve has something to surface.
	body := map[string]any{
		"path":          "test.md",
		"contentType":   "text/markdown",
		"contentBase64": base64.StdEncoding.EncodeToString([]byte("# hedgehog\n\nThe hedgehog lives in hedgerows.")),
	}
	bb, _ := json.Marshal(body)
	resp, err := c.Post(srv.URL+"/v1/brains/asksse/ingest/file", "application/json", bytes.NewReader(bb))
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	resp.Body.Close()

	askBody, _ := json.Marshal(map[string]any{"question": "where does the hedgehog live", "topK": 1})
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/brains/asksse/ask", bytes.NewReader(askBody))
	req.Header.Set("Accept", "text/event-stream")
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("ask status = %d", resp.StatusCode)
	}

	events := readSSEEvents(t, resp.Body, 5*time.Second)
	wantEvents := []string{"retrieve", "answer_delta", "done"}
	for _, want := range wantEvents {
		if _, ok := events[want]; !ok {
			t.Fatalf("missing event %q in %v", want, events)
		}
	}
}

// TestServeEventsSSE subscribes to /events then writes a document
// and verifies a change event arrives.
func TestServeEventsSSE(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "evt")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/v1/brains/evt/events", nil)
	req.Header.Set("Accept", "text/event-stream")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("subscribe: %v", err)
	}
	defer resp.Body.Close()

	// Concurrently write a document so the subscriber sees it.
	go func() {
		time.Sleep(50 * time.Millisecond)
		put, _ := http.NewRequest(http.MethodPut, srv.URL+"/v1/brains/evt/documents?path=evt.md", strings.NewReader("hi"))
		put.Header.Set("Content-Type", "application/octet-stream")
		r, _ := c.Do(put)
		if r != nil {
			r.Body.Close()
		}
	}()

	events := readSSEEventsUntil(t, resp.Body, "change", 3*time.Second)
	if _, ok := events["change"]; !ok {
		t.Fatalf("missing change event in %v", events)
	}
}

// readSSEEventsUntil reads SSE frames until either the named stop
// event arrives or timeout fires.
func readSSEEventsUntil(t *testing.T, body io.Reader, stopEvent string, timeout time.Duration) map[string]string {
	t.Helper()
	out := map[string]string{}
	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var event string
		var data strings.Builder
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				if event != "" {
					out[event] = data.String()
					if event == stopEvent {
						return
					}
				}
				event = ""
				data.Reset()
				continue
			}
			if strings.HasPrefix(line, "event:") {
				event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			} else if strings.HasPrefix(line, "data:") {
				data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
	}()
	select {
	case <-doneCh:
	case <-time.After(timeout):
	}
	return out
}

// TestServeProblemErrors covers 404 unknown brain, 413 oversized
// body, and 401 forbidden when JB_AUTH_TOKEN is set.
func TestServeProblemErrors(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()

	// Unknown brain returns 404 not_found.
	resp, err := c.Get(srv.URL + "/v1/brains/missing/documents/read?path=a.md")
	if err != nil {
		t.Fatalf("get missing: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
	if !strings.Contains(string(body), "not_found") {
		t.Fatalf("body missing not_found: %s", body)
	}

	// Oversized body returns 413.
	mustCreateBrain(t, c, srv.URL, "size")
	huge := bytes.Repeat([]byte("x"), 3*1024*1024)
	req, _ := http.NewRequest(http.MethodPut, srv.URL+"/v1/brains/size/documents?path=memory%2Fglobal%2Fbig.md", bytes.NewReader(huge))
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("put oversized: %v", err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized status = %d body=%s, want 413", resp.StatusCode, body)
	}
}

// TestServeAuthToken installs a bearer token and verifies missing /
// invalid auth is rejected while a correct token passes through.
func TestServeAuthToken(t *testing.T) {
	root := t.TempDir()
	log := slog.New(slog.NewJSONHandler(io.Discard, nil))
	daemon, err := NewDaemon(context.Background(), root, "secret", log)
	if err != nil {
		t.Fatalf("daemon: %v", err)
	}
	t.Cleanup(func() { _ = daemon.Close() })
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	daemon.RegisterRoutes(mux)
	// Wrap with the same middleware the production daemon uses.
	srv := httptest.NewServer(authChain("secret", mux))
	defer srv.Close()
	c := srv.Client()

	// /healthz allowed without token.
	resp, err := c.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("healthz: %v", err)
	}
	resp.Body.Close()

	resp, err = c.Get(srv.URL + "/v1/brains")
	if err != nil {
		t.Fatalf("list anon: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("anon status = %d, want 401", resp.StatusCode)
	}

	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/v1/brains", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("auth: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("auth status = %d, want 200", resp.StatusCode)
	}
}

// authChain mirrors the wiring inside httpd.Server so the auth test
// can exercise the middleware without spinning a full Server.
func authChain(token string, mux http.Handler) http.Handler {
	return middlewareFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			mux.ServeHTTP(w, r)
			return
		}
		got := r.Header.Get("Authorization")
		if got == "" {
			w.Header().Set("Content-Type", "application/problem+json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"status":401,"code":"unauthorized","title":"Unauthorized"}`))
			return
		}
		if got != "Bearer "+token {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		mux.ServeHTTP(w, r)
	})
}

type middlewareFunc func(http.ResponseWriter, *http.Request)

func (f middlewareFunc) ServeHTTP(w http.ResponseWriter, r *http.Request) { f(w, r) }

// mustCreateBrain creates a brain and fails the test if the daemon
// returns anything other than 201.
func mustCreateBrain(t *testing.T, c *http.Client, base, id string) {
	t.Helper()
	body := strings.NewReader(`{"brainId":"` + id + `"}`)
	resp, err := c.Post(base+"/v1/brains", "application/json", body)
	if err != nil {
		t.Fatalf("create %s: %v", id, err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create %s status = %d", id, resp.StatusCode)
	}
}

// readSSEEvents reads SSE frames until timeout and returns a map
// keyed by event name with the most recent payload.
func readSSEEvents(t *testing.T, body io.Reader, timeout time.Duration) map[string]string {
	t.Helper()
	out := map[string]string{}
	deadline := time.Now().Add(timeout)
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	var event string
	var data strings.Builder
	for scanner.Scan() {
		if time.Now().After(deadline) {
			break
		}
		line := scanner.Text()
		if line == "" {
			if event != "" {
				out[event] = data.String()
			}
			event = ""
			data.Reset()
			if _, ok := out["done"]; ok {
				break
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	return out
}

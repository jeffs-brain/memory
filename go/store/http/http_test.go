// SPDX-License-Identifier: Apache-2.0

package http

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// memBackend is a tiny in-memory store that emulates the reference server's
// document surface. It exists so the Go HTTP tests stay hermetic; conformance
// tests reuse it too.
type memBackend struct {
	mu      sync.Mutex
	files   map[string]memEntry
	events  chan brain.ChangeEvent
	clients []chan<- brain.ChangeEvent
}

type memEntry struct {
	content []byte
	mtime   time.Time
}

func newMemBackend() *memBackend {
	return &memBackend{files: make(map[string]memEntry)}
}

func (m *memBackend) subscribe() (<-chan brain.ChangeEvent, func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ch := make(chan brain.ChangeEvent, 64)
	m.clients = append(m.clients, ch)
	unsubscribe := func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		for i, c := range m.clients {
			if c == ch {
				m.clients = append(m.clients[:i], m.clients[i+1:]...)
				break
			}
		}
		close(ch)
	}
	return ch, unsubscribe
}

func (m *memBackend) fan(evt brain.ChangeEvent) {
	m.mu.Lock()
	clients := append([]chan<- brain.ChangeEvent(nil), m.clients...)
	m.mu.Unlock()
	for _, c := range clients {
		select {
		case c <- evt:
		default:
			// drop if slow consumer
		}
	}
}

// validateServerPath mimics the server's validation; tests need errors for
// bad paths to surface as HTTP 400.
func validateServerPath(p string) error {
	if p == "" {
		return errors.New("empty")
	}
	if strings.Contains(p, "\x00") {
		return errors.New("nul")
	}
	if strings.Contains(p, "\\") {
		return errors.New("backslash")
	}
	if strings.HasPrefix(p, "/") {
		return errors.New("leading slash")
	}
	if strings.HasSuffix(p, "/") {
		return errors.New("trailing slash")
	}
	for _, seg := range strings.Split(p, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return errors.New("non-canonical")
		}
	}
	return nil
}

// writeProblem writes a Problem+JSON body. Keeps test server parity with
// the reference server.
func writeProblem(w http.ResponseWriter, status int, code, title, detail string) {
	w.Header().Set("Content-Type", "application/problem+json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": status,
		"code":   code,
		"title":  title,
		"detail": detail,
	})
}

// newTestServer wires a minimal reference server for a single brain.
func newTestServer(t *testing.T, brainID string, backend *memBackend) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	prefix := "/v1/brains/" + url.PathEscape(brainID)

	readHandler := func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Query().Get("path")
		if err := validateServerPath(p); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		backend.mu.Lock()
		entry, ok := backend.files[p]
		backend.mu.Unlock()
		if !ok {
			writeProblem(w, http.StatusNotFound, "not_found", "Not Found", p)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(entry.content)
	}
	statHandler := func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Query().Get("path")
		if err := validateServerPath(p); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		backend.mu.Lock()
		entry, ok := backend.files[p]
		backend.mu.Unlock()
		if !ok {
			writeProblem(w, http.StatusNotFound, "not_found", "Not Found", p)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"path":   p,
			"size":   len(entry.content),
			"mtime":  entry.mtime.UTC().Format(time.RFC3339Nano),
			"is_dir": false,
		})
	}
	listHandler := func(w http.ResponseWriter, r *http.Request) {
		dir := r.URL.Query().Get("dir")
		recursive, _ := strconv.ParseBool(r.URL.Query().Get("recursive"))
		includeGenerated, _ := strconv.ParseBool(r.URL.Query().Get("include_generated"))
		glob := r.URL.Query().Get("glob")
		backend.mu.Lock()
		paths := make([]string, 0, len(backend.files))
		for p := range backend.files {
			paths = append(paths, p)
		}
		backend.mu.Unlock()
		byBase := make(map[string]memEntry)
		for _, p := range paths {
			e := backend.files[p]
			byBase[p] = e
		}
		type item struct {
			Path  string `json:"path"`
			Size  int    `json:"size"`
			MTime string `json:"mtime"`
			IsDir bool   `json:"is_dir"`
		}
		files := make(map[string]item)
		dirs := make(map[string]struct{})
		for p, e := range byBase {
			if dir != "" {
				if p == dir || strings.HasPrefix(p, dir+"/") {
					// ok
				} else {
					continue
				}
			}
			rel := p
			if dir != "" {
				rel = strings.TrimPrefix(p, dir+"/")
			}
			if !recursive {
				if idx := strings.Index(rel, "/"); idx >= 0 {
					dname := rel[:idx]
					full := dname
					if dir != "" {
						full = dir + "/" + dname
					}
					dirs[full] = struct{}{}
					continue
				}
			}
			base := p
			if idx := strings.LastIndex(p, "/"); idx >= 0 {
				base = p[idx+1:]
			}
			if !includeGenerated && strings.HasPrefix(base, "_") {
				continue
			}
			if glob != "" {
				matched, err := testGlobMatch(glob, base)
				if err != nil || !matched {
					continue
				}
			}
			files[p] = item{
				Path:  p,
				Size:  len(e.content),
				MTime: e.mtime.UTC().Format(time.RFC3339Nano),
				IsDir: false,
			}
		}
		out := make([]item, 0, len(files)+len(dirs))
		for _, v := range files {
			out = append(out, v)
		}
		for d := range dirs {
			out = append(out, item{Path: d, IsDir: true})
		}
		// sort ascending by path
		for i := 0; i < len(out); i++ {
			for j := i + 1; j < len(out); j++ {
				if out[j].Path < out[i].Path {
					out[i], out[j] = out[j], out[i]
				}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"items": out})
	}
	docsHandler := func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			listHandler(w, r)
		case http.MethodHead:
			p := r.URL.Query().Get("path")
			if err := validateServerPath(p); err != nil {
				writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
				return
			}
			backend.mu.Lock()
			_, ok := backend.files[p]
			backend.mu.Unlock()
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
		case http.MethodPut:
			p := r.URL.Query().Get("path")
			if err := validateServerPath(p); err != nil {
				writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
				return
			}
			body, err := io.ReadAll(r.Body)
			if err != nil {
				writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
				return
			}
			backend.mu.Lock()
			_, existed := backend.files[p]
			backend.files[p] = memEntry{content: body, mtime: time.Now()}
			backend.mu.Unlock()
			kind := brain.ChangeCreated
			if existed {
				kind = brain.ChangeUpdated
			}
			backend.fan(brain.ChangeEvent{Kind: kind, Path: brain.Path(p), When: time.Now()})
			w.WriteHeader(http.StatusNoContent)
		case http.MethodDelete:
			p := r.URL.Query().Get("path")
			if err := validateServerPath(p); err != nil {
				writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
				return
			}
			backend.mu.Lock()
			_, ok := backend.files[p]
			if ok {
				delete(backend.files, p)
			}
			backend.mu.Unlock()
			if !ok {
				writeProblem(w, http.StatusNotFound, "not_found", "Not Found", p)
				return
			}
			backend.fan(brain.ChangeEvent{Kind: brain.ChangeDeleted, Path: brain.Path(p), When: time.Now()})
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}
	appendHandler := func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Query().Get("path")
		if err := validateServerPath(p); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		backend.mu.Lock()
		entry, existed := backend.files[p]
		entry.content = append(entry.content, body...)
		entry.mtime = time.Now()
		backend.files[p] = entry
		backend.mu.Unlock()
		kind := brain.ChangeCreated
		if existed {
			kind = brain.ChangeUpdated
		}
		backend.fan(brain.ChangeEvent{Kind: kind, Path: brain.Path(p), When: time.Now()})
		w.WriteHeader(http.StatusNoContent)
	}
	renameHandler := func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			From string `json:"from"`
			To   string `json:"to"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		if err := validateServerPath(body.From); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		if err := validateServerPath(body.To); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		backend.mu.Lock()
		entry, ok := backend.files[body.From]
		if !ok {
			backend.mu.Unlock()
			writeProblem(w, http.StatusNotFound, "not_found", "Not Found", body.From)
			return
		}
		delete(backend.files, body.From)
		backend.files[body.To] = entry
		backend.mu.Unlock()
		backend.fan(brain.ChangeEvent{Kind: brain.ChangeRenamed, Path: brain.Path(body.To), OldPath: brain.Path(body.From), When: time.Now()})
		w.WriteHeader(http.StatusNoContent)
	}
	batchHandler := func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Reason string           `json:"reason"`
			Ops    []map[string]any `json:"ops"`
		}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
			return
		}
		backend.mu.Lock()
		snapshot := make(map[string]memEntry, len(backend.files))
		for k, v := range backend.files {
			snapshot[k] = v
		}
		backend.mu.Unlock()
		staged := make(map[string]memEntry, len(snapshot))
		for k, v := range snapshot {
			staged[k] = v
		}
		for _, op := range body.Ops {
			kind, _ := op["type"].(string)
			p, _ := op["path"].(string)
			switch kind {
			case "write":
				if err := validateServerPath(p); err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				b64, _ := op["content_base64"].(string)
				dec, err := base64.StdEncoding.DecodeString(b64)
				if err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				staged[p] = memEntry{content: dec, mtime: time.Now()}
			case "append":
				if err := validateServerPath(p); err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				b64, _ := op["content_base64"].(string)
				dec, err := base64.StdEncoding.DecodeString(b64)
				if err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				entry := staged[p]
				entry.content = append(entry.content, dec...)
				entry.mtime = time.Now()
				staged[p] = entry
			case "delete":
				if err := validateServerPath(p); err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				if _, ok := staged[p]; !ok {
					writeProblem(w, http.StatusNotFound, "not_found", "Not Found", p)
					return
				}
				delete(staged, p)
			case "rename":
				to, _ := op["to"].(string)
				if err := validateServerPath(p); err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				if err := validateServerPath(to); err != nil {
					writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", err.Error())
					return
				}
				entry, ok := staged[p]
				if !ok {
					writeProblem(w, http.StatusNotFound, "not_found", "Not Found", p)
					return
				}
				delete(staged, p)
				staged[to] = entry
			default:
				writeProblem(w, http.StatusBadRequest, "validation_error", "Bad Request", "unknown op: "+kind)
				return
			}
		}
		backend.mu.Lock()
		backend.files = staged
		backend.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"committed": len(body.Ops)})
	}
	eventsHandler := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeProblem(w, http.StatusInternalServerError, "internal_error", "no streaming support", "")
			return
		}
		ch, unsubscribe := backend.subscribe()
		defer unsubscribe()
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "event: ready\ndata: ok\n\n")
		flusher.Flush()
		for {
			select {
			case <-r.Context().Done():
				return
			case evt, ok := <-ch:
				if !ok {
					return
				}
				payload := map[string]any{
					"kind": evt.Kind.String(),
					"path": string(evt.Path),
					"when": evt.When.UTC().Format(time.RFC3339Nano),
				}
				if evt.OldPath != "" {
					payload["old_path"] = string(evt.OldPath)
				}
				if evt.Reason != "" {
					payload["reason"] = evt.Reason
				}
				data, _ := json.Marshal(payload)
				_, _ = fmt.Fprintf(w, "event: change\ndata: %s\n\n", data)
				flusher.Flush()
			}
		}
	}
	mux.HandleFunc(prefix+"/documents/read", readHandler)
	mux.HandleFunc(prefix+"/documents/stat", statHandler)
	mux.HandleFunc(prefix+"/documents/append", appendHandler)
	mux.HandleFunc(prefix+"/documents/rename", renameHandler)
	mux.HandleFunc(prefix+"/documents/batch-ops", batchHandler)
	mux.HandleFunc(prefix+"/documents", docsHandler)
	mux.HandleFunc(prefix+"/events", eventsHandler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

// testGlobMatch mirrors Go's path.Match on a single base name.
func testGlobMatch(pattern, name string) (bool, error) {
	return stdlibPathMatch(pattern, name)
}

// ---------- direct interface tests ----------

func TestReadWriteRoundTrip(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	if err := store.Write(ctx, brain.Path("memory/a.md"), []byte("hello")); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := store.Read(ctx, brain.Path("memory/a.md"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, []byte("hello")) {
		t.Fatalf("want hello got %q", got)
	}
}

func TestReadMissingReturnsNotFound(t *testing.T) {
	store, _ := newStore(t)
	_, err := store.Read(context.Background(), brain.Path("memory/nope.md"))
	if !errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

func TestWriteAppendThenRead(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	if err := store.Append(ctx, brain.Path("wiki/log.md"), []byte("a\n")); err != nil {
		t.Fatalf("append 1: %v", err)
	}
	if err := store.Append(ctx, brain.Path("wiki/log.md"), []byte("b\n")); err != nil {
		t.Fatalf("append 2: %v", err)
	}
	got, err := store.Read(ctx, brain.Path("wiki/log.md"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !bytes.Equal(got, []byte("a\nb\n")) {
		t.Fatalf("want a\\nb\\n got %q", got)
	}
}

func TestDeleteAndExists(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	if err := store.Write(ctx, brain.Path("memory/tmp.md"), []byte("x")); err != nil {
		t.Fatalf("write: %v", err)
	}
	ok, err := store.Exists(ctx, brain.Path("memory/tmp.md"))
	if err != nil || !ok {
		t.Fatalf("exists pre-delete: ok=%v err=%v", ok, err)
	}
	if err := store.Delete(ctx, brain.Path("memory/tmp.md")); err != nil {
		t.Fatalf("delete: %v", err)
	}
	ok, err = store.Exists(ctx, brain.Path("memory/tmp.md"))
	if err != nil || ok {
		t.Fatalf("exists post-delete: ok=%v err=%v", ok, err)
	}
	if err := store.Delete(ctx, brain.Path("memory/tmp.md")); !errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("want ErrNotFound on second delete, got %v", err)
	}
}

func TestRenameOverwrites(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	_ = store.Write(ctx, brain.Path("raw/a.md"), []byte("src"))
	_ = store.Write(ctx, brain.Path("raw/b.md"), []byte("dst"))
	if err := store.Rename(ctx, brain.Path("raw/a.md"), brain.Path("raw/b.md")); err != nil {
		t.Fatalf("rename: %v", err)
	}
	got, err := store.Read(ctx, brain.Path("raw/b.md"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(got) != "src" {
		t.Fatalf("want src got %q", got)
	}
	if _, err := store.Read(ctx, brain.Path("raw/a.md")); !errors.Is(err, brain.ErrNotFound) {
		t.Fatalf("want ErrNotFound for renamed source, got %v", err)
	}
}

func TestStatReturnsSize(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	_ = store.Write(ctx, brain.Path("memory/stat.md"), []byte("twelve bytes"))
	fi, err := store.Stat(ctx, brain.Path("memory/stat.md"))
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if fi.Size != 12 {
		t.Fatalf("want size 12, got %d", fi.Size)
	}
	if fi.Path != brain.Path("memory/stat.md") {
		t.Fatalf("want path memory/stat.md, got %q", fi.Path)
	}
}

func TestListSorted(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	_ = store.Write(ctx, brain.Path("memory/c.md"), []byte("c"))
	_ = store.Write(ctx, brain.Path("memory/a.md"), []byte("a"))
	_ = store.Write(ctx, brain.Path("memory/b.md"), []byte("b"))
	items, err := store.List(ctx, brain.Path("memory"), brain.ListOpts{})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	var paths []string
	for _, it := range items {
		paths = append(paths, string(it.Path))
	}
	want := []string{"memory/a.md", "memory/b.md", "memory/c.md"}
	if !stringSliceEqual(paths, want) {
		t.Fatalf("want %v got %v", want, paths)
	}
}

func TestBatchCommit(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, brain.Path("memory/x.md"), []byte("one")); err != nil {
			return err
		}
		if err := b.Write(ctx, brain.Path("memory/y.md"), []byte("two")); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	x, _ := store.Read(ctx, brain.Path("memory/x.md"))
	y, _ := store.Read(ctx, brain.Path("memory/y.md"))
	if string(x) != "one" || string(y) != "two" {
		t.Fatalf("content mismatch: x=%q y=%q", x, y)
	}
}

func TestBatchWriteDeleteCancels(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, brain.Path("memory/ephem.md"), []byte("A")); err != nil {
			return err
		}
		if err := b.Delete(ctx, brain.Path("memory/ephem.md")); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	ok, _ := store.Exists(ctx, brain.Path("memory/ephem.md"))
	if ok {
		t.Fatalf("ephem.md should not exist after batch write+delete")
	}
}

func TestBatchWriteWriteKeepsLatter(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	err := store.Batch(ctx, brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		if err := b.Write(ctx, brain.Path("memory/ow.md"), []byte("first")); err != nil {
			return err
		}
		if err := b.Write(ctx, brain.Path("memory/ow.md"), []byte("second")); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	got, _ := store.Read(ctx, brain.Path("memory/ow.md"))
	if string(got) != "second" {
		t.Fatalf("want second got %q", got)
	}
}

func TestSubscribeReceivesChange(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	received := make(chan brain.ChangeEvent, 4)
	unsubscribe := store.Subscribe(brain.EventSinkFunc(func(evt brain.ChangeEvent) {
		received <- evt
	}))
	t.Cleanup(unsubscribe)
	// give the SSE goroutine time to attach
	time.Sleep(50 * time.Millisecond)
	if err := store.Write(ctx, brain.Path("memory/e1.md"), []byte("x")); err != nil {
		t.Fatalf("write: %v", err)
	}
	select {
	case evt := <-received:
		if evt.Path != brain.Path("memory/e1.md") {
			t.Fatalf("unexpected event path: %+v", evt)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timeout waiting for change event")
	}
}

func TestAuthHeaderForwarded(t *testing.T) {
	backend := newMemBackend()
	var seen atomic.Value
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/brains/b1/documents", func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusNoContent)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	store := New(Config{BaseURL: srv.URL, BrainID: "b1", Token: "jbk_test_secret"})
	t.Cleanup(func() { _ = store.Close() })
	if err := store.Write(context.Background(), brain.Path("memory/auth.md"), []byte("x")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if v := seen.Load(); v == nil || v.(string) != "Bearer jbk_test_secret" {
		t.Fatalf("auth header missing or wrong: %v", v)
	}
	_ = backend
}

func TestRetryHonoursRetryAfter(t *testing.T) {
	var attempts atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/brains/b1/documents", func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	store := New(Config{BaseURL: srv.URL, BrainID: "b1"})
	t.Cleanup(func() { _ = store.Close() })
	if err := store.Write(context.Background(), brain.Path("memory/a.md"), []byte("x")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("want 2 attempts got %d", got)
	}
}

func TestRetryExhaustedReturnsRateLimited(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/brains/b1/documents", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"code":"rate_limited","status":429}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	store := New(Config{BaseURL: srv.URL, BrainID: "b1", MaxRetries: 1})
	t.Cleanup(func() { _ = store.Close() })
	err := store.Write(context.Background(), brain.Path("memory/a.md"), []byte("x"))
	if !errors.Is(err, brain.ErrRateLimited) {
		t.Fatalf("want ErrRateLimited got %v", err)
	}
}

func TestInvalidPathRejectedClientSide(t *testing.T) {
	store, _ := newStore(t)
	ctx := context.Background()
	cases := []brain.Path{
		"/absolute.md",
		"memory/dir/",
		"memory/a\x00b.md",
		"../etc/passwd",
	}
	for _, p := range cases {
		if err := store.Write(ctx, p, []byte("x")); !errors.Is(err, brain.ErrInvalidPath) {
			t.Fatalf("path %q: want ErrInvalidPath got %v", p, err)
		}
	}
}

func TestPayloadTooLargeEnforcedClientSide(t *testing.T) {
	store, _ := newStore(t)
	big := make([]byte, singleBodyLimit+1)
	err := store.Write(context.Background(), brain.Path("memory/big.md"), big)
	if !errors.Is(err, brain.ErrPayloadTooLarge) {
		t.Fatalf("want ErrPayloadTooLarge got %v", err)
	}
}

func TestCloseMakesStoreReadOnly(t *testing.T) {
	store, _ := newStore(t)
	_ = store.Close()
	err := store.Write(context.Background(), brain.Path("memory/a.md"), []byte("x"))
	if !errors.Is(err, brain.ErrReadOnly) {
		t.Fatalf("want ErrReadOnly got %v", err)
	}
}

// ---------- helpers ----------

func newStore(t *testing.T) (*Store, *memBackend) {
	t.Helper()
	backend := newMemBackend()
	srv := newTestServer(t, "b1", backend)
	store := New(Config{BaseURL: srv.URL, BrainID: "b1"})
	t.Cleanup(func() { _ = store.Close() })
	return store, backend
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

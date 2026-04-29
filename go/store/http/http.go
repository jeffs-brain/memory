// SPDX-License-Identifier: Apache-2.0

// Package http is the HTTP-backed implementation of [brain.Store]. It
// speaks the wire protocol documented at spec/PROTOCOL.md and matches the
// TypeScript reference at packages/memory/src/store/httpstore.ts.
//
// The store is safe for concurrent use. It holds no long-lived resources
// other than the shared http.Client and, when Subscribe is called, a
// single background goroutine that multiplexes SSE events to all sinks.
package http

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Config configures an HTTP-backed Store.
type Config struct {
	// BaseURL is the origin that serves /v1/brains/{brainId}/... routes.
	// Trailing slashes are trimmed.
	BaseURL string

	// BrainID scopes every request to a single brain.
	BrainID string

	// Token is sent as "Authorization: Bearer <token>". Optional.
	Token string

	// Client is used for all requests. When nil, a default client with a
	// 30 second timeout is constructed.
	Client *http.Client

	// UserAgent overrides the default User-Agent header.
	UserAgent string

	// MaxRetries caps retry attempts on 429/502/503/504. Defaults to 3.
	// Set to a negative value to disable retries entirely.
	MaxRetries int
}

// Per-endpoint body size caps from spec/PROTOCOL.md. Enforced client-side
// so callers get a typed error before the bytes hit the wire.
const (
	singleBodyLimit = 2 * 1024 * 1024
	batchBodyLimit  = 8 * 1024 * 1024
	batchOpLimit    = 1024
	defaultTimeout  = 30 * time.Second
	defaultUA       = "jeffs-brain-memory-go/0.1 (+HttpStore)"
)

// Store is the HTTP-backed Store. See package docs for wire protocol.
type Store struct {
	cfg        Config
	client     *http.Client
	authHeader string
	userAgent  string
	maxRetries int

	mu         sync.Mutex
	closed     bool
	nextSinkID uint64
	sinks      map[uint64]brain.EventSink
	subCancel  context.CancelFunc
	subDone    chan struct{}
}

// New returns an HTTP-backed Store. Panics if BrainID is empty because
// every request needs it and catching the error at every call site would
// be noisy.
func New(cfg Config) *Store {
	if cfg.BrainID == "" {
		panic("store/http: Config.BrainID is required")
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: defaultTimeout}
	}
	ua := cfg.UserAgent
	if ua == "" {
		ua = defaultUA
	}
	retries := cfg.MaxRetries
	if retries == 0 {
		retries = 3
	}
	if retries < 0 {
		retries = 0
	}
	auth := ""
	if cfg.Token != "" {
		auth = "Bearer " + cfg.Token
	}
	return &Store{
		cfg:        cfg,
		client:     client,
		authHeader: auth,
		userAgent:  ua,
		maxRetries: retries,
		sinks:      make(map[uint64]brain.EventSink),
	}
}

// compile-time guard.
var _ brain.Store = (*Store)(nil)

// ---------- URL helpers ----------

// brainPath builds a path under /v1/brains/{brainId}. BrainID is URL
// encoded per RFC 3986.
func (s *Store) brainPath(suffix string) string {
	return "/v1/brains/" + url.PathEscape(s.cfg.BrainID) + suffix
}

// buildURL joins the base URL, path and optional query.
func (s *Store) buildURL(path string, query url.Values) string {
	base := strings.TrimRight(s.cfg.BaseURL, "/")
	target := base + path
	if len(query) == 0 {
		return target
	}
	sep := "?"
	if strings.Contains(target, "?") {
		sep = "&"
	}
	return target + sep + query.Encode()
}

// ---------- low-level request pipeline ----------

type requestInit struct {
	method      string
	path        string
	query       url.Values
	body        []byte
	accept      string
	contentType string
}

// do issues the request with retry policy applied to 429/502/503/504.
// Returns the final Response which the caller must drain/close. A non-nil
// error replaces the response.
func (s *Store) do(ctx context.Context, init requestInit) (*http.Response, error) {
	if s.isClosed() {
		return nil, brain.ErrReadOnly
	}
	attempt := 0
	for {
		resp, retry, wait, err := s.doOnce(ctx, init)
		if err != nil {
			return nil, err
		}
		if !retry || attempt >= s.maxRetries {
			return resp, nil
		}
		// Drain + close so the HTTP client can reuse the connection.
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		attempt++
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// doOnce performs a single attempt. Returns (resp, retry, wait, err). When
// err is non-nil the other values are ignored. The caller owns resp.Body
// and must close it.
func (s *Store) doOnce(ctx context.Context, init requestInit) (*http.Response, bool, time.Duration, error) {
	target := s.buildURL(init.path, init.query)
	var bodyReader io.Reader
	if init.body != nil {
		bodyReader = bytes.NewReader(init.body)
	}
	req, err := http.NewRequestWithContext(ctx, init.method, target, bodyReader)
	if err != nil {
		return nil, false, 0, err
	}
	accept := init.accept
	if accept == "" {
		accept = "application/json"
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("User-Agent", s.userAgent)
	if init.contentType != "" {
		req.Header.Set("Content-Type", init.contentType)
	}
	if s.authHeader != "" {
		req.Header.Set("Authorization", s.authHeader)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, false, 0, err
	}
	switch resp.StatusCode {
	case http.StatusTooManyRequests, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return resp, true, retryBackoff(resp, 0), nil
	}
	return resp, false, 0, nil
}

// retryBackoff returns the wait duration for the next attempt. The
// Retry-After header (seconds or HTTP-date) wins over the exponential
// fallback; jitter is added to both paths so many clients desync.
func retryBackoff(resp *http.Response, attempt int) time.Duration {
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil && secs >= 0 {
			return time.Duration(secs)*time.Second + jitter()
		}
		if t, err := http.ParseTime(ra); err == nil {
			d := time.Until(t)
			if d < 0 {
				d = 0
			}
			return d + jitter()
		}
	}
	base := time.Duration(100*(1<<attempt)) * time.Millisecond
	if base > 4*time.Second {
		base = 4 * time.Second
	}
	return base + jitter()
}

// jitter returns a small random slice to keep client retries desynced.
func jitter() time.Duration {
	return time.Duration(rand.Int64N(100)) * time.Millisecond
}

// ---------- interface implementation ----------

// Read returns the raw bytes of a document.
func (s *Store) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := brain.ValidatePath(p); err != nil {
		return nil, err
	}
	resp, err := s.do(ctx, requestInit{
		method: http.MethodGet,
		path:   s.brainPath("/documents/read"),
		query:  url.Values{"path": []string{string(p)}},
		accept: "application/octet-stream",
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, responseError(resp)
	}
	return io.ReadAll(resp.Body)
}

// Write performs a full rewrite of p.
func (s *Store) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	if len(content) > singleBodyLimit {
		return fmt.Errorf("store/http: write %s: %w", p, brain.ErrPayloadTooLarge)
	}
	resp, err := s.do(ctx, requestInit{
		method:      http.MethodPut,
		path:        s.brainPath("/documents"),
		query:       url.Values{"path": []string{string(p)}},
		body:        content,
		contentType: "application/octet-stream",
	})
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return responseError(resp)
	}
	return nil
}

// Append appends bytes to p.
func (s *Store) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	if len(content) > singleBodyLimit {
		return fmt.Errorf("store/http: append %s: %w", p, brain.ErrPayloadTooLarge)
	}
	resp, err := s.do(ctx, requestInit{
		method:      http.MethodPost,
		path:        s.brainPath("/documents/append"),
		query:       url.Values{"path": []string{string(p)}},
		body:        content,
		contentType: "application/octet-stream",
	})
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return responseError(resp)
	}
	return nil
}

// Delete removes a file.
func (s *Store) Delete(ctx context.Context, p brain.Path) error {
	if err := brain.ValidatePath(p); err != nil {
		return err
	}
	resp, err := s.do(ctx, requestInit{
		method: http.MethodDelete,
		path:   s.brainPath("/documents"),
		query:  url.Values{"path": []string{string(p)}},
	})
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return responseError(resp)
	}
	return nil
}

// Rename moves src to dst. Overwrites dst.
func (s *Store) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := brain.ValidatePath(src); err != nil {
		return err
	}
	if err := brain.ValidatePath(dst); err != nil {
		return err
	}
	body, err := json.Marshal(map[string]string{
		"from": string(src),
		"to":   string(dst),
	})
	if err != nil {
		return err
	}
	resp, err := s.do(ctx, requestInit{
		method:      http.MethodPost,
		path:        s.brainPath("/documents/rename"),
		body:        body,
		contentType: "application/json",
	})
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return responseError(resp)
	}
	return nil
}

// Exists reports whether p exists via HEAD.
func (s *Store) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := brain.ValidatePath(p); err != nil {
		return false, err
	}
	resp, err := s.do(ctx, requestInit{
		method: http.MethodHead,
		path:   s.brainPath("/documents"),
		query:  url.Values{"path": []string{string(p)}},
	})
	if err != nil {
		return false, err
	}
	defer func() { _ = resp.Body.Close() }()
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	}
	return false, responseError(resp)
}

// Stat returns metadata for a single path.
func (s *Store) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := brain.ValidatePath(p); err != nil {
		return brain.FileInfo{}, err
	}
	resp, err := s.do(ctx, requestInit{
		method: http.MethodGet,
		path:   s.brainPath("/documents/stat"),
		query:  url.Values{"path": []string{string(p)}},
	})
	if err != nil {
		return brain.FileInfo{}, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return brain.FileInfo{}, responseError(resp)
	}
	var raw rawFileInfo
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return brain.FileInfo{}, err
	}
	return raw.toFileInfo(), nil
}

// List enumerates entries under dir.
func (s *Store) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	query := url.Values{
		"dir":               []string{string(dir)},
		"recursive":         []string{strconv.FormatBool(opts.Recursive)},
		"include_generated": []string{strconv.FormatBool(opts.IncludeGenerated)},
	}
	if opts.Glob != "" {
		query.Set("glob", opts.Glob)
	}
	resp, err := s.do(ctx, requestInit{
		method: http.MethodGet,
		path:   s.brainPath("/documents"),
		query:  query,
	})
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, responseError(resp)
	}
	var body struct {
		Items []rawFileInfo `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	out := make([]brain.FileInfo, 0, len(body.Items))
	for _, it := range body.Items {
		out = append(out, it.toFileInfo())
	}
	return out, nil
}

// Batch buffers mutations and flushes them as a single batch-ops POST.
// fn must not retain the Batch beyond its own return.
func (s *Store) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	if fn == nil {
		return errors.New("store/http: batch fn is required")
	}
	b := &batch{store: s, ctx: ctx}
	if err := fn(b); err != nil {
		return err
	}
	if len(b.journal) == 0 {
		return nil
	}
	if len(b.journal) > batchOpLimit {
		return fmt.Errorf("store/http: batch has %d ops (cap %d): %w", len(b.journal), batchOpLimit, brain.ErrPayloadTooLarge)
	}
	payload := map[string]any{
		"reason": opts.Reason,
	}
	if opts.Message != "" {
		payload["message"] = opts.Message
	}
	if opts.Author != "" {
		payload["author"] = opts.Author
	}
	if opts.Email != "" {
		payload["email"] = opts.Email
	}
	ops := make([]map[string]any, 0, len(b.journal))
	total := 0
	for _, op := range b.journal {
		total += len(op.Content)
		if total > batchBodyLimit {
			return fmt.Errorf("store/http: batch payload exceeds %d bytes: %w", batchBodyLimit, brain.ErrPayloadTooLarge)
		}
		switch op.Kind {
		case jOpWrite:
			ops = append(ops, map[string]any{
				"type":           "write",
				"path":           string(op.Path),
				"content_base64": base64.StdEncoding.EncodeToString(op.Content),
			})
		case jOpAppend:
			ops = append(ops, map[string]any{
				"type":           "append",
				"path":           string(op.Path),
				"content_base64": base64.StdEncoding.EncodeToString(op.Content),
			})
		case jOpDelete:
			ops = append(ops, map[string]any{
				"type": "delete",
				"path": string(op.Path),
			})
		case jOpRename:
			ops = append(ops, map[string]any{
				"type": "rename",
				"path": string(op.Src),
				"to":   string(op.Path),
			})
		}
	}
	payload["ops"] = ops
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	resp, err := s.do(ctx, requestInit{
		method:      http.MethodPost,
		path:        s.brainPath("/documents/batch-ops"),
		body:        body,
		contentType: "application/json",
	})
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return responseError(resp)
	}
	return nil
}

// Subscribe registers a sink. The first subscribe kicks off the single
// background SSE reader; subsequent subscribes just fan in. The returned
// unsubscribe function is safe to call multiple times.
func (s *Store) Subscribe(sink brain.EventSink) func() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return func() {}
	}
	id := s.nextSinkID
	s.nextSinkID++
	s.sinks[id] = sink
	if s.subCancel == nil {
		ctx, cancel := context.WithCancel(context.Background())
		s.subCancel = cancel
		s.subDone = make(chan struct{})
		go s.runSubscription(ctx, s.subDone)
	}
	s.mu.Unlock()
	return func() {
		s.mu.Lock()
		delete(s.sinks, id)
		if len(s.sinks) == 0 && s.subCancel != nil {
			cancel := s.subCancel
			done := s.subDone
			s.subCancel = nil
			s.subDone = nil
			s.mu.Unlock()
			cancel()
			if done != nil {
				<-done
			}
			return
		}
		s.mu.Unlock()
	}
}

// runSubscription reads the SSE stream and fans events out to all sinks.
// Exits when the context cancels or the stream closes. The done channel is
// owned by the subscription goroutine and closed on exit; callers wait on
// their own copy passed at construction.
func (s *Store) runSubscription(ctx context.Context, done chan struct{}) {
	defer close(done)
	target := s.buildURL(s.brainPath("/events"), nil)
	runSubscription(ctx, s.client, target, s.authHeader, s.userAgent, func(evt brain.ChangeEvent) {
		s.mu.Lock()
		sinks := make([]brain.EventSink, 0, len(s.sinks))
		for _, sink := range s.sinks {
			sinks = append(sinks, sink)
		}
		s.mu.Unlock()
		for _, sink := range sinks {
			sink.OnBrainChange(evt)
		}
	}, func(err error) {
		// Swallow transient errors the store stays usable for non-SSE ops.
		_ = err
	})
}

// LocalPath returns "", false: HTTP stores never expose an on-disk path.
func (s *Store) LocalPath(p brain.Path) (string, bool) {
	return "", false
}

// Close tears down the background SSE goroutine and marks the store as
// read-only. Safe to call multiple times.
func (s *Store) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	cancel := s.subCancel
	done := s.subDone
	s.subCancel = nil
	s.subDone = nil
	s.sinks = map[uint64]brain.EventSink{}
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
	return nil
}

func (s *Store) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

// ---------- wire types ----------

type rawFileInfo struct {
	Path  string `json:"path"`
	Size  int64  `json:"size"`
	MTime string `json:"mtime"`
	IsDir bool   `json:"is_dir"`
}

func (r rawFileInfo) toFileInfo() brain.FileInfo {
	mt, _ := time.Parse(time.RFC3339Nano, r.MTime)
	return brain.FileInfo{
		Path:    brain.Path(r.Path),
		Size:    r.Size,
		ModTime: mt,
		IsDir:   r.IsDir,
	}
}

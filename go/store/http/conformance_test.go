// SPDX-License-Identifier: Apache-2.0

package http

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// conformanceFile returns the language-neutral conformance fixture path. The
// cases live in ../../../spec/conformance/http-contract.json relative to
// this test file. Resolved via the repository root two levels above
// go/store/http.
func conformanceFile(t *testing.T) string {
	t.Helper()
	// Walk up from this package directory until we find spec/conformance.
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	cur := wd
	for i := 0; i < 8; i++ {
		candidate := filepath.Join(cur, "spec", "conformance", "http-contract.json")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	t.Fatalf("could not find spec/conformance/http-contract.json from %s", wd)
	return ""
}

// conformanceCase is the JSON shape emitted by spec/conformance. Only the
// fields the Go harness needs are bound; unknown fields decode to their
// zero values.
type conformanceCase struct {
	Name             string                 `json:"name"`
	Setup            []conformanceStep      `json:"setup,omitempty"`
	Request          map[string]any         `json:"request"`
	ExpectedResponse map[string]any         `json:"expectedResponse"`
	FollowUp         []conformanceStep      `json:"followUp,omitempty"`
	Teardown         []conformanceStep      `json:"teardown,omitempty"`
	Notes            string                 `json:"notes,omitempty"`
	Placeholders     map[string]string      `json:"placeholders,omitempty"`
	Extra            map[string]interface{} `json:"-"`
}

// conformanceStep covers HTTP steps and SSE control steps.
type conformanceStep struct {
	Kind               string         `json:"kind,omitempty"`
	Name               string         `json:"name,omitempty"`
	Method             string         `json:"method,omitempty"`
	Path               string         `json:"path,omitempty"`
	Headers            map[string]any `json:"headers,omitempty"`
	BodyBase64         string         `json:"bodyBase64,omitempty"`
	BodyJSON           any            `json:"bodyJson,omitempty"`
	Event              string         `json:"event,omitempty"`
	ExpectedStatus     int            `json:"expectedStatus,omitempty"`
	ExpectedBodyBase64 string         `json:"expectedBodyBase64,omitempty"`
}

type conformanceFile_ struct {
	Placeholders map[string]string `json:"placeholders"`
	Cases        []conformanceCase `json:"cases"`
}

// TestConformance replays every case from the language-neutral HTTP contract
// fixture against the reference-compatible test server.
func TestConformance(t *testing.T) {
	path := conformanceFile(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read conformance: %v", err)
	}
	var doc conformanceFile_
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse conformance: %v", err)
	}
	t.Logf("loaded %d conformance cases", len(doc.Cases))
	subs := map[string]string{}
	for k, v := range doc.Placeholders {
		subs[k] = v
	}

	pass := 0
	for _, c := range doc.Cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			runConformanceCase(t, c, subs)
			pass++
		})
	}
	t.Logf("conformance pass count: %d/%d", pass, len(doc.Cases))
}

// runConformanceCase wires a fresh backend per case so state never leaks
// between cases.
func runConformanceCase(t *testing.T, c conformanceCase, placeholders map[string]string) {
	t.Helper()
	const brainID = "conformance-brain"
	backend := newMemBackend()
	srv := newTestServer(t, brainID, backend)

	subs := map[string]string{}
	for k, v := range placeholders {
		if k == "BRAIN_ID" {
			continue // placeholder file carries a description, not a value
		}
		subs[k] = v
	}
	subs["BRAIN_ID"] = brainID

	// Per-case SSE state map keyed by step.name.
	sseCtx, sseCancel := context.WithCancel(context.Background())
	defer sseCancel()
	ssePool := map[string]*sseSubscriber{}
	defer func() {
		for _, s := range ssePool {
			s.close()
		}
	}()

	// Replace longest placeholders first so prefixes like BASE64_C do not
	// accidentally consume BASE64_CONTENT.
	keys := make([]string, 0, len(subs))
	for k := range subs {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return len(keys[i]) > len(keys[j]) })
	replaceAll := func(s string) string {
		out := s
		for _, k := range keys {
			out = strings.ReplaceAll(out, k, subs[k])
		}
		return out
	}

	for _, step := range c.Setup {
		runConformanceStep(t, srv, backend, sseCtx, ssePool, step, replaceAll)
	}

	// Execute the main request. The request field may itself be an SSE
	// control step (for SSE await cases).
	if kind, _ := c.Request["kind"].(string); kind == "await-sse-event" {
		name, _ := c.Request["name"].(string)
		event, _ := c.Request["event"].(string)
		sub, ok := ssePool[name]
		if !ok {
			t.Fatalf("SSE subscriber %q not opened", name)
		}
		raw, ok := sub.waitForEvent(event, 5*time.Second)
		if !ok {
			t.Fatalf("timeout waiting for SSE event %q", event)
		}
		assertSSEEvent(t, c.ExpectedResponse, raw)
	} else {
		mainStep := stepFromMap(c.Request)
		resp := doConformanceRequest(t, srv, mainStep, replaceAll)
		defer func() { _ = resp.Body.Close() }()
		assertExpectedResponse(t, c.ExpectedResponse, resp, replaceAll)
	}

	for _, step := range c.FollowUp {
		runConformanceStep(t, srv, backend, sseCtx, ssePool, step, replaceAll)
	}
	for _, step := range c.Teardown {
		runConformanceStep(t, srv, backend, sseCtx, ssePool, step, replaceAll)
	}
}

// stepFromMap coerces an arbitrary map (read from JSON) into a step struct.
func stepFromMap(m map[string]any) conformanceStep {
	b, _ := json.Marshal(m)
	var s conformanceStep
	_ = json.Unmarshal(b, &s)
	return s
}

// runConformanceStep dispatches either an HTTP step or an SSE control step.
func runConformanceStep(
	t *testing.T,
	srv *httptest.Server,
	backend *memBackend,
	ctx context.Context,
	pool map[string]*sseSubscriber,
	step conformanceStep,
	subst func(string) string,
) {
	t.Helper()
	switch step.Kind {
	case "open-sse":
		if _, dup := pool[step.Name]; dup {
			pool[step.Name].close()
		}
		sub := newSSESubscriber(ctx, srv.URL+subst(step.Path))
		pool[step.Name] = sub
	case "await-sse-event":
		sub, ok := pool[step.Name]
		if !ok {
			t.Fatalf("SSE subscriber %q not opened", step.Name)
		}
		if _, ok := sub.waitForEvent(step.Event, 5*time.Second); !ok {
			t.Fatalf("timeout waiting for SSE event %q", step.Event)
		}
	case "close-sse":
		if sub, ok := pool[step.Name]; ok {
			sub.close()
			delete(pool, step.Name)
		}
	case "":
		// plain HTTP step
		resp := doConformanceRequest(t, srv, step, subst)
		defer func() { _ = resp.Body.Close() }()
		if step.ExpectedStatus != 0 && resp.StatusCode != step.ExpectedStatus {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("setup step %s %s: want status %d got %d body=%s",
				step.Method, step.Path, step.ExpectedStatus, resp.StatusCode, string(body))
		}
		if step.ExpectedBodyBase64 != "" {
			body, _ := io.ReadAll(resp.Body)
			want, _ := base64.StdEncoding.DecodeString(subst(step.ExpectedBodyBase64))
			if !bytes.Equal(body, want) {
				t.Fatalf("setup step body mismatch: want %q got %q", want, body)
			}
		}
	default:
		t.Fatalf("unknown step kind %q", step.Kind)
	}
	_ = backend
}

// doConformanceRequest issues the HTTP request described by step and returns
// the raw response for the caller to assert on.
func doConformanceRequest(t *testing.T, srv *httptest.Server, step conformanceStep, subst func(string) string) *http.Response {
	t.Helper()
	target := srv.URL + subst(step.Path)
	var body io.Reader
	if step.BodyBase64 != "" {
		decoded, err := base64.StdEncoding.DecodeString(subst(step.BodyBase64))
		if err != nil {
			t.Fatalf("decode bodyBase64: %v", err)
		}
		body = bytes.NewReader(decoded)
	} else if step.BodyJSON != nil {
		b, err := json.Marshal(step.BodyJSON)
		if err != nil {
			t.Fatalf("marshal bodyJson: %v", err)
		}
		body = bytes.NewReader(b)
	}
	method := step.Method
	if method == "" {
		method = http.MethodGet
	}
	req, err := http.NewRequest(method, target, body)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	// Default content-type for JSON bodies the fixture is explicit.
	if step.BodyJSON != nil {
		req.Header.Set("Content-Type", "application/json")
	} else if step.BodyBase64 != "" {
		req.Header.Set("Content-Type", "application/octet-stream")
	}
	for k, v := range step.Headers {
		req.Header.Set(k, toString(v))
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	return resp
}

// assertExpectedResponse walks the expectedResponse map and verifies each
// assertion kind documented in the fixture.
func assertExpectedResponse(t *testing.T, expected map[string]any, resp *http.Response, subst func(string) string) {
	t.Helper()
	if status, ok := numberField(expected, "status"); ok {
		if resp.StatusCode != status {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("want status %d got %d body=%q", status, resp.StatusCode, body)
		}
	}
	if ct, ok := expected["contentType"].(string); ok {
		actual := resp.Header.Get("Content-Type")
		if !strings.Contains(actual, ct) {
			t.Fatalf("want content-type %q got %q", ct, actual)
		}
	}
	if wantB64, ok := expected["bodyBase64"].(string); ok {
		body, _ := io.ReadAll(resp.Body)
		want, _ := base64.StdEncoding.DecodeString(subst(wantB64))
		if !bytes.Equal(body, want) {
			t.Fatalf("body mismatch: want %q got %q", want, body)
		}
	}
	if wantBody, ok := expected["body"]; ok {
		body, _ := io.ReadAll(resp.Body)
		assertJSONMatches(t, wantBody, body)
	}
	if assertions, ok := expected["bodyAssertions"].([]any); ok {
		body, _ := io.ReadAll(resp.Body)
		for _, raw := range assertions {
			assertion, _ := raw.(map[string]any)
			runBodyAssertion(t, assertion, body)
		}
	}
	if streamAsserts, ok := expected["streamAssertions"].([]any); ok {
		// SSE stream assertions: consume the response body until every
		// expect-event has been observed.
		want := map[string]bool{}
		for _, raw := range streamAsserts {
			assertion, _ := raw.(map[string]any)
			if kind, _ := assertion["kind"].(string); kind == "expect-event" {
				name, _ := assertion["event"].(string)
				want[name] = false
			}
		}
		seen := map[string]bool{}
		deadline := time.Now().Add(3 * time.Second)
		reader := newSSEReader(resp.Body)
		for time.Now().Before(deadline) && len(seen) < len(want) {
			evt, _, err := reader.readFrame()
			if err != nil {
				break
			}
			if _, ok := want[evt]; ok {
				seen[evt] = true
			}
		}
		_ = reader.close()
		for name := range want {
			if !seen[name] {
				t.Fatalf("expected SSE event %q but it never arrived", name)
			}
		}
	}
}

// assertJSONMatches compares expected JSON against actual response body,
// tolerating the "<ISO-8601>" marker as a parseability assertion.
func assertJSONMatches(t *testing.T, expected any, actual []byte) {
	t.Helper()
	var got any
	if err := json.Unmarshal(actual, &got); err != nil {
		t.Fatalf("decode response JSON: %v body=%q", err, actual)
	}
	if err := compareJSON(expected, got); err != nil {
		t.Fatalf("JSON mismatch: %v\nwant=%v\ngot=%v", err, expected, got)
	}
}

func compareJSON(expected, actual any) error {
	switch e := expected.(type) {
	case map[string]any:
		a, ok := actual.(map[string]any)
		if !ok {
			return structureErr("want object got %T", actual)
		}
		for k, v := range e {
			av, present := a[k]
			if !present {
				return structureErr("missing key %q", k)
			}
			if err := compareJSON(v, av); err != nil {
				return structureErr("%q: %v", k, err)
			}
		}
		return nil
	case []any:
		a, ok := actual.([]any)
		if !ok {
			return structureErr("want array got %T", actual)
		}
		if len(a) < len(e) {
			return structureErr("array length %d < expected %d", len(a), len(e))
		}
		for i, v := range e {
			if err := compareJSON(v, a[i]); err != nil {
				return structureErr("[%d]: %v", i, err)
			}
		}
		return nil
	case string:
		if e == "<ISO-8601>" {
			s, ok := actual.(string)
			if !ok {
				return structureErr("want ISO-8601 string got %T", actual)
			}
			if _, err := time.Parse(time.RFC3339Nano, s); err != nil {
				return structureErr("not ISO-8601: %v", err)
			}
			return nil
		}
		s, ok := actual.(string)
		if !ok || s != e {
			return structureErr("want %q got %v", e, actual)
		}
		return nil
	case float64:
		n, ok := numericValue(actual)
		if !ok || n != e {
			return structureErr("want number %v got %v", e, actual)
		}
		return nil
	case bool:
		b, ok := actual.(bool)
		if !ok || b != e {
			return structureErr("want %v got %v", e, actual)
		}
		return nil
	case nil:
		if actual != nil {
			return structureErr("want nil got %v", actual)
		}
		return nil
	default:
		return structureErr("unhandled expected type %T", expected)
	}
}

func numericValue(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

type jsonErr struct{ msg string }

func (e *jsonErr) Error() string { return e.msg }

func structureErr(format string, args ...any) error {
	return &jsonErr{msg: sprintfHelper(format, args...)}
}

// sprintfHelper avoids pulling in fmt in the hot path of compareJSON.
// Kept as a thin wrapper so the call sites read cleanly.
func sprintfHelper(format string, args ...any) string {
	// Defer to stdlib.
	return stdlibSprintf(format, args...)
}

// runBodyAssertion implements the bodyAssertions vocabulary documented in
// spec/conformance/README.md (items-include-path, items-files-equal, etc.).
func runBodyAssertion(t *testing.T, assertion map[string]any, body []byte) {
	t.Helper()
	kind, _ := assertion["kind"].(string)
	switch kind {
	case "items-include-path":
		wantPath, _ := assertion["path"].(string)
		items := extractItems(t, body)
		for _, it := range items {
			if p, _ := it["path"].(string); p == wantPath {
				return
			}
		}
		t.Fatalf("items do not include %q. items=%v", wantPath, items)
	case "items-exclude-path":
		wantPath, _ := assertion["path"].(string)
		items := extractItems(t, body)
		for _, it := range items {
			if p, _ := it["path"].(string); p == wantPath {
				t.Fatalf("items unexpectedly include %q. items=%v", wantPath, items)
			}
		}
	case "items-files-equal":
		want := toStringSlice(assertion["paths"])
		items := extractItems(t, body)
		var got []string
		for _, it := range items {
			if isDir, _ := it["is_dir"].(bool); isDir {
				continue
			}
			if p, _ := it["path"].(string); p != "" {
				got = append(got, p)
			}
		}
		if !stringSetEqual(got, want) {
			t.Fatalf("items-files-equal: want %v got %v", want, got)
		}
	case "items-dirs-equal":
		want := toStringSlice(assertion["paths"])
		items := extractItems(t, body)
		var got []string
		for _, it := range items {
			if isDir, _ := it["is_dir"].(bool); !isDir {
				continue
			}
			if p, _ := it["path"].(string); p != "" {
				got = append(got, p)
			}
		}
		if !stringSetEqual(got, want) {
			t.Fatalf("items-dirs-equal: want %v got %v", want, got)
		}
	case "json-field-equals":
		field, _ := assertion["field"].(string)
		wantValue := assertion["value"]
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("decode body: %v body=%q", err, body)
		}
		if got := parsed[field]; got != wantValue {
			t.Fatalf("field %q: want %v got %v", field, wantValue, got)
		}
	default:
		t.Fatalf("unknown bodyAssertion kind %q", kind)
	}
}

// extractItems pulls the items array from a listing response.
func extractItems(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var outer struct {
		Items []map[string]any `json:"items"`
	}
	if err := json.Unmarshal(body, &outer); err != nil {
		t.Fatalf("decode list body: %v body=%q", err, body)
	}
	return outer.Items
}

// assertSSEEvent applies an expectedResponse whose assertions target a
// single SSE change frame's JSON payload.
func assertSSEEvent(t *testing.T, expected map[string]any, raw string) {
	t.Helper()
	if assertions, ok := expected["bodyAssertions"].([]any); ok {
		for _, a := range assertions {
			assertion, _ := a.(map[string]any)
			runBodyAssertion(t, assertion, []byte(raw))
		}
	}
}

// toString best-effort string coerce for header values read out of JSON.
func toString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case float64:
		return stdlibSprintf("%v", t)
	}
	return ""
}

// numberField reads a numeric field from a decoded JSON object.
func numberField(m map[string]any, key string) (int, bool) {
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	n, ok := numericValue(v)
	if !ok {
		return 0, false
	}
	return int(n), true
}

func toStringSlice(v any) []string {
	arr, _ := v.([]any)
	out := make([]string, 0, len(arr))
	for _, a := range arr {
		if s, ok := a.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func stringSetEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	am := map[string]int{}
	bm := map[string]int{}
	for _, s := range a {
		am[s]++
	}
	for _, s := range b {
		bm[s]++
	}
	for k, v := range am {
		if bm[k] != v {
			return false
		}
	}
	return true
}

// sseSubscriber models the open-sse/await-sse-event/close-sse triad. It
// runs in its own goroutine, buffering frames until the harness awaits
// them.
type sseSubscriber struct {
	cancel context.CancelFunc
	mu     sync.Mutex
	cond   *sync.Cond
	events []sseFrame
	closed bool
}

type sseFrame struct {
	Event string
	Data  string
}

func newSSESubscriber(ctx context.Context, url string) *sseSubscriber {
	ctx, cancel := context.WithCancel(ctx)
	s := &sseSubscriber{cancel: cancel}
	s.cond = sync.NewCond(&s.mu)
	go s.run(ctx, url)
	return s
}

func (s *sseSubscriber) run(ctx context.Context, url string) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept", "text/event-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return
	}
	defer func() { _ = resp.Body.Close() }()
	reader := newSSEReader(resp.Body)
	defer func() { _ = reader.close() }()
	for {
		event, data, err := reader.readFrame()
		if err != nil {
			return
		}
		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			return
		}
		s.events = append(s.events, sseFrame{Event: event, Data: data})
		s.cond.Broadcast()
		s.mu.Unlock()
	}
}

func (s *sseSubscriber) waitForEvent(name string, timeout time.Duration) (string, bool) {
	deadline := time.Now().Add(timeout)
	s.mu.Lock()
	defer s.mu.Unlock()
	cursor := 0
	for {
		for i := cursor; i < len(s.events); i++ {
			if s.events[i].Event == name {
				cursor = i + 1
				return s.events[i].Data, true
			}
		}
		cursor = len(s.events)
		if time.Now().After(deadline) {
			return "", false
		}
		waitCh := make(chan struct{})
		go func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.cond.Wait()
			close(waitCh)
		}()
		// Release the lock so the reader can append; re-acquire via cond wait.
		s.mu.Unlock()
		timer := time.NewTimer(200 * time.Millisecond)
		select {
		case <-waitCh:
		case <-timer.C:
			s.cond.Broadcast()
		}
		timer.Stop()
		s.mu.Lock()
	}
}

func (s *sseSubscriber) close() {
	s.mu.Lock()
	s.closed = true
	s.cond.Broadcast()
	s.mu.Unlock()
	s.cancel()
}

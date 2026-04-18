// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
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

// conformanceFile resolves the language-neutral fixture path by
// walking up the working directory until it finds spec/conformance.
func conformanceFile(t *testing.T) string {
	t.Helper()
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

// conformanceCase / conformanceStep mirror the JSON fixture shape.
type conformanceCase struct {
	Name             string            `json:"name"`
	Setup            []conformanceStep `json:"setup,omitempty"`
	Request          map[string]any    `json:"request"`
	ExpectedResponse map[string]any    `json:"expectedResponse"`
	FollowUp         []conformanceStep `json:"followUp,omitempty"`
	Teardown         []conformanceStep `json:"teardown,omitempty"`
}

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

// known cases that exercise wire features the daemon does not yet
// implement (auth header forwarding inside the harness, idempotency,
// etc.). Skipped explicitly so the pass tally reflects what the
// daemon honours rather than masking real regressions.
var skipConformanceCases = map[string]string{
	"auth header forwarded when apiKey is set": "harness asserts header forwarding rather than server behaviour",
}

// TestConformance replays every case from spec/conformance against a
// freshly provisioned daemon. Each case runs in its own subtest so
// targeted skips and failures stay scoped.
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
	pass := 0
	skipped := 0
	for _, c := range doc.Cases {
		c := c
		ok := t.Run(c.Name, func(t *testing.T) {
			if reason, skip := skipConformanceCases[c.Name]; skip {
				skipped++
				t.Skip(reason)
				return
			}
			runConformanceCase(t, c, doc.Placeholders)
		})
		if ok {
			pass++
		}
	}
	t.Logf("conformance pass count: %d/%d (skipped %d)", pass-skipped, len(doc.Cases), skipped)
}

// runConformanceCase wires a fresh daemon per case so state never
// leaks between cases.
func runConformanceCase(t *testing.T, c conformanceCase, placeholders map[string]string) {
	t.Helper()
	const brainID = "conformance-brain"

	root := t.TempDir()
	log := slog.New(slog.NewJSONHandler(io.Discard, nil))
	daemon, err := NewDaemon(context.Background(), root, "", log)
	if err != nil {
		t.Fatalf("daemon: %v", err)
	}
	if _, err := daemon.Brains.Create(context.Background(), brainID); err != nil {
		t.Fatalf("create brain: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	daemon.RegisterRoutes(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		_ = daemon.Close()
	})

	subs := map[string]string{}
	for k, v := range placeholders {
		if k == "BRAIN_ID" {
			continue
		}
		subs[k] = v
	}
	subs["BRAIN_ID"] = brainID

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

	sseCtx, sseCancel := context.WithCancel(context.Background())
	defer sseCancel()
	ssePool := map[string]*sseSubscriber{}
	defer func() {
		for _, s := range ssePool {
			s.close()
		}
	}()

	for _, step := range c.Setup {
		runConformanceStep(t, srv, sseCtx, ssePool, step, replaceAll)
	}

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
		runConformanceStep(t, srv, sseCtx, ssePool, step, replaceAll)
	}
	for _, step := range c.Teardown {
		runConformanceStep(t, srv, sseCtx, ssePool, step, replaceAll)
	}
}

func stepFromMap(m map[string]any) conformanceStep {
	b, _ := json.Marshal(m)
	var s conformanceStep
	_ = json.Unmarshal(b, &s)
	return s
}

func runConformanceStep(t *testing.T, srv *httptest.Server, ctx context.Context, pool map[string]*sseSubscriber, step conformanceStep, subst func(string) string) {
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
}

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
	if step.BodyJSON != nil {
		req.Header.Set("Content-Type", "application/json")
	} else if step.BodyBase64 != "" {
		req.Header.Set("Content-Type", "application/octet-stream")
	}
	for k, v := range step.Headers {
		req.Header.Set(k, fmt.Sprint(v))
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	return resp
}

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
		reader := bufio.NewScanner(resp.Body)
		reader.Buffer(make([]byte, 0, 64*1024), 1<<20)
		var event string
		for time.Now().Before(deadline) && len(seen) < len(want) {
			if !reader.Scan() {
				break
			}
			line := reader.Text()
			if line == "" {
				if event != "" {
					if _, ok := want[event]; ok {
						seen[event] = true
					}
				}
				event = ""
				continue
			}
			if strings.HasPrefix(line, "event:") {
				event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			}
		}
		for name := range want {
			if !seen[name] {
				t.Fatalf("expected SSE event %q but it never arrived", name)
			}
		}
	}
}

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
			return fmt.Errorf("want object got %T", actual)
		}
		for k, v := range e {
			av, present := a[k]
			if !present {
				return fmt.Errorf("missing key %q", k)
			}
			if err := compareJSON(v, av); err != nil {
				return fmt.Errorf("%q: %v", k, err)
			}
		}
		return nil
	case []any:
		a, ok := actual.([]any)
		if !ok {
			return fmt.Errorf("want array got %T", actual)
		}
		if len(a) < len(e) {
			return fmt.Errorf("array length %d < expected %d", len(a), len(e))
		}
		for i, v := range e {
			if err := compareJSON(v, a[i]); err != nil {
				return fmt.Errorf("[%d]: %v", i, err)
			}
		}
		return nil
	case string:
		if e == "<ISO-8601>" {
			s, ok := actual.(string)
			if !ok {
				return fmt.Errorf("want ISO-8601 string got %T", actual)
			}
			if _, err := time.Parse(time.RFC3339Nano, s); err != nil {
				return fmt.Errorf("not ISO-8601: %v", err)
			}
			return nil
		}
		s, ok := actual.(string)
		if !ok || s != e {
			return fmt.Errorf("want %q got %v", e, actual)
		}
		return nil
	case float64:
		n, ok := numericValue(actual)
		if !ok || n != e {
			return fmt.Errorf("want number %v got %v", e, actual)
		}
		return nil
	case bool:
		b, ok := actual.(bool)
		if !ok || b != e {
			return fmt.Errorf("want %v got %v", e, actual)
		}
		return nil
	case nil:
		if actual != nil {
			return fmt.Errorf("want nil got %v", actual)
		}
		return nil
	default:
		return fmt.Errorf("unhandled expected type %T", expected)
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

func assertSSEEvent(t *testing.T, expected map[string]any, raw string) {
	t.Helper()
	if assertions, ok := expected["bodyAssertions"].([]any); ok {
		for _, a := range assertions {
			assertion, _ := a.(map[string]any)
			runBodyAssertion(t, assertion, []byte(raw))
		}
	}
}

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

// sseSubscriber, sseFrame, newSSESubscriber, waitForEvent, close
// match the helpers in store/http/conformance_test.go so the
// behaviour is identical across SDKs.
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

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	var event string
	var data strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			s.mu.Lock()
			if s.closed {
				s.mu.Unlock()
				return
			}
			s.events = append(s.events, sseFrame{Event: event, Data: data.String()})
			s.cond.Broadcast()
			s.mu.Unlock()
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

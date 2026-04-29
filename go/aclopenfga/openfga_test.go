// SPDX-License-Identifier: Apache-2.0

package aclopenfga

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/jeffs-brain/memory/go/acl"
)

func userSubject(id string) acl.Subject { return acl.Subject{Kind: acl.SubjectUser, ID: id} }
func brainResource(id string) acl.Resource {
	return acl.Resource{Type: acl.ResourceBrain, ID: id}
}

// recordingServer captures every request the provider issues so tests
// can assert on URL, headers and JSON bodies.
type recordingServer struct {
	mu       sync.Mutex
	srv      *httptest.Server
	requests []recordedRequest
	respond  func(req *http.Request) (status int, body string)
}

type recordedRequest struct {
	method  string
	path    string
	headers http.Header
	body    string
}

func newRecordingServer(t *testing.T, respond func(req *http.Request) (status int, body string)) *recordingServer {
	rs := &recordingServer{respond: respond}
	rs.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("server read body: %v", err)
		}
		rs.mu.Lock()
		rs.requests = append(rs.requests, recordedRequest{
			method:  r.Method,
			path:    r.URL.Path,
			headers: r.Header.Clone(),
			body:    string(body),
		})
		rs.mu.Unlock()
		status, payload := rs.respond(r)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, payload)
	}))
	t.Cleanup(rs.srv.Close)
	return rs
}

func (rs *recordingServer) URL() string { return rs.srv.URL }

func (rs *recordingServer) lastRequest(t *testing.T) recordedRequest {
	t.Helper()
	rs.mu.Lock()
	defer rs.mu.Unlock()
	if len(rs.requests) == 0 {
		t.Fatal("no requests recorded")
	}
	return rs.requests[len(rs.requests)-1]
}

func (rs *recordingServer) callCount() int {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return len(rs.requests)
}

func TestNewProviderRequiresAPIURL(t *testing.T) {
	if _, err := NewProvider(Options{StoreID: "s"}); err == nil {
		t.Fatal("expected error for missing APIURL")
	}
}

func TestNewProviderRequiresStoreID(t *testing.T) {
	if _, err := NewProvider(Options{APIURL: "https://x"}); err == nil {
		t.Fatal("expected error for missing StoreID")
	}
}

func TestCheckSendsCorrectShape(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"allowed": true}`
	})
	provider, err := NewProvider(Options{
		APIURL:  rs.URL(),
		StoreID: "store-1",
		ModelID: "model-42",
		Token:   "secret",
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	res, err := provider.Check(context.Background(), userSubject("alice"), acl.ActionWrite, brainResource("notes"))
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if !res.Allowed {
		t.Errorf("Check returned %+v", res)
	}

	if rs.callCount() != 1 {
		t.Fatalf("expected 1 request, got %d", rs.callCount())
	}
	req := rs.lastRequest(t)
	if req.method != http.MethodPost {
		t.Errorf("method = %s, want POST", req.method)
	}
	if req.path != "/stores/store-1/check" {
		t.Errorf("path = %s, want /stores/store-1/check", req.path)
	}
	if got := req.headers.Get("Authorization"); got != "Bearer secret" {
		t.Errorf("Authorization = %q, want Bearer secret", got)
	}
	if got := req.headers.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}

	var body struct {
		TupleKey struct {
			User     string `json:"user"`
			Relation string `json:"relation"`
			Object   string `json:"object"`
		} `json:"tuple_key"`
		AuthorizationModelID string `json:"authorization_model_id"`
	}
	if err := json.Unmarshal([]byte(req.body), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body.TupleKey.User != "user:alice" || body.TupleKey.Relation != "writer" || body.TupleKey.Object != "brain:notes" {
		t.Errorf("tuple_key = %+v", body.TupleKey)
	}
	if body.AuthorizationModelID != "model-42" {
		t.Errorf("model id = %q, want model-42", body.AuthorizationModelID)
	}
}

func TestCheckMapsAllowedFalseToDeny(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"allowed": false, "resolution": "no tuple"}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL() + "/", StoreID: "store-1"})
	res, err := provider.Check(context.Background(), userSubject("alice"), acl.ActionRead, brainResource("notes"))
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if res.Allowed {
		t.Errorf("expected deny, got %+v", res)
	}
	if res.Reason != "no tuple" {
		t.Errorf("reason = %q, want no tuple", res.Reason)
	}
}

func TestCheckDefaultsDenyReasonWhenServerOmitsIt(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"allowed": false}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	res, _ := provider.Check(context.Background(), userSubject("alice"), acl.ActionRead, brainResource("notes"))
	if res.Reason == "" {
		t.Fatal("expected default reason for missing resolution")
	}
}

func TestCheckReturnsHTTPErrorOnNon2xx(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 500, "boom"
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	_, err := provider.Check(context.Background(), userSubject("alice"), acl.ActionRead, brainResource("notes"))
	var herr *HTTPError
	if !errors.As(err, &herr) {
		t.Fatalf("error = %v, want *HTTPError", err)
	}
	if herr.Status != 500 {
		t.Errorf("status = %d, want 500", herr.Status)
	}
	if herr.Endpoint != "/check" {
		t.Errorf("endpoint = %q, want /check", herr.Endpoint)
	}
}

func TestCheckOmitsAuthorizationHeaderWhenNoToken(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"allowed": true}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	if _, err := provider.Check(context.Background(), userSubject("alice"), acl.ActionRead, brainResource("notes")); err != nil {
		t.Fatalf("Check: %v", err)
	}
	req := rs.lastRequest(t)
	if got := req.headers.Get("Authorization"); got != "" {
		t.Errorf("Authorization = %q, want empty", got)
	}
}

func TestCheckHandlesUnknownAction(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"allowed": true}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	res, err := provider.Check(context.Background(), userSubject("alice"), acl.Action("frobnicate"), brainResource("notes"))
	if err != nil {
		t.Fatalf("Check: %v", err)
	}
	if res.Allowed {
		t.Fatal("unknown action should deny without a server round-trip")
	}
	if rs.callCount() != 0 {
		t.Errorf("expected no HTTP calls, got %d", rs.callCount())
	}
}

func TestWriteSendsCorrectBody(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1", ModelID: "model-42"})
	err := provider.Write(context.Background(), acl.WriteTuplesRequest{
		Writes:  []acl.Tuple{{Subject: userSubject("alice"), Relation: "writer", Resource: brainResource("notes")}},
		Deletes: []acl.Tuple{{Subject: userSubject("bob"), Relation: "reader", Resource: brainResource("notes")}},
	})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	req := rs.lastRequest(t)
	if req.path != "/stores/store-1/write" {
		t.Errorf("path = %q", req.path)
	}
	var body struct {
		Writes *struct {
			TupleKeys []map[string]string `json:"tuple_keys"`
		} `json:"writes,omitempty"`
		Deletes *struct {
			TupleKeys []map[string]string `json:"tuple_keys"`
		} `json:"deletes,omitempty"`
		AuthorizationModelID string `json:"authorization_model_id"`
	}
	if err := json.Unmarshal([]byte(req.body), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body.Writes == nil || len(body.Writes.TupleKeys) != 1 {
		t.Fatalf("writes = %+v", body.Writes)
	}
	w0 := body.Writes.TupleKeys[0]
	if w0["user"] != "user:alice" || w0["relation"] != "writer" || w0["object"] != "brain:notes" {
		t.Errorf("write tuple = %+v", w0)
	}
	if body.Deletes == nil || len(body.Deletes.TupleKeys) != 1 {
		t.Fatalf("deletes = %+v", body.Deletes)
	}
	d0 := body.Deletes.TupleKeys[0]
	if d0["user"] != "user:bob" || d0["relation"] != "reader" || d0["object"] != "brain:notes" {
		t.Errorf("delete tuple = %+v", d0)
	}
	if body.AuthorizationModelID != "model-42" {
		t.Errorf("model id = %q", body.AuthorizationModelID)
	}
}

func TestWriteEmptyDoesNotCallServer(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	if err := provider.Write(context.Background(), acl.WriteTuplesRequest{}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if rs.callCount() != 0 {
		t.Errorf("expected no HTTP calls, got %d", rs.callCount())
	}
}

func TestWriteWrapsTransportFailures(t *testing.T) {
	// A client whose Transport always errors out simulates a network
	// failure without bringing httptest into the picture.
	provider, _ := NewProvider(Options{
		APIURL:  "http://invalid.example",
		StoreID: "store-1",
		HTTPClient: &http.Client{Transport: failingTransport{
			err: errors.New("network down"),
		}},
	})
	err := provider.Write(context.Background(), acl.WriteTuplesRequest{
		Writes: []acl.Tuple{{Subject: userSubject("a"), Relation: "reader", Resource: brainResource("x")}},
	})
	var rerr *RequestError
	if !errors.As(err, &rerr) {
		t.Fatalf("error = %v, want *RequestError", err)
	}
	if !strings.Contains(err.Error(), "network down") {
		t.Errorf("error %q missing transport message", err.Error())
	}
	if !errors.Is(err, rerr.Cause) {
		t.Errorf("Unwrap chain broken")
	}
}

type failingTransport struct{ err error }

func (f failingTransport) RoundTrip(*http.Request) (*http.Response, error) { return nil, f.err }

func TestReadDecodesTuples(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"tuples": [
			{"key": {"user": "user:alice", "relation": "writer", "object": "brain:notes"}},
			{"key": {"user": "junk", "relation": "writer", "object": "brain:notes"}}
		]}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	target := brainResource("notes")
	tuples, err := provider.Read(context.Background(), acl.ReadTuplesQuery{Resource: &target})
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(tuples) != 1 {
		t.Fatalf("decoded %d tuples, want 1", len(tuples))
	}
	got := tuples[0]
	if got.Subject != userSubject("alice") {
		t.Errorf("subject = %+v", got.Subject)
	}
	if got.Relation != "writer" {
		t.Errorf("relation = %q", got.Relation)
	}
	if got.Resource != brainResource("notes") {
		t.Errorf("resource = %+v", got.Resource)
	}
}

func TestReadFiltersOutMissingFields(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"tuples": [
			{"key": {"user": "user:alice", "relation": "writer"}},
			{"key": {"user": "user:alice", "object": "brain:notes"}},
			{"key": {"relation": "writer", "object": "brain:notes"}}
		]}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	tuples, err := provider.Read(context.Background(), acl.ReadTuplesQuery{})
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(tuples) != 0 {
		t.Errorf("Read returned %d tuples, want 0", len(tuples))
	}
}

func TestReadSendsTupleKeyFilters(t *testing.T) {
	rs := newRecordingServer(t, func(*http.Request) (int, string) {
		return 200, `{"tuples": []}`
	})
	provider, _ := NewProvider(Options{APIURL: rs.URL(), StoreID: "store-1"})
	subj := userSubject("alice")
	resrc := brainResource("notes")
	if _, err := provider.Read(context.Background(), acl.ReadTuplesQuery{
		Subject: &subj, Relation: "writer", Resource: &resrc,
	}); err != nil {
		t.Fatalf("Read: %v", err)
	}
	req := rs.lastRequest(t)
	var body struct {
		TupleKey map[string]string `json:"tuple_key"`
	}
	if err := json.Unmarshal([]byte(req.body), &body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body.TupleKey["user"] != "user:alice" || body.TupleKey["relation"] != "writer" || body.TupleKey["object"] != "brain:notes" {
		t.Errorf("tuple_key = %+v", body.TupleKey)
	}
}

func TestEncodeDecodeSubject(t *testing.T) {
	cases := []acl.Subject{
		{Kind: acl.SubjectUser, ID: "alice"},
		{Kind: acl.SubjectAPIKey, ID: "key-1"},
		{Kind: acl.SubjectService, ID: "workspace:acme"},
	}
	for _, want := range cases {
		got, ok := DecodeSubject(EncodeSubject(want))
		if !ok || got != want {
			t.Errorf("round-trip %+v failed: got %+v ok=%v", want, got, ok)
		}
	}
	if _, ok := DecodeSubject("nope"); ok {
		t.Error("DecodeSubject accepted malformed input")
	}
	if _, ok := DecodeSubject("alien:x"); ok {
		t.Error("DecodeSubject accepted unknown kind")
	}
}

func TestEncodeDecodeResource(t *testing.T) {
	cases := []acl.Resource{
		{Type: acl.ResourceWorkspace, ID: "acme"},
		{Type: acl.ResourceBrain, ID: "notes"},
		{Type: acl.ResourceCollection, ID: "meetings"},
		{Type: acl.ResourceDocument, ID: "doc-1"},
	}
	for _, want := range cases {
		got, ok := DecodeResource(EncodeResource(want))
		if !ok || got != want {
			t.Errorf("round-trip %+v failed: got %+v ok=%v", want, got, ok)
		}
	}
	if _, ok := DecodeResource("nope"); ok {
		t.Error("DecodeResource accepted malformed input")
	}
	if _, ok := DecodeResource("alien:x"); ok {
		t.Error("DecodeResource accepted unknown type")
	}
}

func TestCloseIsNoOpWithCallerSuppliedClient(t *testing.T) {
	caller := &http.Client{Transport: failingTransport{err: errors.New("unused")}}
	provider, err := NewProvider(Options{
		APIURL:     "https://fga.example",
		StoreID:    "store-1",
		HTTPClient: caller,
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	if err := provider.Close(); err != nil {
		t.Fatalf("Close returned %v, want nil", err)
	}
	// The caller's client must remain usable after Close: ownership
	// stays with the caller.
	if caller.Transport == nil {
		t.Fatal("caller transport was cleared by Close")
	}
}

func TestCloseIsNoOpWithDefaultClient(t *testing.T) {
	provider, err := NewProvider(Options{APIURL: "https://fga.example", StoreID: "store-1"})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}
	if err := provider.Close(); err != nil {
		t.Fatalf("Close returned %v, want nil", err)
	}
	// http.DefaultClient is process-wide; closing the adapter must not
	// touch its transport. A second close must also be safe.
	if err := provider.Close(); err != nil {
		t.Fatalf("second Close returned %v, want nil", err)
	}
}

func TestHTTPErrorBodyTruncation(t *testing.T) {
	long := strings.Repeat("x", 500)
	herr := &HTTPError{Status: 500, Endpoint: "/check", Body: long}
	msg := herr.Error()
	if !strings.Contains(msg, strings.Repeat("x", 200)) {
		t.Error("expected 200 chars of body in message")
	}
	if strings.Contains(msg, strings.Repeat("x", 201)) {
		t.Error("HTTPError did not truncate body to 200 chars")
	}
}

func TestRequestErrorUnwrap(t *testing.T) {
	root := errors.New("root")
	rerr := &RequestError{Endpoint: "/x", Cause: root}
	if !errors.Is(rerr, root) {
		t.Fatal("errors.Is failed for RequestError")
	}
}

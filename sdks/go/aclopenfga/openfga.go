// SPDX-License-Identifier: Apache-2.0

// Package aclopenfga implements [acl.Provider] against an OpenFGA
// HTTP API.
//
// The adapter is stateless and depends only on net/http; tests can
// inject a custom *http.Client. The authorisation model the adapter
// expects is documented in spec/openfga/schema.fga.
//
// Endpoints used (per OpenFGA HTTP API):
//
//	POST {api_url}/stores/{store_id}/check
//	POST {api_url}/stores/{store_id}/write
//	POST {api_url}/stores/{store_id}/read
package aclopenfga

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/jeffs-brain/memory/go/acl"
)

// Options configures a new OpenFGA provider. APIURL and StoreID are
// required; the rest are optional.
type Options struct {
	// APIURL is the root URL of the OpenFGA API, for example
	// "https://fga.example.com". A trailing slash is normalised away.
	APIURL string
	// StoreID identifies the OpenFGA store to talk to.
	StoreID string
	// ModelID, when non-empty, is forwarded as authorization_model_id
	// on every check/write request so callers can pin an exact model
	// version.
	ModelID string
	// Token is a bearer credential included in the Authorization
	// header. Omit for a public OpenFGA instance.
	Token string
	// HTTPClient is the client used to dispatch requests. Defaults to
	// http.DefaultClient.
	HTTPClient *http.Client
}

// actionToRelation is the canonical mapping from an [acl.Action] to
// the OpenFGA relation name in spec/openfga/schema.fga. Callers that
// run a variant model can sidestep this map by writing pre-encoded
// tuples directly via Write/Read.
var actionToRelation = map[acl.Action]string{
	acl.ActionRead:   "reader",
	acl.ActionWrite:  "writer",
	acl.ActionDelete: "can_delete",
	acl.ActionAdmin:  "admin",
	acl.ActionExport: "can_export",
}

// HTTPError is returned for any non-2xx response from the OpenFGA
// server. Body is truncated to the first 200 characters to keep error
// messages bounded; full responses can be captured by the caller's
// HTTP middleware if needed.
type HTTPError struct {
	Status   int
	Endpoint string
	Body     string
}

func (e *HTTPError) Error() string {
	body := e.Body
	if len(body) > 200 {
		body = body[:200]
	}
	return fmt.Sprintf("openfga: %s failed: HTTP %d: %s", e.Endpoint, e.Status, body)
}

// RequestError wraps any non-HTTP failure (network, marshalling,
// unmarshalling). The cause chain is preserved so errors.Is and
// errors.As keep working.
type RequestError struct {
	Endpoint string
	Cause    error
}

func (e *RequestError) Error() string {
	return fmt.Sprintf("openfga: %s request failed: %s", e.Endpoint, e.Cause)
}

func (e *RequestError) Unwrap() error { return e.Cause }

// NewProvider validates opts and returns an [acl.Provider] backed by
// an OpenFGA HTTP server. Returns an error when APIURL or StoreID is
// missing.
func NewProvider(opts Options) (acl.Provider, error) {
	if opts.APIURL == "" {
		return nil, errors.New("aclopenfga: APIURL is required")
	}
	if opts.StoreID == "" {
		return nil, errors.New("aclopenfga: StoreID is required")
	}
	// ownsClient stays false today: a caller-supplied client is owned
	// by the caller, and the http.DefaultClient fallback is process-
	// wide and must not be closed. The flag exists so a future change
	// that gives the adapter its own *http.Transport can flip it
	// without altering the public surface.
	client := opts.HTTPClient
	ownsClient := false
	if client == nil {
		client = http.DefaultClient
	}
	baseURL := strings.TrimRight(opts.APIURL, "/")
	return &provider{
		storePath:  baseURL + "/stores/" + opts.StoreID,
		modelID:    opts.ModelID,
		token:      opts.Token,
		client:     client,
		ownsClient: ownsClient,
	}, nil
}

type provider struct {
	storePath string
	modelID   string
	token     string
	client    *http.Client
	// ownsClient records whether the adapter is responsible for
	// releasing the client's transport when Close is called. See
	// [NewProvider] for the rationale.
	ownsClient bool
}

func (p *provider) Name() string { return "openfga" }

// Close releases adapter-owned resources. Today this is always a
// no-op: a caller-supplied *http.Client is owned by the caller, and a
// nil client falls back to http.DefaultClient which is process-wide
// and must never be closed. The method exists so the [acl.Provider]
// contract is honoured uniformly and so a future change that gives
// the adapter its own *http.Transport can release it here without a
// breaking API change.
func (p *provider) Close() error {
	if !p.ownsClient {
		return nil
	}
	if transport, ok := p.client.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
	return nil
}

// checkResponse is the subset of the OpenFGA /check response we use.
type checkResponse struct {
	Allowed    *bool  `json:"allowed,omitempty"`
	Resolution string `json:"resolution,omitempty"`
}

// readResponse is the subset of the OpenFGA /read response we use.
// Rows that are missing any of the user/relation/object fields are
// silently dropped so a partial server reply does not poison the
// returned slice.
type readResponse struct {
	Tuples []struct {
		Key struct {
			User     string `json:"user,omitempty"`
			Relation string `json:"relation,omitempty"`
			Object   string `json:"object,omitempty"`
		} `json:"key"`
	} `json:"tuples,omitempty"`
	ContinuationToken string `json:"continuation_token,omitempty"`
}

func (p *provider) Check(ctx context.Context, subject acl.Subject, action acl.Action, resource acl.Resource) (acl.CheckResult, error) {
	relation, ok := actionToRelation[action]
	if !ok {
		return acl.Deny(fmt.Sprintf("openfga: unknown action %q", action)), nil
	}
	body := map[string]any{
		"tuple_key": map[string]string{
			"user":     EncodeSubject(subject),
			"relation": relation,
			"object":   EncodeResource(resource),
		},
	}
	if p.modelID != "" {
		body["authorization_model_id"] = p.modelID
	}
	var resp checkResponse
	if err := p.post(ctx, "/check", body, &resp); err != nil {
		return acl.CheckResult{}, err
	}
	if resp.Allowed != nil && *resp.Allowed {
		return acl.Allow(resp.Resolution), nil
	}
	reason := resp.Resolution
	if reason == "" {
		reason = "openfga: not allowed"
	}
	return acl.Deny(reason), nil
}

func (p *provider) Write(ctx context.Context, req acl.WriteTuplesRequest) error {
	if len(req.Writes) == 0 && len(req.Deletes) == 0 {
		// No work to do; skipping the round-trip avoids a needless
		// 400/422 from servers that reject empty bodies.
		return nil
	}
	body := make(map[string]any, 3)
	if len(req.Writes) > 0 {
		body["writes"] = map[string]any{"tuple_keys": tupleKeys(req.Writes)}
	}
	if len(req.Deletes) > 0 {
		body["deletes"] = map[string]any{"tuple_keys": tupleKeys(req.Deletes)}
	}
	if p.modelID != "" {
		body["authorization_model_id"] = p.modelID
	}
	var ignored map[string]any
	return p.post(ctx, "/write", body, &ignored)
}

func (p *provider) Read(ctx context.Context, query acl.ReadTuplesQuery) ([]acl.Tuple, error) {
	tupleKey := make(map[string]string, 3)
	if query.Subject != nil {
		tupleKey["user"] = EncodeSubject(*query.Subject)
	}
	if query.Relation != "" {
		tupleKey["relation"] = query.Relation
	}
	if query.Resource != nil {
		tupleKey["object"] = EncodeResource(*query.Resource)
	}
	body := map[string]any{"tuple_key": tupleKey}
	var resp readResponse
	if err := p.post(ctx, "/read", body, &resp); err != nil {
		return nil, err
	}
	out := make([]acl.Tuple, 0, len(resp.Tuples))
	for _, row := range resp.Tuples {
		k := row.Key
		if k.User == "" || k.Relation == "" || k.Object == "" {
			continue
		}
		subject, ok := DecodeSubject(k.User)
		if !ok {
			continue
		}
		resource, ok := DecodeResource(k.Object)
		if !ok {
			continue
		}
		out = append(out, acl.Tuple{Subject: subject, Relation: k.Relation, Resource: resource})
	}
	return out, nil
}

// post serialises body, dispatches a POST to storePath+endpoint, and
// decodes the response into out. All non-HTTP failures are wrapped in
// [*RequestError]; non-2xx responses become [*HTTPError].
func (p *provider) post(ctx context.Context, endpoint string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return &RequestError{Endpoint: endpoint, Cause: err}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.storePath+endpoint, bytes.NewReader(payload))
	if err != nil {
		return &RequestError{Endpoint: endpoint, Cause: err}
	}
	req.Header.Set("Content-Type", "application/json")
	if p.token != "" {
		req.Header.Set("Authorization", "Bearer "+p.token)
	}
	res, err := p.client.Do(req)
	if err != nil {
		return &RequestError{Endpoint: endpoint, Cause: err}
	}
	defer res.Body.Close()

	respBody, readErr := io.ReadAll(res.Body)
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		text := "<no body>"
		if readErr == nil {
			text = string(respBody)
		}
		return &HTTPError{Status: res.StatusCode, Endpoint: endpoint, Body: text}
	}
	if readErr != nil {
		return &RequestError{Endpoint: endpoint, Cause: readErr}
	}
	if len(respBody) == 0 {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return &RequestError{Endpoint: endpoint, Cause: err}
	}
	return nil
}

// tupleKeys converts a slice of [acl.Tuple] into the wire shape used
// by /write. We allocate a fresh slice rather than reuse the input so
// the caller can mutate their slice without disturbing in-flight
// payloads.
func tupleKeys(tuples []acl.Tuple) []map[string]string {
	out := make([]map[string]string, len(tuples))
	for i, t := range tuples {
		out[i] = map[string]string{
			"user":     EncodeSubject(t.Subject),
			"relation": t.Relation,
			"object":   EncodeResource(t.Resource),
		}
	}
	return out
}

// EncodeSubject returns the OpenFGA-compatible "kind:id" encoding.
func EncodeSubject(s acl.Subject) string { return string(s.Kind) + ":" + s.ID }

// EncodeResource returns the OpenFGA-compatible "type:id" encoding.
func EncodeResource(r acl.Resource) string { return string(r.Type) + ":" + r.ID }

// DecodeSubject reverses [EncodeSubject]. Returns ok=false for
// malformed input or unknown subject kinds, so a poisoned response
// never produces a Subject the rest of the system cannot reason about.
func DecodeSubject(s string) (acl.Subject, bool) {
	idx := strings.IndexByte(s, ':')
	if idx == -1 {
		return acl.Subject{}, false
	}
	kind := acl.SubjectKind(s[:idx])
	switch kind {
	case acl.SubjectUser, acl.SubjectAPIKey, acl.SubjectService:
		return acl.Subject{Kind: kind, ID: s[idx+1:]}, true
	}
	return acl.Subject{}, false
}

// DecodeResource reverses [EncodeResource].
func DecodeResource(s string) (acl.Resource, bool) {
	idx := strings.IndexByte(s, ':')
	if idx == -1 {
		return acl.Resource{}, false
	}
	rt := acl.ResourceType(s[:idx])
	switch rt {
	case acl.ResourceWorkspace, acl.ResourceBrain, acl.ResourceCollection, acl.ResourceDocument:
		return acl.Resource{Type: rt, ID: s[idx+1:]}, true
	}
	return acl.Resource{}, false
}

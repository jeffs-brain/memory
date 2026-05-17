// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ---- Mock dispatcher ----

type mockDispatcher struct {
	dispatched []ConnectorDocument
	failIDs    map[string]bool
	nextDocID  int
}

func newMockDispatcher() *mockDispatcher {
	return &mockDispatcher{failIDs: make(map[string]bool)}
}

func (m *mockDispatcher) Dispatch(doc ConnectorDocument) (string, error) {
	if m.failIDs[doc.ExternalID] {
		return "", fmt.Errorf("dispatch error for %s", doc.ExternalID)
	}
	m.dispatched = append(m.dispatched, doc)
	m.nextDocID++
	return fmt.Sprintf("doc-%d", m.nextDocID), nil
}

// ---- Helpers ----

func newTestReceiver(auth WebhookAuthConfig, dispatcher DocumentDispatcher) *WebhookReceiver {
	cfg := WebhookReceiverConfig{
		BrainID:           "test-brain",
		Auth:              auth,
		EnableIdempotency: true,
		Dispatcher:        dispatcher,
	}
	return NewWebhookReceiver(cfg)
}

func bearerAuth() WebhookAuthConfig {
	return WebhookAuthConfig{Method: AuthBearer, Secret: "test-secret-token"}
}

func hmacAuth() WebhookAuthConfig {
	return WebhookAuthConfig{Method: AuthHMAC, Secret: "hmac-secret-key"}
}

func makePayload(docs []WebhookDocument) []byte {
	payload := WebhookPayload{Documents: docs}
	data, _ := json.Marshal(payload)
	return data
}

func singleDoc() []WebhookDocument {
	return []WebhookDocument{
		{ExternalID: "ext-1", Content: "hello world"},
	}
}

func doRequest(receiver *WebhookReceiver, method string, body []byte, headers map[string]string) *httptest.ResponseRecorder {
	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}
	req := httptest.NewRequest(method, "/webhook/ingest", bodyReader)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	if _, ok := headers["Content-Type"]; !ok {
		req.Header.Set("Content-Type", "application/json")
	}
	rr := httptest.NewRecorder()
	receiver.ServeHTTP(rr, req)
	return rr
}

func computeHMAC(secret string, timestamp int64, body []byte) string {
	ts := strconv.FormatInt(timestamp, 10)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(ts))
	mac.Write([]byte("."))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func hmacHeaders(secret string, body []byte) map[string]string {
	ts := time.Now().Unix()
	return map[string]string{
		"Content-Type":        "application/json",
		"X-Webhook-Signature": computeHMAC(secret, ts, body),
		"X-Webhook-Timestamp": strconv.FormatInt(ts, 10),
	}
}

func decodeResponse(t *testing.T, rr *httptest.ResponseRecorder) WebhookResponse {
	t.Helper()
	var resp WebhookResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	return resp
}

// ---- Tests ----

func TestValidPayloadSingleDocument(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}
	if resp.Rejected != 0 {
		t.Errorf("expected rejected=0, got %d", resp.Rejected)
	}
	if len(d.dispatched) != 1 {
		t.Errorf("expected 1 dispatched doc, got %d", len(d.dispatched))
	}
}

func TestValidPayloadMultipleDocuments(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1", Content: "doc one"},
		{ExternalID: "ext-2", Content: "doc two"},
		{ExternalID: "ext-3", Content: "doc three"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 3 {
		t.Errorf("expected accepted=3, got %d", resp.Accepted)
	}
}

func TestPartialFailure(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1", Content: "valid"},
		{ExternalID: "", Content: "missing id"},
		{ExternalID: "ext-3", Content: "also valid"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 2 {
		t.Errorf("expected accepted=2, got %d", resp.Accepted)
	}
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestBearerAuthSuccess(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
}

func TestBearerAuthFailure(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer wrong-token",
	})
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rr.Code)
	}
	// Body must be empty to prevent information leakage.
	if rr.Body.Len() != 0 {
		t.Errorf("expected empty body on auth failure, got %d bytes", rr.Body.Len())
	}
}

func TestMissingAuthHeader(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{})
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rr.Code)
	}
}

func TestHMACAuthSuccess(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(hmacAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	headers := hmacHeaders("hmac-secret-key", body)
	rr := doRequest(w, http.MethodPost, body, headers)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHMACAuthFailure(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(hmacAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	headers := hmacHeaders("wrong-secret", body)
	rr := doRequest(w, http.MethodPost, body, headers)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rr.Code)
	}
}

func TestHMACMissingSignature(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(hmacAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"X-Webhook-Timestamp": strconv.FormatInt(time.Now().Unix(), 10),
	})
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rr.Code)
	}
}

func TestHMACTimestampExpiry(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(hmacAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	// Timestamp 10 minutes ago — should be rejected.
	oldTs := time.Now().Add(-10 * time.Minute).Unix()
	sig := computeHMAC("hmac-secret-key", oldTs, body)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"X-Webhook-Signature": sig,
		"X-Webhook-Timestamp": strconv.FormatInt(oldTs, 10),
	})
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for expired timestamp, got %d", rr.Code)
	}
}

func TestHMACMissingTimestamp(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(hmacAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"X-Webhook-Signature": "sha256=deadbeef",
	})
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for missing timestamp, got %d", rr.Code)
	}
}

func TestEmptyDocumentsArray(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload([]WebhookDocument{})
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestTooManyDocuments(t *testing.T) {
	d := newMockDispatcher()
	cfg := WebhookReceiverConfig{
		BrainID:      "test-brain",
		Auth:         bearerAuth(),
		MaxDocuments: 3,
		Dispatcher:   d,
	}
	w := NewWebhookReceiver(cfg)
	defer w.Close()

	docs := make([]WebhookDocument, 4)
	for i := range docs {
		docs[i] = WebhookDocument{
			ExternalID: fmt.Sprintf("ext-%d", i),
			Content:    "content",
		}
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestPayloadTooLarge(t *testing.T) {
	d := newMockDispatcher()
	cfg := WebhookReceiverConfig{
		BrainID:         "test-brain",
		Auth:            bearerAuth(),
		MaxPayloadBytes: 100,
		Dispatcher:      d,
	}
	w := NewWebhookReceiver(cfg)
	defer w.Close()

	bigContent := strings.Repeat("x", 200)
	body := makePayload([]WebhookDocument{
		{ExternalID: "ext-1", Content: bigContent},
	})
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d", rr.Code)
	}
}

func TestIndividualDocumentTooLarge(t *testing.T) {
	d := newMockDispatcher()
	cfg := WebhookReceiverConfig{
		BrainID:          "test-brain",
		Auth:             bearerAuth(),
		MaxDocumentBytes: 50,
		Dispatcher:       d,
	}
	w := NewWebhookReceiver(cfg)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "small", Content: "ok"},
		{ExternalID: "big", Content: strings.Repeat("x", 100)},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestBase64Encoding(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	originalContent := "binary content here"
	encoded := base64.StdEncoding.EncodeToString([]byte(originalContent))
	docs := []WebhookDocument{
		{ExternalID: "ext-b64", Content: encoded, Encoding: "base64", MIME: "application/octet-stream"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}
	if len(d.dispatched) != 1 {
		t.Fatalf("expected 1 dispatched, got %d", len(d.dispatched))
	}
	if string(d.dispatched[0].Content) != originalContent {
		t.Errorf("expected decoded content %q, got %q", originalContent, string(d.dispatched[0].Content))
	}
}

func TestInvalidBase64(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-bad64", Content: "not-valid-base64!!!", Encoding: "base64"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestWrongContentType(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
		"Content-Type":  "text/plain",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestNonPostMethod(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	rr := doRequest(w, http.MethodGet, nil, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", rr.Code)
	}
}

func TestIdempotencyKeyFirstRequest(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization":    "Bearer test-secret-token",
		"X-Idempotency-Key": "unique-key-123",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}
}

func TestIdempotencyKeyDuplicate(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	body := makePayload(singleDoc())
	headers := map[string]string{
		"Authorization":    "Bearer test-secret-token",
		"X-Idempotency-Key": "dup-key-456",
	}

	// First request
	rr1 := doRequest(w, http.MethodPost, body, headers)
	if rr1.Code != http.StatusOK {
		t.Fatalf("first request: expected 200, got %d", rr1.Code)
	}

	// Second request with same key — should return cached response.
	rr2 := doRequest(w, http.MethodPost, body, headers)
	if rr2.Code != http.StatusOK {
		t.Fatalf("second request: expected 200, got %d", rr2.Code)
	}

	// Dispatcher should only have received the document once.
	if len(d.dispatched) != 1 {
		t.Errorf("expected 1 dispatched (idempotent), got %d", len(d.dispatched))
	}
}

func TestMalformedJSON(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	rr := doRequest(w, http.MethodPost, []byte("{invalid"), map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestDocumentMissingExternalID(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{Content: "content but no id"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestDocumentMissingContent(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestDispatcherFailure(t *testing.T) {
	d := newMockDispatcher()
	d.failIDs["ext-fail"] = true
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-ok", Content: "good"},
		{ExternalID: "ext-fail", Content: "will fail"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

func TestDefaultMIME(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1", Content: "plain text"},
	}
	body := makePayload(docs)
	doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if len(d.dispatched) != 1 {
		t.Fatalf("expected 1 dispatched, got %d", len(d.dispatched))
	}
	if d.dispatched[0].MIME != "text/plain" {
		t.Errorf("expected default MIME text/plain, got %s", d.dispatched[0].MIME)
	}
}

func TestCustomMIME(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1", Content: "<html></html>", MIME: "text/html"},
	}
	body := makePayload(docs)
	doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if len(d.dispatched) != 1 {
		t.Fatalf("expected 1 dispatched, got %d", len(d.dispatched))
	}
	if d.dispatched[0].MIME != "text/html" {
		t.Errorf("expected MIME text/html, got %s", d.dispatched[0].MIME)
	}
}

func TestRateLimiting(t *testing.T) {
	d := newMockDispatcher()
	cfg := WebhookReceiverConfig{
		BrainID:    "test-brain",
		Auth:       bearerAuth(),
		Dispatcher: d,
	}
	w := NewWebhookReceiver(cfg)
	defer w.Close()

	// Drain all tokens from the bucket.
	for i := 0; i < defaultRateLimitTokens+10; i++ {
		w.limiter.tryAcquire()
	}

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", rr.Code)
	}
}

func TestMetadataPassthrough(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{
			ExternalID: "ext-meta",
			Content:    "content",
			Title:      "My Document",
			URL:        "https://example.com/doc",
			Metadata:   map[string]string{"author": "test", "source": "webhook"},
		},
	}
	body := makePayload(docs)
	doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})

	if len(d.dispatched) != 1 {
		t.Fatalf("expected 1 dispatched, got %d", len(d.dispatched))
	}
	doc := d.dispatched[0]
	if doc.Title != "My Document" {
		t.Errorf("expected title 'My Document', got %q", doc.Title)
	}
	if doc.URL != "https://example.com/doc" {
		t.Errorf("expected URL, got %q", doc.URL)
	}
	if doc.Metadata["author"] != "test" {
		t.Errorf("expected metadata author=test, got %q", doc.Metadata["author"])
	}
}

func TestNoDispatcherConfigured(t *testing.T) {
	w := newTestReceiver(bearerAuth(), nil)
	defer w.Close()

	body := makePayload(singleDoc())
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1 (no dispatcher), got %d", resp.Rejected)
	}
}

func TestTokenBucketRefill(t *testing.T) {
	tb := newTokenBucket(1, 100.0) // 1 token, refills 100/sec

	// Drain the bucket.
	if !tb.tryAcquire() {
		t.Fatal("first acquire should succeed")
	}
	if tb.tryAcquire() {
		t.Fatal("second acquire should fail (bucket empty)")
	}

	// Wait for refill.
	time.Sleep(50 * time.Millisecond)

	if !tb.tryAcquire() {
		t.Fatal("acquire after refill should succeed")
	}
}

func TestIdempotencyStoreExpiry(t *testing.T) {
	store := newIdempotencyStore()
	store.set("key-1", WebhookResponse{Accepted: 1}, 10*time.Millisecond)

	// Should be available immediately.
	resp, ok := store.get("key-1")
	if !ok {
		t.Fatal("expected key to exist")
	}
	if resp.Accepted != 1 {
		t.Errorf("expected accepted=1, got %d", resp.Accepted)
	}

	// Wait for expiry.
	time.Sleep(20 * time.Millisecond)

	_, ok = store.get("key-1")
	if ok {
		t.Fatal("expected key to be expired")
	}
}

func TestIdempotencyStoreSweep(t *testing.T) {
	store := newIdempotencyStore()
	store.set("expired", WebhookResponse{Accepted: 1}, 1*time.Millisecond)
	store.set("valid", WebhookResponse{Accepted: 2}, 1*time.Hour)

	time.Sleep(5 * time.Millisecond)
	store.sweep()

	store.mu.RLock()
	defer store.mu.RUnlock()
	if _, ok := store.entries["expired"]; ok {
		t.Error("expected expired key to be swept")
	}
	if _, ok := store.entries["valid"]; !ok {
		t.Error("expected valid key to remain after sweep")
	}
}

func TestUnsupportedEncoding(t *testing.T) {
	d := newMockDispatcher()
	w := newTestReceiver(bearerAuth(), d)
	defer w.Close()

	docs := []WebhookDocument{
		{ExternalID: "ext-1", Content: "data", Encoding: "gzip"},
	}
	body := makePayload(docs)
	rr := doRequest(w, http.MethodPost, body, map[string]string{
		"Authorization": "Bearer test-secret-token",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	resp := decodeResponse(t, rr)
	if resp.Rejected != 1 {
		t.Errorf("expected rejected=1, got %d", resp.Rejected)
	}
}

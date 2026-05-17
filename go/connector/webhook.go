// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- Configuration types ----

// WebhookAuthMethod enumerates the supported authentication strategies.
type WebhookAuthMethod string

const (
	// AuthBearer uses a shared bearer token in the Authorization header.
	AuthBearer WebhookAuthMethod = "bearer"
	// AuthHMAC uses HMAC-SHA256 signature verification via X-Webhook-Signature.
	AuthHMAC WebhookAuthMethod = "hmac"
)

// WebhookAuthConfig holds the authentication configuration for a webhook endpoint.
type WebhookAuthConfig struct {
	Method WebhookAuthMethod `json:"method"`
	Secret string            `json:"secret"`
}

// WebhookReceiverConfig configures a [WebhookReceiver] instance.
type WebhookReceiverConfig struct {
	BrainID           string            `json:"brainId"`
	Auth              WebhookAuthConfig `json:"auth"`
	MaxDocuments      int               `json:"maxDocuments,omitempty"`
	MaxPayloadBytes   int64             `json:"maxPayloadBytes,omitempty"`
	MaxDocumentBytes  int64             `json:"maxDocumentBytes,omitempty"`
	EnableIdempotency bool              `json:"enableIdempotency,omitempty"`
	TimestampExpiry   time.Duration     `json:"timestampExpiry,omitempty"`
	Dispatcher        DocumentDispatcher
	Logger            *slog.Logger
}

// ---- Payload types ----

// WebhookPayload is the JSON body of a webhook request.
type WebhookPayload struct {
	Documents []WebhookDocument `json:"documents"`
}

// WebhookDocument represents a single document within the webhook payload.
type WebhookDocument struct {
	ExternalID string            `json:"externalId"`
	Content    string            `json:"content"`
	Encoding   string            `json:"encoding,omitempty"`
	MIME       string            `json:"mime,omitempty"`
	Title      string            `json:"title,omitempty"`
	URL        string            `json:"url,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// WebhookResponse is the JSON response returned by the webhook handler.
type WebhookResponse struct {
	Accepted int                     `json:"accepted"`
	Rejected int                     `json:"rejected"`
	Results  []WebhookDocumentResult `json:"results"`
}

// WebhookDocumentResult reports the outcome for a single document.
type WebhookDocumentResult struct {
	ExternalID string `json:"externalId"`
	Status     string `json:"status"`
	DocumentID string `json:"documentId,omitempty"`
	Error      string `json:"error,omitempty"`
}

// ---- Rate limiter ----

// tokenBucket implements a simple token bucket rate limiter.
type tokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func newTokenBucket(maxTokens int, refillRate float64) *tokenBucket {
	return &tokenBucket{
		tokens:     float64(maxTokens),
		maxTokens:  float64(maxTokens),
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

func (tb *tokenBucket) tryAcquire() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	tb.refill()
	if tb.tokens < 1 {
		return false
	}
	tb.tokens--
	return true
}

func (tb *tokenBucket) refill() {
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens = math.Min(tb.maxTokens, tb.tokens+elapsed*tb.refillRate)
	tb.lastRefill = now
}

// ---- Idempotency store ----

// idempotencyEntry stores a cached response for replay protection.
type idempotencyEntry struct {
	response  WebhookResponse
	expiresAt time.Time
}

// idempotencyStore is a simple in-memory store for idempotency keys.
type idempotencyStore struct {
	mu      sync.RWMutex
	entries map[string]idempotencyEntry
}

func newIdempotencyStore() *idempotencyStore {
	return &idempotencyStore{
		entries: make(map[string]idempotencyEntry),
	}
}

func (s *idempotencyStore) get(key string) (WebhookResponse, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	entry, ok := s.entries[key]
	if !ok {
		return WebhookResponse{}, false
	}
	if time.Now().After(entry.expiresAt) {
		return WebhookResponse{}, false
	}
	return entry.response, true
}

func (s *idempotencyStore) set(key string, resp WebhookResponse, ttl time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[key] = idempotencyEntry{
		response:  resp,
		expiresAt: time.Now().Add(ttl),
	}
}

// sweep removes expired entries. Called periodically to prevent unbounded growth.
func (s *idempotencyStore) sweep() {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now()
	for key, entry := range s.entries {
		if now.After(entry.expiresAt) {
			delete(s.entries, key)
		}
	}
}

// ---- WebhookReceiver ----

const (
	defaultMaxDocuments    = 50
	defaultMaxPayloadBytes = 8 * 1024 * 1024  // 8 MiB
	defaultMaxDocBytes     = 10 * 1024 * 1024  // 10 MiB
	defaultTimestampExpiry = 5 * time.Minute
	defaultRateLimitTokens = 60
	defaultRateLimitRate   = 10.0 // tokens per second
	idempotencyTTL         = 24 * time.Hour
	sweepInterval          = 1 * time.Hour
)

// WebhookReceiver is an HTTP handler that accepts POST requests containing
// document payloads, validates authentication and schema, and dispatches
// accepted documents to the ingestion pipeline.
type WebhookReceiver struct {
	brainID           string
	auth              WebhookAuthConfig
	maxDocuments      int
	maxPayloadBytes   int64
	maxDocumentBytes  int64
	enableIdempotency bool
	timestampExpiry   time.Duration
	dispatcher        DocumentDispatcher
	log               *slog.Logger
	limiter           *tokenBucket
	idempotency       *idempotencyStore
	sweepDone         chan struct{}
}

// NewWebhookReceiver creates a new [WebhookReceiver] from the given configuration.
func NewWebhookReceiver(cfg WebhookReceiverConfig) *WebhookReceiver {
	maxDocs := cfg.MaxDocuments
	if maxDocs <= 0 {
		maxDocs = defaultMaxDocuments
	}
	maxPayload := cfg.MaxPayloadBytes
	if maxPayload <= 0 {
		maxPayload = defaultMaxPayloadBytes
	}
	maxDoc := cfg.MaxDocumentBytes
	if maxDoc <= 0 {
		maxDoc = defaultMaxDocBytes
	}
	expiry := cfg.TimestampExpiry
	if expiry <= 0 {
		expiry = defaultTimestampExpiry
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}

	w := &WebhookReceiver{
		brainID:           cfg.BrainID,
		auth:              cfg.Auth,
		maxDocuments:      maxDocs,
		maxPayloadBytes:   maxPayload,
		maxDocumentBytes:  maxDoc,
		enableIdempotency: cfg.EnableIdempotency,
		timestampExpiry:   expiry,
		dispatcher:        cfg.Dispatcher,
		log:               log,
		limiter:           newTokenBucket(defaultRateLimitTokens, defaultRateLimitRate),
		idempotency:       newIdempotencyStore(),
		sweepDone:         make(chan struct{}),
	}

	go w.sweepLoop()
	return w
}

// Close stops the background sweep goroutine.
func (w *WebhookReceiver) Close() {
	close(w.sweepDone)
}

func (w *WebhookReceiver) sweepLoop() {
	ticker := time.NewTicker(sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			w.idempotency.sweep()
		case <-w.sweepDone:
			return
		}
	}
}

// ServeHTTP implements [http.Handler].
func (w *WebhookReceiver) ServeHTTP(rw http.ResponseWriter, r *http.Request) {
	// Method check
	if r.Method != http.MethodPost {
		w.writeError(rw, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	// Rate limiting
	if !w.limiter.tryAcquire() {
		w.writeError(rw, http.StatusTooManyRequests, "rate limit exceeded")
		return
	}

	// Content-Type check
	ct := r.Header.Get("Content-Type")
	if !strings.HasPrefix(strings.ToLower(ct), "application/json") {
		w.writeError(rw, http.StatusBadRequest, "content-type must be application/json")
		return
	}

	// Read body with size limit
	body, err := readLimitedBody(r, w.maxPayloadBytes)
	if err != nil {
		w.writeError(rw, http.StatusRequestEntityTooLarge, "payload too large")
		return
	}

	// Authentication
	if !w.authenticate(r, body) {
		// Return empty body on auth failure to prevent information leakage.
		rw.WriteHeader(http.StatusUnauthorized)
		return
	}

	// Idempotency check
	idempotencyKey := r.Header.Get("X-Idempotency-Key")
	if w.enableIdempotency && idempotencyKey != "" {
		if cached, ok := w.idempotency.get(idempotencyKey); ok {
			w.log.Info("webhook idempotency hit", "key", idempotencyKey)
			w.writeJSON(rw, http.StatusOK, cached)
			return
		}
	}

	// Parse payload
	var payload WebhookPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		w.writeError(rw, http.StatusBadRequest, "invalid json")
		return
	}

	// Validate document count
	if len(payload.Documents) == 0 {
		w.writeError(rw, http.StatusBadRequest, "documents array is empty")
		return
	}
	if len(payload.Documents) > w.maxDocuments {
		w.writeError(rw, http.StatusBadRequest,
			fmt.Sprintf("exceeds maximum of %d documents", w.maxDocuments))
		return
	}

	// Process each document
	resp := w.processDocuments(payload.Documents)

	// Cache response for idempotency
	if w.enableIdempotency && idempotencyKey != "" {
		w.idempotency.set(idempotencyKey, resp, idempotencyTTL)
	}

	w.writeJSON(rw, http.StatusOK, resp)
}

// authenticate verifies the request using the configured authentication method.
func (w *WebhookReceiver) authenticate(r *http.Request, body []byte) bool {
	authenticators := map[WebhookAuthMethod]func(*http.Request, []byte) bool{
		AuthBearer: w.authenticateBearer,
		AuthHMAC:   w.authenticateHMAC,
	}
	fn, ok := authenticators[w.auth.Method]
	if !ok {
		w.log.Error("unsupported auth method", "method", w.auth.Method)
		return false
	}
	return fn(r, body)
}

func (w *WebhookReceiver) authenticateBearer(r *http.Request, _ []byte) bool {
	header := r.Header.Get("Authorization")
	if header == "" {
		return false
	}
	const prefix = "Bearer "
	if len(header) < len(prefix) {
		return false
	}
	if !strings.EqualFold(header[:len(prefix)-1], "bearer") {
		return false
	}
	actual := strings.TrimSpace(header[len(prefix)-1:])
	if len(actual) == 0 || actual[0] == ' ' {
		actual = strings.TrimSpace(actual)
	}
	if actual == "" {
		return false
	}
	// Timing-safe comparison via SHA-256 hash to prevent length-based leaks.
	actualHash := sha256.Sum256([]byte(actual))
	expectedHash := sha256.Sum256([]byte(w.auth.Secret))
	return subtle.ConstantTimeCompare(actualHash[:], expectedHash[:]) == 1
}

func (w *WebhookReceiver) authenticateHMAC(r *http.Request, body []byte) bool {
	sigHeader := r.Header.Get("X-Webhook-Signature")
	if sigHeader == "" {
		return false
	}

	// Verify timestamp expiry for replay protection.
	tsHeader := r.Header.Get("X-Webhook-Timestamp")
	if tsHeader == "" {
		return false
	}
	ts, err := strconv.ParseInt(tsHeader, 10, 64)
	if err != nil {
		return false
	}
	requestTime := time.Unix(ts, 0)
	if time.Since(requestTime).Abs() > w.timestampExpiry {
		return false
	}

	// Parse signature: expect "sha256=<hex>"
	const sigPrefix = "sha256="
	if !strings.HasPrefix(sigHeader, sigPrefix) {
		return false
	}
	providedSig, err := hex.DecodeString(sigHeader[len(sigPrefix):])
	if err != nil {
		return false
	}

	// Compute expected HMAC-SHA256 over timestamp + "." + body to bind
	// the signature to both the timestamp and the payload.
	mac := hmac.New(sha256.New, []byte(w.auth.Secret))
	mac.Write([]byte(tsHeader))
	mac.Write([]byte("."))
	mac.Write(body)
	expectedSig := mac.Sum(nil)

	return hmac.Equal(providedSig, expectedSig)
}

// processDocuments validates and dispatches each document, collecting results.
func (w *WebhookReceiver) processDocuments(docs []WebhookDocument) WebhookResponse {
	results := make([]WebhookDocumentResult, 0, len(docs))
	accepted := 0
	rejected := 0

	for _, doc := range docs {
		result := w.processDocument(doc)
		statusCounts := map[string]*int{
			"accepted": &accepted,
			"rejected": &rejected,
		}
		if counter, ok := statusCounts[result.Status]; ok {
			*counter++
		}
		results = append(results, result)
	}

	return WebhookResponse{
		Accepted: accepted,
		Rejected: rejected,
		Results:  results,
	}
}

func (w *WebhookReceiver) processDocument(doc WebhookDocument) WebhookDocumentResult {
	// Validate required fields.
	if doc.ExternalID == "" {
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      "externalId is required",
		}
	}
	if doc.Content == "" {
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      "content is required",
		}
	}

	// Decode content.
	content, err := w.decodeContent(doc)
	if err != nil {
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      err.Error(),
		}
	}

	// Check individual document size.
	if int64(len(content)) > w.maxDocumentBytes {
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      fmt.Sprintf("document exceeds maximum size of %d bytes", w.maxDocumentBytes),
		}
	}

	// Build connector document.
	mime := doc.MIME
	if mime == "" {
		mime = "text/plain"
	}
	connDoc := ConnectorDocument{
		ExternalID: doc.ExternalID,
		Content:    content,
		MIME:       mime,
		Title:      doc.Title,
		URL:        doc.URL,
		Metadata:   doc.Metadata,
		ModifiedAt: time.Now(),
	}

	// Dispatch to pipeline.
	if w.dispatcher == nil {
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      "no dispatcher configured",
		}
	}

	docID, err := w.dispatcher.Dispatch(connDoc)
	if err != nil {
		w.log.Error("webhook dispatch failed",
			"externalId", doc.ExternalID,
			"err", err,
		)
		return WebhookDocumentResult{
			ExternalID: doc.ExternalID,
			Status:     "rejected",
			Error:      "dispatch failed",
		}
	}

	return WebhookDocumentResult{
		ExternalID: doc.ExternalID,
		Status:     "accepted",
		DocumentID: docID,
	}
}

func (w *WebhookReceiver) decodeContent(doc WebhookDocument) ([]byte, error) {
	decoders := map[string]func(string) ([]byte, error){
		"":      func(s string) ([]byte, error) { return []byte(s), nil },
		"utf8":  func(s string) ([]byte, error) { return []byte(s), nil },
		"base64": func(s string) ([]byte, error) {
			decoded, err := base64.StdEncoding.DecodeString(s)
			if err != nil {
				return nil, fmt.Errorf("invalid base64 content: %w", err)
			}
			return decoded, nil
		},
	}
	decoder, ok := decoders[doc.Encoding]
	if !ok {
		return nil, fmt.Errorf("unsupported encoding: %s", doc.Encoding)
	}
	return decoder(doc.Content)
}

// readLimitedBody reads the full request body up to limit bytes.
// Returns an error when the body exceeds the limit.
func readLimitedBody(r *http.Request, limit int64) ([]byte, error) {
	if r.Body == nil {
		return nil, fmt.Errorf("empty body")
	}
	defer func() { _ = r.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("payload exceeds %d bytes", limit)
	}
	return data, nil
}

func (w *WebhookReceiver) writeJSON(rw http.ResponseWriter, status int, v any) {
	rw.Header().Set("Content-Type", "application/json")
	rw.WriteHeader(status)
	if err := json.NewEncoder(rw).Encode(v); err != nil {
		w.log.Error("failed to encode webhook response", "err", err)
	}
}

func (w *WebhookReceiver) writeError(rw http.ResponseWriter, status int, detail string) {
	rw.Header().Set("Content-Type", "application/problem+json")
	rw.WriteHeader(status)
	_ = json.NewEncoder(rw).Encode(map[string]any{
		"status": status,
		"title":  http.StatusText(status),
		"detail": detail,
	})
}

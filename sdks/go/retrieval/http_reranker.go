// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"
)

// HTTPRerankerDefaultTimeout is the upper bound for a single /rerank
// round trip. The bge-reranker-v2-m3 served by llama-server answers
// in well under 100 ms p95 on a 4090; five seconds absorbs a cold-load
// stall without blocking interactive search for long.
const HTTPRerankerDefaultTimeout = 5 * time.Second

// HTTPRerankerDefaultMaxDocChars caps the per-document payload shipped
// to the cross-encoder. bge-reranker-v2-m3 tokenises at roughly four
// characters per token and is trained for 512-token input pairs; 1200
// chars leaves enough headroom for the query + separator tokens
// without risking truncation on the server side.
const HTTPRerankerDefaultMaxDocChars = 1200

// HTTPRerankerConfig configures [NewHTTPReranker].
type HTTPRerankerConfig struct {
	// Endpoint is the fully qualified /rerank URL. When the caller
	// supplies only a base URL (no path), NewHTTPReranker appends
	// "/v1/rerank" to match the llama-server convention. An empty
	// Endpoint is rejected at construction time.
	Endpoint string
	// APIKey, when non-empty, is forwarded as "Authorization: Bearer
	// <key>". Leave empty for local llama-server deployments.
	APIKey string
	// Model is sent in the request body. The llama-server /v1/rerank
	// endpoint requires an exact match against the model alias; other
	// providers (Cohere, Jina) ignore this field for single-model
	// endpoints. NewHTTPReranker defaults to "bge-reranker-v2-m3" when
	// empty so local deployments work out of the box.
	Model string
	// Timeout bounds a single /rerank call. When zero or negative,
	// HTTPRerankerDefaultTimeout applies.
	Timeout time.Duration
	// MaxDocChars caps the per-candidate document payload before it
	// hits the wire. When zero or negative,
	// HTTPRerankerDefaultMaxDocChars applies.
	MaxDocChars int
	// Client overrides the default HTTP client. Useful for tests that
	// want to pin a custom transport or round-tripper.
	Client *http.Client
	// Logger receives warnings when the server returns malformed JSON
	// or a non-200 status. Defaults to slog.Default when nil.
	Logger *slog.Logger
}

// HTTPReranker calls an external cross-encoder HTTP service
// (bge-reranker, cohere rerank, jina reranker) that returns a score
// for every (query, passage) pair in a single request. Matches the
// contract of jeff's cross_encoder.go (runCrossEncoderRerank at lines
// 85-189) so the retrieval trace looks identical across SDKs.
//
// Failures degrade gracefully: a transport error, non-200 status, or
// malformed response returns the input slice unchanged with the error
// logged. The retrieval pipeline is responsible for flagging the
// failure on the trace via RerankSkipReason.
type HTTPReranker struct {
	endpoint    string
	apiKey      string
	model       string
	timeout     time.Duration
	maxDocChars int
	client      *http.Client
	logger      *slog.Logger
}

// NewHTTPReranker builds an [HTTPReranker] from the supplied config.
// Returns an error when Endpoint is blank because the reranker is
// useless without a target.
func NewHTTPReranker(cfg HTTPRerankerConfig) (*HTTPReranker, error) {
	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		return nil, errors.New("retrieval: HTTPRerankerConfig.Endpoint is required")
	}
	endpoint = normaliseRerankEndpoint(endpoint)
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = HTTPRerankerDefaultTimeout
	}
	maxDoc := cfg.MaxDocChars
	if maxDoc <= 0 {
		maxDoc = HTTPRerankerDefaultMaxDocChars
	}
	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = "bge-reranker-v2-m3"
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}
	return &HTTPReranker{
		endpoint:    endpoint,
		apiKey:      cfg.APIKey,
		model:       model,
		timeout:     timeout,
		maxDocChars: maxDoc,
		client:      client,
		logger:      cfg.Logger,
	}, nil
}

// normaliseRerankEndpoint appends "/v1/rerank" when the caller passed
// only a scheme + host + optional port. Callers who already know the
// exact path (e.g. "/rerank" for Cohere) keep their spelling intact.
func normaliseRerankEndpoint(endpoint string) string {
	trimmed := strings.TrimRight(endpoint, "/")
	// Parse the path portion by stripping scheme + host.
	slash := strings.Index(trimmed, "://")
	tail := trimmed
	if slash >= 0 {
		tail = trimmed[slash+3:]
	}
	if idx := strings.IndexByte(tail, '/'); idx >= 0 {
		// Path present; assume the caller knows what they want.
		return trimmed
	}
	return trimmed + "/v1/rerank"
}

// httpRerankRequest is the JSON body posted to the cross-encoder. The
// shape matches llama-server's /v1/rerank and is compatible with
// Cohere and Jina's rerank endpoints modulo the "top_n" convention.
type httpRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// httpRerankResponse mirrors the cross-encoder reply. The "results"
// array carries one entry per input document with a 0-indexed reference
// back to the request slice.
type httpRerankResponse struct {
	Results []httpRerankHit `json:"results"`
}

type httpRerankHit struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// Rerank implements [Reranker]. Builds a single /rerank call, parses
// the response, and re-orders the candidate slice by relevance score
// descending. Any error short-circuits back to the input order so the
// caller's pipeline keeps working; the error is returned to the
// retrieval layer which records it on the trace.
func (r *HTTPReranker) Rerank(ctx context.Context, query string, candidates []RetrievedChunk) ([]RetrievedChunk, error) {
	if r == nil {
		return candidates, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return candidates, nil
	}

	documents := make([]string, len(candidates))
	for i, c := range candidates {
		doc := composeHTTPRerankDoc(c)
		if len(doc) > r.maxDocChars {
			doc = doc[:r.maxDocChars]
		}
		documents[i] = doc
	}

	payload, err := json.Marshal(httpRerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: documents,
		TopN:      len(documents),
	})
	if err != nil {
		return nil, fmt.Errorf("retrieval: HTTPReranker marshal: %w", err)
	}

	callCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(callCtx, http.MethodPost, r.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("retrieval: HTTPReranker build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("retrieval: HTTPReranker request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("retrieval: HTTPReranker server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed httpRerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("retrieval: HTTPReranker decode: %w", err)
	}

	if len(parsed.Results) == 0 {
		// The server answered but returned nothing usable. The caller's
		// retrieval trace records this via the failure path; we
		// surface a typed error so the decision is explicit rather
		// than a silent identity reorder.
		return nil, errors.New("retrieval: HTTPReranker returned no scored documents")
	}

	type scoredPair struct {
		chunk RetrievedChunk
		score float64
		order int
	}
	pairs := make([]scoredPair, len(candidates))
	for i, c := range candidates {
		pairs[i] = scoredPair{chunk: c, order: i}
	}
	for _, hit := range parsed.Results {
		if hit.Index < 0 || hit.Index >= len(candidates) {
			continue
		}
		pairs[hit.Index].score = hit.RelevanceScore
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		if pairs[i].score != pairs[j].score {
			return pairs[i].score > pairs[j].score
		}
		if pairs[i].chunk.Score != pairs[j].chunk.Score {
			return pairs[i].chunk.Score > pairs[j].chunk.Score
		}
		return pairs[i].order < pairs[j].order
	})

	out := make([]RetrievedChunk, len(pairs))
	for i, p := range pairs {
		chunk := p.chunk
		chunk.RerankScore = p.score
		out[i] = chunk
	}
	return out, nil
}

// Name implements the namedReranker interface so the retrieval trace
// can attribute a rerank pass to the configured model. Falls through
// to a generic label when no model is set so the name is never empty.
func (r *HTTPReranker) Name() string {
	if r == nil || r.model == "" {
		return "http-reranker"
	}
	return "http:" + r.model
}

// composeHTTPRerankDoc builds the per-candidate document sent in the
// request. The preference order is title + summary joined by a newline
// (matching jeff's composeCrossEncoderDoc at lines 195-231), falling
// back to the raw text when both are empty. Returns the path as a last
// resort so the cross-encoder always has something to score.
func composeHTTPRerankDoc(c RetrievedChunk) string {
	title := strings.TrimSpace(c.Title)
	summary := strings.TrimSpace(c.Summary)
	var b strings.Builder
	if title != "" {
		b.WriteString(title)
	}
	if summary != "" {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(summary)
	}
	if b.Len() > 0 {
		return b.String()
	}
	body := strings.Join(strings.Fields(c.Text), " ")
	if body != "" {
		return body
	}
	return c.Path
}

// compile-time interface check.
var _ Reranker = (*HTTPReranker)(nil)

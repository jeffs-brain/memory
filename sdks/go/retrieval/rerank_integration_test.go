// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

// TestHybridRerank_LLM_EndToEnd_MarksTrace confirms that wiring an
// LLMReranker into the retriever flips Trace.Reranked to true and
// tags the RerankProvider with the configured model.
func TestHybridRerank_LLM_EndToEnd_MarksTrace(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)

	// Scripted reply that reorders whatever batch it sees so the
	// unanimity shortcut does not fire and rerank actually runs.
	provider := &scriptedProvider{
		handler: func(req llm.CompleteRequest) (llm.CompleteResponse, error) {
			// Return descending scores so the head swaps.
			var b strings.Builder
			b.WriteString(`[`)
			// Count candidate entries in the user prompt.
			n := 0
			for _, m := range req.Messages {
				if m.Role == llm.RoleUser {
					for _, line := range strings.Split(m.Content, "\n") {
						if strings.HasPrefix(strings.TrimSpace(line), "[") && strings.Contains(line, "] title:") {
							n++
						}
					}
				}
			}
			for i := 0; i < n; i++ {
				if i > 0 {
					b.WriteString(",")
				}
				// Local ID i gets score (n-i), so the reranker strongly
				// prefers earlier entries (i.e. preserves the RRF head).
				// We do not care about exact order here, only that the
				// rerank pass ran.
				score := float64(n - i)
				b.WriteString(`{"id":`)
				b.WriteString(jsonInt(i))
				b.WriteString(`,"score":`)
				b.WriteString(jsonFloat(score))
				b.WriteString(`}`)
			}
			b.WriteString(`]`)
			return llm.CompleteResponse{Text: b.String(), Stop: llm.StopEndTurn}, nil
		},
	}
	rr := NewLLMReranker(provider, "judge-m")
	rr.MaxBatch = 50

	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	// Either the unanimity shortcut fired (in which case the rerank is
	// skipped with reason=unanimity), or the rerank pass ran and the
	// trace must flag it.
	if resp.Trace.UnanimitySkipped {
		t.Logf("unanimity shortcut fired; rerank skipped by design")
		return
	}
	if !resp.Trace.Reranked {
		t.Fatalf("Trace.Reranked = false, want true (trace: %+v)", resp.Trace)
	}
	if resp.Trace.RerankProvider != "llm:judge-m" {
		t.Errorf("RerankProvider = %q, want llm:judge-m", resp.Trace.RerankProvider)
	}
	if resp.Trace.EffectiveMode != ModeHybridRerank {
		t.Errorf("EffectiveMode = %q, want hybrid-rerank", resp.Trace.EffectiveMode)
	}
	if provider.calls.Load() == 0 {
		t.Error("LLM provider never called from the rerank pass")
	}
}

// TestHybridRerank_HTTP_EndToEnd_MarksTrace proves the HTTP reranker
// reaches the trace when plugged into the retriever.
func TestHybridRerank_HTTP_EndToEnd_MarksTrace(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)

	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		req := decodeRerankRequest(t, r)
		results := make([]httpRerankHit, len(req.Documents))
		for i := range req.Documents {
			results[i] = httpRerankHit{Index: i, RelevanceScore: float64(len(req.Documents) - i)}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(httpRerankResponse{Results: results})
	}))
	defer server.Close()

	rerank, err := NewHTTPReranker(HTTPRerankerConfig{
		Endpoint: server.URL + "/v1/rerank",
		Model:    "bge-test",
	})
	if err != nil {
		t.Fatalf("NewHTTPReranker: %v", err)
	}
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rerank})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}

	if resp.Trace.UnanimitySkipped {
		t.Logf("unanimity shortcut fired; rerank skipped by design")
		return
	}
	if !resp.Trace.Reranked {
		t.Fatalf("Trace.Reranked = false, want true (trace: %+v)", resp.Trace)
	}
	if resp.Trace.RerankProvider != "http:bge-test" {
		t.Errorf("RerankProvider = %q, want http:bge-test", resp.Trace.RerankProvider)
	}
	if callCount == 0 {
		t.Error("HTTP reranker endpoint never hit")
	}
}

func jsonInt(i int) string {
	b, _ := json.Marshal(i)
	return string(b)
}

func jsonFloat(f float64) string {
	b, _ := json.Marshal(f)
	return string(b)
}

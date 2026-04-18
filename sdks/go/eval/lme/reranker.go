// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
)

const (
	rerankDefaultBaseURL = "http://localhost:8012"
	rerankDefaultTopN    = 50
	rerankTimeout        = 30 * time.Second
	rerankModel          = "bge-reranker-v2-m3"
)

// RerankerConfig controls the cross-encoder reranker.
type RerankerConfig struct {
	BaseURL string
	TopN    int
}

// RerankResult is a single scored document.
type RerankResult struct {
	Index int
	Score float64
	Text  string
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResponse struct {
	Results []rerankHit `json:"results"`
}

type rerankHit struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// Rerank sends documents to the cross-encoder and returns them sorted by
// relevance score (highest first).
func Rerank(ctx context.Context, cfg RerankerConfig, query string, documents []string) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = rerankDefaultBaseURL
	}
	topN := cfg.TopN
	if topN <= 0 {
		topN = rerankDefaultTopN
	}

	body, err := json.Marshal(rerankRequest{
		Model:     rerankModel,
		Query:     query,
		Documents: documents,
		TopN:      topN,
	})
	if err != nil {
		return nil, fmt.Errorf("rerank: marshal request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, rerankTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.BaseURL+"/v1/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("rerank: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("rerank: server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var parsed rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("rerank: decode response: %w", err)
	}

	results := make([]RerankResult, 0, len(parsed.Results))
	for _, hit := range parsed.Results {
		if hit.Index < 0 || hit.Index >= len(documents) {
			continue
		}
		results = append(results, RerankResult{
			Index: hit.Index,
			Score: hit.RelevanceScore,
			Text:  documents[hit.Index],
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

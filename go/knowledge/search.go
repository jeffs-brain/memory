// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

// defaultMaxResults caps the returned hit count when the caller does
// not set [SearchRequest.MaxResults].
const defaultMaxResults = 10

// defaultCandidateK is the per-retriever slate size requested when the
// hybrid retriever is available. Matches jeff's defaultHybridCandidateK
// so the behaviour carries across SDKs.
const defaultCandidateK = 50

// Search implements [Base].
//
// Routing:
//
//   - SearchHybrid (and Auto when a retriever is bound): delegate to
//     the [retrieval.Retriever]. A retriever error or zero hits falls
//     back to BM25 so the caller never sees a silent dead end.
//   - SearchBM25 (and Auto without a retriever): delegate to the bound
//     [*search.Index]. When no index is bound the request falls
//     through to the in-memory scan.
func (k *kbase) Search(ctx context.Context, req SearchRequest) (SearchResponse, error) {
	start := time.Now()
	if err := k.requireStore(); err != nil {
		return SearchResponse{}, err
	}
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return SearchResponse{Elapsed: time.Since(start)}, nil
	}
	maxResults := req.MaxResults
	if maxResults <= 0 {
		maxResults = defaultMaxResults
	}

	idx, retriever, _ := k.snapshot()
	mode := req.Mode
	if mode == SearchAuto {
		if retriever != nil {
			mode = SearchHybrid
		} else {
			mode = SearchBM25
		}
	}

	switch mode {
	case SearchHybrid:
		if retriever != nil {
			hits, trace, err := k.runHybrid(ctx, retriever, query, maxResults)
			if err == nil && len(hits) > 0 {
				return SearchResponse{
					Hits:    hits,
					Mode:    string(trace.EffectiveMode),
					Elapsed: time.Since(start),
				}, nil
			}
			// Graceful fallback: retriever absent, errored, or empty.
			bm25, bmErr := k.runBM25(ctx, idx, query, maxResults)
			if bmErr != nil {
				return SearchResponse{}, bmErr
			}
			return SearchResponse{
				Hits:     bm25,
				Mode:     "bm25",
				Elapsed:  time.Since(start),
				FellBack: true,
			}, nil
		}
		// No retriever: fall through to BM25 without marking as a
		// fallback, because the caller explicitly asked for hybrid
		// and got the next-best available signal.
		fallthrough
	case SearchBM25:
		hits, err := k.runBM25(ctx, idx, query, maxResults)
		if err != nil {
			return SearchResponse{}, err
		}
		return SearchResponse{
			Hits:    hits,
			Mode:    "bm25",
			Elapsed: time.Since(start),
		}, nil
	}
	return SearchResponse{Elapsed: time.Since(start)}, nil
}

// runHybrid delegates to the bound retriever.
func (k *kbase) runHybrid(ctx context.Context, r retrieval.Retriever, q string, maxResults int) ([]SearchHit, retrieval.Trace, error) {
	resp, err := r.Retrieve(ctx, retrieval.Request{
		Query:      q,
		TopK:       maxResults,
		Mode:       retrieval.ModeAuto,
		CandidateK: defaultCandidateK,
		BrainID:    k.brainID,
	})
	if err != nil {
		return nil, retrieval.Trace{}, err
	}
	hits := make([]SearchHit, 0, len(resp.Chunks))
	for _, res := range resp.Chunks {
		hits = append(hits, SearchHit{
			Path:    brain.Path(res.Path),
			Title:   res.Title,
			Summary: res.Summary,
			Snippet: truncateSnippet(res.Text, 240),
			Score:   res.Score,
			Source:  "fused",
		})
	}
	if len(hits) > maxResults {
		hits = hits[:maxResults]
	}
	return hits, resp.Trace, nil
}

// truncateSnippet shortens text to at most n characters.
func truncateSnippet(text string, n int) string {
	t := strings.TrimSpace(text)
	if len(t) <= n {
		return t
	}
	return t[:n]
}

// runBM25 asks the bound search index. Falls back to an in-memory scan
// over raw/documents when no index is wired in.
func (k *kbase) runBM25(ctx context.Context, idx *search.Index, q string, maxResults int) ([]SearchHit, error) {
	if idx == nil {
		return k.searchInMemory(ctx, q, maxResults)
	}
	results, err := idx.Search(q, search.SearchOpts{
		MaxResults: maxResults,
	})
	if err != nil {
		return nil, err
	}
	hits := make([]SearchHit, 0, len(results))
	for _, r := range results {
		hits = append(hits, SearchHit{
			Path:     brain.Path(r.Path),
			Title:    r.Title,
			Summary:  r.Summary,
			Snippet:  r.Snippet,
			Score:    r.Score,
			Modified: r.Modified,
			Source:   "bm25",
		})
	}
	return hits, nil
}

// searchInMemory is the fallback scorer used when no FTS index is
// attached. Walks raw/documents in the brain store and scores each
// document against the tokenised query. Deliberately simple: no
// stemming, no snippets, just good enough to keep the tests independent of
// the search package's SQLite dependency.
func (k *kbase) searchInMemory(ctx context.Context, q string, maxResults int) ([]SearchHit, error) {
	terms := tokeniseQuery(q)
	if len(terms) == 0 {
		return nil, nil
	}

	entries, err := k.store.List(ctx, rawDocumentsPrefix, brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return nil, nil
		}
		return nil, err
	}

	var hits []SearchHit
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		if !strings.HasSuffix(string(e.Path), ".md") {
			continue
		}
		data, readErr := k.store.Read(ctx, e.Path)
		if readErr != nil {
			continue
		}
		fm, body := ParseFrontmatter(string(data))
		score := scoreDocument(terms, fm, body)
		if score <= 0 {
			continue
		}
		hits = append(hits, SearchHit{
			Path:    e.Path,
			Title:   firstNonEmpty(fm.Title, fm.Name, strings.TrimSuffix(lastSegment(string(e.Path)), ".md")),
			Summary: firstNonEmpty(fm.Summary, fm.Description),
			Snippet: snippetFor(body, terms),
			Score:   score,
			Source:  "memory",
		})
	}

	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].Path < hits[j].Path
	})
	if len(hits) > maxResults {
		hits = hits[:maxResults]
	}
	return hits, nil
}

// scoreDocument weighs matches across title, summary, tags, and body.
// Mirrors jeff's [scoreArticle] helper at a coarse grain: jeff's
// version is tuned for wiki articles, not raw documents, but the
// weighting pattern carries across.
func scoreDocument(terms []string, fm Frontmatter, body string) float64 {
	titleLower := strings.ToLower(firstNonEmpty(fm.Title, fm.Name))
	summaryLower := strings.ToLower(firstNonEmpty(fm.Summary, fm.Description))
	tagsLower := strings.ToLower(strings.Join(fm.Tags, " "))
	bodyLower := strings.ToLower(body)

	var score float64
	for _, term := range terms {
		score += float64(strings.Count(titleLower, term)) * 3
		score += float64(strings.Count(summaryLower, term)) * 2
		score += float64(strings.Count(tagsLower, term)) * 2
		score += float64(strings.Count(bodyLower, term)) * 1
	}
	return score
}

// snippetFor returns up to 200 characters of body surrounding the first
// hit of any term. Keeps the in-memory fallback self-contained so tests
// never depend on the FTS snippet builder.
func snippetFor(body string, terms []string) string {
	if body == "" || len(terms) == 0 {
		return ""
	}
	lower := strings.ToLower(body)
	for _, term := range terms {
		if term == "" {
			continue
		}
		idx := strings.Index(lower, term)
		if idx < 0 {
			continue
		}
		startWindow := idx - 60
		if startWindow < 0 {
			startWindow = 0
		}
		endWindow := idx + len(term) + 140
		if endWindow > len(body) {
			endWindow = len(body)
		}
		return strings.TrimSpace(body[startWindow:endWindow])
	}
	return ""
}

// tokeniseQuery lowercases the query and returns the non-empty tokens.
func tokeniseQuery(q string) []string {
	raw := strings.Fields(strings.ToLower(strings.TrimSpace(q)))
	out := make([]string, 0, len(raw))
	for _, t := range raw {
		t = strings.Trim(t, `.,;:!?"'()[]{}<>`)
		if t == "" {
			continue
		}
		out = append(out, t)
	}
	return out
}


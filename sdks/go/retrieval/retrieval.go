// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

const (
	defaultTopK        = 10
	defaultCandidateK  = 60
	defaultRerankTopN  = 20
	unanimityWindow    = 3
	unanimityAgreeMin  = 2
	rerankSnippetLimit = 280
)

// Config wires the retrieval layer together. A nil Source yields an
// error at construction time because retrieval without a source is
// meaningless. Every other field is optional; callers opt into
// semantic retrieval by supplying an Embedder, and opt into the
// rerank pass by supplying a Reranker.
type Config struct {
	Source   Source
	Embedder llm.Embedder
	Reranker Reranker
	RRFK     int
	// TrigramChunks overrides the trigram fallback corpus. When nil
	// the retriever lazily asks the Source for its chunks on the
	// first fallback invocation.
	TrigramChunks []trigramChunk
}

// New constructs a [Retriever] from the supplied Config. Returns an
// error if mandatory dependencies are missing.
func New(cfg Config) (Retriever, error) {
	if cfg.Source == nil {
		return nil, fmt.Errorf("retrieval: Config.Source is required")
	}
	k := cfg.RRFK
	if k <= 0 {
		k = RRFDefaultK
	}
	return &retriever{
		source:        cfg.Source,
		embedder:      cfg.Embedder,
		reranker:      cfg.Reranker,
		rrfK:          k,
		trigramSource: cfg.TrigramChunks,
	}, nil
}

type retriever struct {
	source   Source
	embedder llm.Embedder
	reranker Reranker
	rrfK     int

	trigramSource []trigramChunk
	trigramOnce   sync.Once
	trigramIdx    *trigramIndex
}

// Retrieve runs the full hybrid pipeline. The returned Response
// reports everything the pipeline did so eval consumers can reason
// about retrieval quality without re-running.
func (r *retriever) Retrieve(ctx context.Context, req Request) (Response, error) {
	started := time.Now()

	topK := req.TopK
	if topK <= 0 {
		topK = defaultTopK
	}
	candidateK := req.CandidateK
	if candidateK <= 0 {
		candidateK = defaultCandidateK
	}
	rerankTopN := req.RerankTopN
	if rerankTopN <= 0 {
		rerankTopN = defaultRerankTopN
	}

	requestedMode := req.Mode
	if requestedMode == "" {
		requestedMode = ModeAuto
	}
	mode, fellBack := resolveMode(requestedMode, r.embedder != nil)

	trace := Trace{
		RequestedMode:  requestedMode,
		EffectiveMode:  mode,
		RRFK:           r.rrfK,
		CandidateK:     candidateK,
		RerankTopN:     rerankTopN,
		FellBackToBM25: fellBack,
	}
	attempts := make([]Attempt, 0, 6)

	// -- BM25 leg with retry ladder on zero hits. --
	bmCandidates, bmAttempts, usedRetry, bmErr := r.runBM25Leg(ctx, req, candidateK)
	attempts = append(attempts, bmAttempts...)
	trace.UsedRetry = usedRetry
	if bmErr != nil {
		return Response{}, bmErr
	}
	trace.BM25Hits = len(bmCandidates)

	// -- Vector leg (only when the mode requests it). --
	var vecCandidates []rrfCandidate
	if r.embedder != nil && (mode == ModeHybrid || mode == ModeSemantic || mode == ModeHybridRerank) {
		hits, err := r.runVectorLeg(ctx, req, candidateK)
		if err == nil && len(hits) > 0 {
			trace.EmbedderUsed = true
			vecCandidates = hits
		}
	}
	trace.VectorHits = len(vecCandidates)

	// -- Fuse according to mode. --
	fused := r.fuse(mode, bmCandidates, vecCandidates)
	trace.FusedHits = len(fused)

	// -- Intent-aware reweighting (English-only). --
	intent := detectRetrievalIntent(req.Query)
	trace.Intent = intent.label()
	fused = reweightSharedMemoryRanking(req.Query, fused)

	// -- Optional rerank pass. --
	final := r.maybeRerank(ctx, req, mode, fused, bmCandidates, vecCandidates, rerankTopN, &trace)

	if len(final) > topK {
		final = final[:topK]
	}

	return Response{
		Chunks:   final,
		TookMs:   int(time.Since(started).Milliseconds()),
		Trace:    trace,
		Attempts: attempts,
	}, nil
}

// resolveMode collapses an auto request into the concrete mode the
// pipeline will actually run. The fallback flag records cases where
// the caller explicitly asked for semantic or hybrid but no
// embedder was available.
func resolveMode(requested Mode, hasEmbedder bool) (Mode, bool) {
	if !hasEmbedder && (requested == ModeAuto || requested == ModeHybrid || requested == ModeSemantic || requested == ModeHybridRerank) {
		fellBack := requested != ModeAuto
		return ModeBM25, fellBack
	}
	switch requested {
	case ModeAuto:
		return ModeHybrid, false
	default:
		return requested, false
	}
}

// runBM25Leg runs the initial BM25 call and, when it returns zero
// hits and the caller has not suppressed the ladder, walks rungs 1-5
// in order. Returns the accepted candidate list, the attempt log and
// whether the retry ladder was used at all.
func (r *retriever) runBM25Leg(ctx context.Context, req Request, candidateK int) ([]rrfCandidate, []Attempt, bool, error) {
	attempts := make([]Attempt, 0, 6)

	initialExpr := compileToFTS(req.Query)
	candidates, err := r.runBM25(ctx, initialExpr, candidateK, req.Filters)
	if err != nil {
		return nil, attempts, false, fmt.Errorf("retrieval: bm25 leg: %w", err)
	}
	attempts = append(attempts, Attempt{
		Rung:   0,
		Mode:   ModeBM25,
		TopK:   candidateK,
		Reason: "initial",
		Query:  initialExpr,
		Chunks: len(candidates),
	})

	if len(candidates) > 0 || req.SkipRetryLadder {
		return candidates, attempts, false, nil
	}

	// Rung 1: strongest term. Skipped silently when strongest
	// matches the raw trimmed lowered query (no new information).
	loweredRaw := strings.ToLower(strings.TrimSpace(req.Query))
	strongest := strongestTerm(req.Query)
	if strongest != "" && strongest != loweredRaw {
		expr := compileToFTS(strongest)
		hits, err := r.runBM25(ctx, expr, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, true, fmt.Errorf("retrieval: rung 1 bm25: %w", err)
		}
		attempts = append(attempts, Attempt{
			Rung:   1,
			Mode:   ModeBM25,
			TopK:   candidateK,
			Reason: "strongest_term",
			Query:  expr,
			Chunks: len(hits),
		})
		if len(hits) > 0 {
			return hits, attempts, true, nil
		}
	}

	// Rung 2: force-refresh pass-through. No trace row; documented
	// no-op so later SDKs can see the boundary.
	forceRefreshIndex()

	// Rung 3: refreshed sanitised query.
	sanitised := sanitiseQuery(req.Query)
	if sanitised != "" {
		expr := compileToFTS(sanitised)
		hits, err := r.runBM25(ctx, expr, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, true, fmt.Errorf("retrieval: rung 3 bm25: %w", err)
		}
		attempts = append(attempts, Attempt{
			Rung:   3,
			Mode:   ModeBM25,
			TopK:   candidateK,
			Reason: "refreshed_sanitised",
			Query:  expr,
			Chunks: len(hits),
		})
		if len(hits) > 0 {
			return hits, attempts, true, nil
		}
	}

	// Rung 4: refreshed strongest term.
	if s := strongestTerm(sanitised); s != "" {
		expr := compileToFTS(s)
		hits, err := r.runBM25(ctx, expr, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, true, fmt.Errorf("retrieval: rung 4 bm25: %w", err)
		}
		attempts = append(attempts, Attempt{
			Rung:   4,
			Mode:   ModeBM25,
			TopK:   candidateK,
			Reason: "refreshed_strongest",
			Query:  expr,
			Chunks: len(hits),
		})
		if len(hits) > 0 {
			return hits, attempts, true, nil
		}
	}

	// Rung 5: trigram fuzzy fallback.
	tokens := queryTokens(req.Query)
	if len(tokens) > 0 {
		idx := r.ensureTrigramIndex(ctx)
		if idx != nil {
			fuzzy := idx.search(tokens, candidateK)
			candidates := make([]rrfCandidate, 0, len(fuzzy))
			for i, h := range fuzzy {
				candidates = append(candidates, rrfCandidate{
					id:           h.ID,
					path:         h.Path,
					title:        h.Title,
					summary:      h.Summary,
					content:      h.Content,
					bm25Rank:     i,
					haveBM25Rank: true,
				})
			}
			attempts = append(attempts, Attempt{
				Rung:   5,
				Mode:   ModeBM25,
				TopK:   candidateK,
				Reason: "trigram_fuzzy",
				Query:  strings.Join(tokens, " "),
				Chunks: len(candidates),
			})
			if len(candidates) > 0 {
				return candidates, attempts, true, nil
			}
		}
	}

	return nil, attempts, true, nil
}

func (r *retriever) runBM25(ctx context.Context, expr string, k int, filters Filters) ([]rrfCandidate, error) {
	if expr == "" {
		return nil, nil
	}
	hits, err := r.source.SearchBM25(ctx, expr, k, filters)
	if err != nil {
		return nil, err
	}
	out := make([]rrfCandidate, 0, len(hits))
	for i, h := range hits {
		out = append(out, rrfCandidate{
			id:           pickID(h.ID, h.Path),
			path:         h.Path,
			title:        h.Title,
			summary:      h.Summary,
			content:      h.Content,
			bm25Rank:     i,
			haveBM25Rank: true,
		})
	}
	return out, nil
}

func (r *retriever) runVectorLeg(ctx context.Context, req Request, k int) ([]rrfCandidate, error) {
	if r.embedder == nil {
		return nil, nil
	}
	vectors, err := r.embedder.Embed(ctx, []string{req.Query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, nil
	}
	hits, err := r.source.SearchVector(ctx, vectors[0], k, req.Filters)
	if err != nil {
		return nil, err
	}
	out := make([]rrfCandidate, 0, len(hits))
	for i, h := range hits {
		out = append(out, rrfCandidate{
			id:               pickID(h.ID, h.Path),
			path:             h.Path,
			title:            h.Title,
			summary:          h.Summary,
			content:          h.Content,
			vectorSimilarity: h.Similarity,
			haveVectorSim:    true,
			bm25Rank:         i,
			haveBM25Rank:     true,
		})
	}
	return out, nil
}

func (r *retriever) fuse(mode Mode, bm, vec []rrfCandidate) []RetrievedChunk {
	switch mode {
	case ModeBM25:
		return singleList(bm, r.rrfK)
	case ModeSemantic:
		return singleList(vec, r.rrfK)
	default:
		lists := make([][]rrfCandidate, 0, 2)
		if len(bm) > 0 {
			lists = append(lists, bm)
		}
		if len(vec) > 0 {
			lists = append(lists, vec)
		}
		if len(lists) == 0 {
			return nil
		}
		return reciprocalRankFusion(lists, r.rrfK)
	}
}

func singleList(cands []rrfCandidate, k int) []RetrievedChunk {
	if len(cands) == 0 {
		return nil
	}
	safeK := k
	if safeK <= 0 {
		safeK = RRFDefaultK
	}
	out := make([]RetrievedChunk, 0, len(cands))
	for i, c := range cands {
		chunk := RetrievedChunk{
			ChunkID:    c.id,
			DocumentID: c.id,
			Path:       c.path,
			Score:      1.0 / float64(safeK+i+1),
			Text:       c.content,
			Title:      c.title,
			Summary:    c.summary,
		}
		if c.haveBM25Rank {
			chunk.BM25Rank = c.bm25Rank
		}
		if c.haveVectorSim {
			chunk.VectorSimilarity = c.vectorSimilarity
		}
		out = append(out, chunk)
	}
	return out
}

func (r *retriever) maybeRerank(
	ctx context.Context,
	req Request,
	mode Mode,
	fused []RetrievedChunk,
	bm, vec []rrfCandidate,
	rerankTopN int,
	trace *Trace,
) []RetrievedChunk {
	if len(fused) == 0 {
		trace.RerankSkipReason = "empty_candidates"
		return fused
	}
	if r.reranker == nil {
		trace.RerankSkipReason = "no_reranker"
		return fused
	}
	// Only ModeHybridRerank opts into the rerank pass; every other
	// mode returns the fused ranking untouched. Callers that want
	// rerank without hybrid fusion should use ModeHybridRerank and
	// let fusion fall through to the single-list shortcut.
	if mode != ModeHybridRerank {
		trace.RerankSkipReason = "mode_off"
		return fused
	}

	// Unanimity shortcut: when BM25 and vector agree on the head,
	// rerank is unlikely to change the outcome.
	agreements, shortcut := unanimityShortcut(bm, vec, unanimityWindow, unanimityAgreeMin)
	if shortcut {
		trace.RerankSkipReason = "unanimity"
		trace.UnanimitySkipped = true
		trace.Agreements = agreements
		return fused
	}

	n := rerankTopN
	if n > len(fused) {
		n = len(fused)
	}
	head := append([]RetrievedChunk(nil), fused[:n]...)
	tail := append([]RetrievedChunk(nil), fused[n:]...)

	reranked, err := r.reranker.Rerank(ctx, req.Query, head)
	if err != nil || len(reranked) == 0 {
		trace.RerankSkipReason = "rerank_failed"
		return fused
	}
	trace.Reranked = true
	trace.RerankProvider = rerankerName(r.reranker)
	out := make([]RetrievedChunk, 0, len(reranked)+len(tail))
	out = append(out, reranked...)
	out = append(out, tail...)
	return out
}

// rerankerName pulls a best-effort label out of a Reranker. Consumers
// can implement the Named interface to expose a stable identifier
// for traces; anything else falls back to a generic token.
type namedReranker interface {
	Name() string
}

func rerankerName(r Reranker) string {
	if n, ok := r.(namedReranker); ok {
		return n.Name()
	}
	return "custom"
}

// unanimityShortcut reports whether the top `window` positions of the
// BM25 and vector lists agree in at least `min` places. Matches the
// spec's pure-check semantics: no rerank is run when the two
// retrievers converge on the head.
func unanimityShortcut(bm, vec []rrfCandidate, window, min int) (int, bool) {
	if len(bm) < window || len(vec) < window {
		return 0, false
	}
	agreements := 0
	for i := 0; i < window; i++ {
		if bm[i].id != "" && bm[i].id == vec[i].id {
			agreements++
		}
	}
	return agreements, agreements >= min
}

func (r *retriever) ensureTrigramIndex(ctx context.Context) *trigramIndex {
	r.trigramOnce.Do(func() {
		if r.trigramSource != nil {
			r.trigramIdx = buildTrigramIndex(r.trigramSource)
			return
		}
		chunks, err := r.source.Chunks(ctx)
		if err != nil {
			// Best-effort fallback: leave the index nil so the
			// rung silently skips rather than failing the whole
			// retrieval call.
			return
		}
		r.trigramIdx = buildTrigramIndex(chunks)
	})
	return r.trigramIdx
}

func pickID(id, path string) string {
	if id != "" {
		return id
	}
	return path
}

// ComposeRerankText assembles the `title\nsummary` payload used by
// every reranker in the reference implementation. Exposed so
// [Reranker] adapters that want the canonical text shape can reuse
// it without re-deriving the trimming rules.
func ComposeRerankText(r RetrievedChunk) string {
	title := strings.TrimSpace(r.Title)
	summary := strings.TrimSpace(r.Summary)
	switch {
	case title != "" && summary != "":
		return title + "\n" + summary
	case title != "":
		return title
	case summary != "":
		return summary
	}
	body := strings.Join(strings.Fields(r.Text), " ")
	if len(body) <= rerankSnippetLimit {
		return body
	}
	return body[:rerankSnippetLimit] + "..."
}

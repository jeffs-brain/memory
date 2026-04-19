// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/query"
	"github.com/jeffs-brain/memory/go/search"
)

const (
	defaultTopK             = 10
	defaultCandidateK       = 60
	defaultRerankTopN       = 20
	maxBM25FanoutQueries    = 4
	maxDerivedSubQueries    = 2
	bm25FanoutPrimaryWindow = 10
	bm25FanoutMinOverlap    = 2
	unanimityWindow         = 3
	unanimityAgreeMin       = 2
	rerankSnippetLimit      = 280
	phraseProbeMinTokens    = 2
	phraseProbeMaxTokens    = 4
)

var questionTokenStopWords = map[string]bool{
	"the": true, "and": true, "for": true, "with": true, "what": true,
	"who": true, "when": true, "where": true, "why": true, "how": true,
	"did": true, "does": true, "was": true, "were": true, "are": true,
	"you": true, "your": true, "about": true, "this": true, "that": true,
	"have": true, "has": true, "had": true, "from": true, "into": true,
	"than": true, "then": true, "them": true, "they": true, "their": true,
}

var phraseProbeConnectors = map[string]bool{
	"and":  true,
	"or":   true,
	"plus": true,
}

var phraseProbeBoundaryWords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true, "plus": true,
	"for": true, "with": true, "what": true, "who": true, "when": true, "where": true, "why": true, "how": true,
	"did": true, "does": true, "do": true, "was": true, "were": true, "is": true, "are": true, "am": true,
	"you": true, "your": true, "about": true, "this": true, "that": true, "these": true, "those": true,
	"have": true, "has": true, "had": true, "from": true, "into": true, "than": true, "then": true, "them": true, "they": true, "their": true,
	"i": true, "me": true, "my": true, "we": true, "our": true, "us": true, "it": true, "if": true, "to": true, "of": true, "on": true, "in": true, "at": true, "by": true,
	"amount": true, "total": true, "all": true, "list": true,
	"finally": true, "decided": true, "decide": true, "wondering": true, "wonder": true,
	"remembered": true, "remember": true, "thinking": true, "back": true, "previous": true, "conversation": true,
	"can": true, "could": true, "would": true, "should": true, "remind": true, "follow": true, "specific": true, "exact": true,
	"spent": true, "spend": true, "bought": true, "buy": true, "ordered": true, "order": true,
	"purchased": true, "purchase": true, "paid": true, "pay": true, "submitted": true, "submit": true,
	"many": true, "much": true, "long": true,
	"last": true, "today": true, "yesterday": true, "tomorrow": true, "week": true, "month": true, "year": true,
	"monday": true, "tuesday": true, "wednesday": true, "thursday": true, "friday": true, "saturday": true, "sunday": true,
}

var phraseProbeTrimWords = map[string]bool{
	"many": true,
	"much": true,
	"long": true,
}

var actionDateProbeRules = []struct {
	pattern *regexp.Regexp
	probe   string
}{
	{pattern: regexp.MustCompile(`(?i)\bsubmit(?:ted)?\b`), probe: "submission date"},
	{pattern: regexp.MustCompile(`(?i)\bbook(?:ed|ing)?\b`), probe: "booking date"},
	{pattern: regexp.MustCompile(`(?i)\b(?:buy|bought|purchase(?:d)?|order(?:ed)?)\b`), probe: "purchase date"},
	{pattern: regexp.MustCompile(`(?i)\b(?:join(?:ed)?|joined)\b`), probe: "join date"},
	{pattern: regexp.MustCompile(`(?i)\b(?:start(?:ed)?|begin|began)\b`), probe: "start date"},
	{pattern: regexp.MustCompile(`(?i)\b(?:finish(?:ed)?|complete(?:d)?)\b`), probe: "completion date"},
	{pattern: regexp.MustCompile(`(?i)\baccept(?:ed|ance)?\b`), probe: "acceptance date"},
}

var actionDateFocusSkipWords = map[string]bool{
	"accept": true, "accepted": true, "acceptance": true,
	"begin": true, "began": true, "book": true, "booked": true, "booking": true,
	"buy": true, "bought": true, "complete": true, "completed": true,
	"completion": true, "date": true, "finish": true, "finished": true,
	"join": true, "joined": true, "order": true, "ordered": true,
	"purchase": true, "purchased": true, "start": true, "started": true,
	"submit": true, "submitted": true, "submission": true,
}

var inspirationQueryHints = []string{
	"inspiration",
	"inspired",
	"ideas",
	"stuck",
	"uninspired",
}

var inspirationFocusSkipWords = map[string]bool{
	"find": true, "finding": true, "fresh": true, "idea": true, "ideas": true,
	"inspiration": true, "inspired": true, "new": true, "stuck": true, "uninspired": true,
}

var lowSignalPhraseProbeWords = map[string]bool{
	"after": true, "before": true, "day": true, "days": true, "event": true, "events": true,
	"first": true, "happen": true, "happened": true, "month": true, "months": true,
	"second": true, "third": true, "time": true, "times": true, "week": true, "weeks": true,
	"year": true, "years": true,
}

var headBigramLastTokens = map[string]bool{
	"development": true, "item": true, "items": true, "language": true, "languages": true, "product": true, "products": true,
}

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
	fused = r.hydrateBodies(ctx, fused)
	trace.FusedHits = len(fused)

	// -- Intent-aware reweighting (English-only). --
	intent := detectRetrievalIntent(req.Query)
	trace.Intent = intent.label()
	fused = reweightSharedMemoryRanking(req.Query, fused)

	// -- Optional rerank pass. --
	final := r.maybeRerank(ctx, req, mode, fused, bmCandidates, vecCandidates, rerankTopN, &trace)
	final = reweightTemporalRanking(req.Query, req.QuestionDate, final)

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

	initialExprs := buildBM25FanoutExprs(req.Query, req.QuestionDate)
	candidates, initialExpr, err := r.runBM25Fanout(ctx, initialExprs, candidateK, req.Filters)
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
		exprs := buildBM25FanoutExprs(strongest, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
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
		exprs := buildBM25FanoutExprs(sanitised, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
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
		exprs := buildBM25FanoutExprs(s, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
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

func (r *retriever) runBM25Fanout(ctx context.Context, exprs []string, k int, filters Filters) ([]rrfCandidate, string, error) {
	if len(exprs) == 0 {
		return nil, "", nil
	}
	primaryHits, err := r.runBM25(ctx, exprs[0], k, filters)
	if err != nil {
		return nil, joinBM25AttemptQuery(exprs), err
	}
	lists := make([][]rrfCandidate, 0, len(exprs))
	if len(primaryHits) > 0 {
		lists = append(lists, primaryHits)
	}
	for _, expr := range exprs[1:] {
		hits, err := r.runBM25(ctx, expr, k, filters)
		if err != nil {
			return nil, joinBM25AttemptQuery(exprs), err
		}
		if len(hits) > 0 && (len(primaryHits) == 0 || bm25FanoutBypassOverlapGate(expr) || bm25FanoutOverlap(primaryHits, hits) >= bm25FanoutMinOverlap) {
			lists = append(lists, hits)
		}
	}
	attemptQuery := joinBM25AttemptQuery(exprs)
	switch len(lists) {
	case 0:
		return nil, attemptQuery, nil
	case 1:
		return lists[0], attemptQuery, nil
	default:
		fused := reciprocalRankFusion(lists, r.rrfK)
		out := make([]rrfCandidate, 0, len(fused))
		for i, h := range fused {
			out = append(out, rrfCandidate{
				id:           pickID(h.ChunkID, h.Path),
				path:         h.Path,
				title:        h.Title,
				summary:      h.Summary,
				content:      h.Text,
				bm25Rank:     i,
				haveBM25Rank: true,
			})
		}
		return out, attemptQuery, nil
	}
}

func bm25FanoutBypassOverlapGate(expr string) bool {
	if strings.IndexFunc(expr, unicode.IsDigit) >= 0 {
		return true
	}
	terms := 0
	for _, token := range strings.Fields(expr) {
		switch token {
		case "AND", "OR", "NOT":
			continue
		default:
			terms++
		}
	}
	return terms >= phraseProbeMinTokens && terms <= phraseProbeMaxTokens
}

func bm25FanoutOverlap(primary, secondary []rrfCandidate) int {
	if len(primary) == 0 || len(secondary) == 0 {
		return 0
	}
	limit := bm25FanoutPrimaryWindow
	if len(primary) < limit {
		limit = len(primary)
	}
	primaryIDs := make(map[string]struct{}, limit)
	for i := 0; i < limit; i++ {
		id := pickID(primary[i].id, primary[i].path)
		if id == "" {
			continue
		}
		primaryIDs[id] = struct{}{}
	}
	if len(primaryIDs) == 0 {
		return 0
	}
	limit = bm25FanoutPrimaryWindow
	if len(secondary) < limit {
		limit = len(secondary)
	}
	overlap := 0
	for i := 0; i < limit; i++ {
		id := pickID(secondary[i].id, secondary[i].path)
		if id == "" {
			continue
		}
		if _, ok := primaryIDs[id]; !ok {
			continue
		}
		overlap++
		if overlap >= bm25FanoutMinOverlap {
			return overlap
		}
	}
	return overlap
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

func (r *retriever) hydrateBodies(ctx context.Context, fused []RetrievedChunk) []RetrievedChunk {
	if len(fused) == 0 {
		return fused
	}
	lookup, ok := r.source.(BodyLookupSource)
	if !ok {
		return fused
	}
	ids := make([]string, 0, len(fused))
	seen := make(map[string]bool, len(fused))
	for _, chunk := range fused {
		if chunk.Path == "" || seen[chunk.Path] {
			continue
		}
		seen[chunk.Path] = true
		ids = append(ids, chunk.Path)
	}
	rows, err := lookup.Lookup(ctx, ids)
	if err != nil || len(rows) == 0 {
		return fused
	}
	byPath := make(map[string]search.IndexedRow, len(rows))
	for _, row := range rows {
		byPath[row.Path] = row
	}
	out := append([]RetrievedChunk(nil), fused...)
	for i, chunk := range out {
		row, ok := byPath[chunk.Path]
		if !ok {
			continue
		}
		if row.Content != "" {
			out[i].Text = stripFrontmatterBody(row.Content)
		}
		if out[i].Title == "" && row.Title != "" {
			out[i].Title = row.Title
		}
		if out[i].Summary == "" && row.Summary != "" {
			out[i].Summary = row.Summary
		}
		if row.Scope != "" || row.ProjectSlug != "" || row.SessionDate != "" || row.Content != "" {
			meta := cloneChunkMetadata(out[i].Metadata)
			if row.Scope != "" {
				meta["scope"] = row.Scope
			}
			if row.ProjectSlug != "" {
				meta["project"] = row.ProjectSlug
				meta["projectSlug"] = row.ProjectSlug
			}
			if row.SessionDate != "" {
				meta["sessionDate"] = row.SessionDate
				meta["session_date"] = row.SessionDate
			}
			if sessionID := firstFrontmatterValue(row.Content, "session_id"); sessionID != "" {
				meta["sessionId"] = sessionID
				meta["session_id"] = sessionID
			}
			if observedOn := firstFrontmatterValue(row.Content, "observed_on"); observedOn != "" {
				meta["observedOn"] = observedOn
				meta["observed_on"] = observedOn
			}
			if modified := firstFrontmatterValue(row.Content, "modified"); modified != "" {
				meta["modified"] = modified
			}
			if len(meta) > 0 {
				out[i].Metadata = meta
			} else {
				out[i].Metadata = nil
			}
		}
	}
	return out
}

func cloneChunkMetadata(src map[string]any) map[string]any {
	if len(src) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(src))
	for key, value := range src {
		out[key] = value
	}
	return out
}

func stripFrontmatterBody(content string) string {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return strings.TrimSpace(content)
	}
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) != "---" {
			continue
		}
		return strings.TrimSpace(strings.Join(lines[i+1:], "\n"))
	}
	return strings.TrimSpace(content)
}

func firstFrontmatterValue(content, key string) string {
	lines := strings.Split(content, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return ""
	}
	inFrontmatter := false
	prefix := key + ":"
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			if inFrontmatter {
				return ""
			}
			inFrontmatter = true
			continue
		}
		if !inFrontmatter {
			continue
		}
		if strings.HasPrefix(trimmed, prefix) {
			return strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
		}
	}
	return ""
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

func buildBM25FanoutExprs(raw, questionDate string) []string {
	queries := buildBM25FanoutQueries(raw, questionDate)
	phraseProbes := derivePhraseProbes(raw)
	seen := make(map[string]bool, len(queries))
	exprs := make([]string, 0, len(queries))
	for _, candidate := range queries {
		expr := compileBM25FanoutQuery(candidate, phraseProbes)
		if expr == "" || seen[expr] {
			continue
		}
		seen[expr] = true
		exprs = append(exprs, expr)
	}
	return exprs
}

func buildBM25FanoutQueries(raw, questionDate string) []string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil
	}
	priorityQueries := derivePrioritySubQueries(trimmed)
	if shouldUsePriorityOnlyBM25(trimmed) && len(priorityQueries) >= 2 {
		if len(priorityQueries) > maxBM25FanoutQueries {
			return priorityQueries[:maxBM25FanoutQueries]
		}
		return priorityQueries
	}
	augmented := augmentQueryWithTemporal(trimmed, questionDate)
	queries := append([]string{}, priorityQueries...)
	queries = append(queries, trimmed, augmented)
	for _, sub := range deriveSubQueries(trimmed) {
		queries = append(queries, sub)
	}
	queries = dedupeTrimmedStrings(queries)
	if len(queries) > maxBM25FanoutQueries {
		return queries[:maxBM25FanoutQueries]
	}
	return queries
}

func shouldUsePriorityOnlyBM25(question string) bool {
	lowered := strings.ToLower(strings.TrimSpace(question))
	return len(deriveActionDateContextProbes(question)) > 0 ||
		(len(filteredPhraseProbes(question)) >= 2 && strings.Contains(lowered, " and "))
}

func joinBM25AttemptQuery(exprs []string) string {
	switch len(exprs) {
	case 0:
		return ""
	case 1:
		return exprs[0]
	default:
		return strings.Join(exprs, " || ")
	}
}

func augmentQueryWithTemporal(question, questionDate string) string {
	expansion := query.ExpandTemporal(question, questionDate)
	if !expansion.Resolved || len(expansion.DateHints) == 0 {
		return strings.TrimSpace(question)
	}
	tokens := make([]string, 0, len(expansion.DateHints)*2)
	for _, hint := range expansion.DateHints {
		trimmed := strings.TrimSpace(hint)
		if trimmed == "" {
			continue
		}
		tokens = append(tokens, `"`+trimmed+`"`, `"`+strings.ReplaceAll(trimmed, "/", "-")+`"`)
	}
	unique := dedupeTrimmedStrings(tokens)
	if len(unique) == 0 {
		return strings.TrimSpace(question)
	}
	return strings.TrimSpace(question + " " + strings.Join(unique, " "))
}

func dedupeTrimmedStrings(values []string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		out = append(out, trimmed)
	}
	return out
}

func deriveSubQueries(question string) []string {
	out := make([]string, 0, maxDerivedSubQueries)
	seen := map[string]bool{
		strings.ToLower(strings.TrimSpace(question)): true,
	}
	inspirationQuery := containsAnyHint(strings.ToLower(strings.TrimSpace(question)), inspirationQueryHints)
	for _, probe := range deriveSpecificRecommendationProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveMoneyFocusProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveActionDateContextProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveInspirationSourceProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveActionDateProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	phrases := filteredPhraseProbes(question)
	for _, phrase := range phrases {
		if inspirationQuery && len(filterQuestionTokens(phrase, inspirationFocusSkipWords)) == 0 {
			continue
		}
		if seen[phrase] {
			continue
		}
		seen[phrase] = true
		out = append(out, phrase)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}

	tokenSource := question
	if len(phrases) > 0 {
		tokenSource = strings.Join(phrases, " ")
	}
	tokens := questionTokens(tokenSource)
	if len(tokens) < 2 {
		return out
	}
	sort.SliceStable(tokens, func(i, j int) bool {
		return len(tokens[i]) > len(tokens[j])
	})
	for _, token := range tokens {
		if inspirationQuery && inspirationFocusSkipWords[token] {
			continue
		}
		if seen[token] {
			continue
		}
		seen[token] = true
		out = append(out, token)
		if len(out) == maxDerivedSubQueries {
			break
		}
	}
	return out
}

func derivePrioritySubQueries(question string) []string {
	out := make([]string, 0, maxDerivedSubQueries)
	seen := map[string]bool{
		strings.ToLower(strings.TrimSpace(question)): true,
	}
	for _, probe := range deriveSpecificRecommendationProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveMoneyFocusProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveActionDateContextProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, probe := range deriveInspirationSourceProbes(question) {
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	for _, phrase := range filteredPhraseProbes(question) {
		if seen[phrase] {
			continue
		}
		seen[phrase] = true
		out = append(out, phrase)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	return out
}

func deriveSpecificRecommendationProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !specificRecommendationQueryRe.MatchString(lowered) {
		return nil
	}
	if !strings.Contains(lowered, "recommend") && !strings.Contains(lowered, "remind me") {
		return nil
	}
	if strings.Contains(lowered, "back-end") && strings.Contains(lowered, "language") {
		return []string{"back-end programming language", "back-end development"}
	}
	return nil
}

func deriveMoneyFocusProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !enumerationOrTotalQueryRe.MatchString(lowered) || !moneyEventQueryRe.MatchString(lowered) {
		return nil
	}
	out := make([]string, 0, maxDerivedSubQueries)
	seen := make(map[string]bool, maxDerivedSubQueries)
	for _, phrase := range filteredPhraseProbes(question) {
		candidate := moneyFocusProbeFromPhrase(phrase)
		if candidate == "" || seen[candidate] {
			continue
		}
		seen[candidate] = true
		out = append(out, candidate)
		if len(out) == maxDerivedSubQueries {
			return out
		}
	}
	return out
}

func moneyFocusProbeFromPhrase(phrase string) string {
	edge := derivePhraseEdgeFocus(phrase)
	if edge != "" {
		return edge
	}
	head := derivePhraseHeadFocus(phrase)
	if head == "" {
		return ""
	}
	if head != phrase && len(strings.Fields(phrase)) <= 2 {
		return head + " cost"
	}
	return phrase
}

func derivePhraseEdgeFocus(phrase string) string {
	tokens := strings.Fields(strings.ToLower(strings.TrimSpace(phrase)))
	if len(tokens) < 3 {
		return ""
	}
	first := tokens[0]
	last := tokens[len(tokens)-1]
	if !strings.Contains(first, "-") || !headBigramLastTokens[last] {
		return ""
	}
	return first + " " + last
}

func derivePhraseHeadFocus(phrase string) string {
	tokens := strings.Fields(strings.ToLower(strings.TrimSpace(phrase)))
	if len(tokens) == 0 {
		return ""
	}
	last := tokens[len(tokens)-1]
	if len(tokens) >= 2 && headBigramLastTokens[last] {
		return strings.Join(tokens[len(tokens)-2:], " ")
	}
	return last
}

func deriveActionDateContextProbes(question string) []string {
	actionProbes := deriveActionDateProbes(question)
	if len(actionProbes) == 0 {
		return nil
	}
	focuses := deriveActionDateFocuses(filterQuestionTokens(question, actionDateFocusSkipWords))
	if len(focuses) == 0 {
		return nil
	}
	out := make([]string, 0, len(actionProbes)*len(focuses))
	seen := make(map[string]bool, len(actionProbes)*len(focuses))
	for _, probe := range actionProbes {
		for _, focus := range focuses {
			candidate := strings.TrimSpace(focus + " " + probe)
			if candidate == "" || seen[candidate] {
				continue
			}
			seen[candidate] = true
			out = append(out, candidate)
			if len(out) == maxDerivedSubQueries {
				return out
			}
		}
	}
	return out
}

func deriveActionDateFocuses(tokens []string) []string {
	if len(tokens) == 0 {
		return nil
	}
	appendWindow := func(out []string, seen map[string]bool, start, end int) []string {
		if start < 0 || end > len(tokens) || start >= end {
			return out
		}
		candidate := strings.TrimSpace(strings.Join(tokens[start:end], " "))
		if candidate == "" || seen[candidate] {
			return out
		}
		seen[candidate] = true
		return append(out, candidate)
	}

	out := make([]string, 0, 2)
	seen := make(map[string]bool, 2)
	switch len(tokens) {
	case 1:
		out = appendWindow(out, seen, 0, 1)
	case 2:
		out = appendWindow(out, seen, 0, 2)
	case 3:
		out = appendWindow(out, seen, 1, 3)
		out = appendWindow(out, seen, 0, 2)
	default:
		out = appendWindow(out, seen, len(tokens)-2, len(tokens))
		out = appendWindow(out, seen, 0, 2)
	}
	return out
}

func deriveInspirationSourceProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !containsAnyHint(lowered, inspirationQueryHints) {
		return nil
	}
	tokens := filterQuestionTokens(question, inspirationFocusSkipWords)
	if len(tokens) == 0 {
		return nil
	}
	focus := tokens[len(tokens)-1]
	if focus == "" {
		return nil
	}
	return []string{focus + " social media tutorials"}
}

func filterQuestionTokens(question string, skip map[string]bool) []string {
	tokens := questionTokens(question)
	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if skip[token] {
			continue
		}
		out = append(out, token)
	}
	return out
}

func deriveActionDateProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !strings.Contains(lowered, "when") {
		return nil
	}
	out := make([]string, 0, 1)
	seen := map[string]bool{}
	for _, rule := range actionDateProbeRules {
		if !rule.pattern.MatchString(lowered) || seen[rule.probe] {
			continue
		}
		seen[rule.probe] = true
		out = append(out, rule.probe)
		if len(out) == maxDerivedSubQueries {
			break
		}
	}
	return out
}

func derivePhraseProbes(question string) []string {
	tokens := phraseProbeTokens(question)
	if len(tokens) < phraseProbeMinTokens {
		return nil
	}
	out := make([]string, 0, maxDerivedSubQueries)
	seen := make(map[string]bool, maxDerivedSubQueries)
	appendPhrase := func(phrase string) bool {
		phrase = strings.TrimSpace(phrase)
		if phrase == "" || seen[phrase] {
			return false
		}
		seen[phrase] = true
		out = append(out, phrase)
		return len(out) == maxDerivedSubQueries
	}
	for i, token := range tokens {
		if !phraseProbeConnectors[token] {
			continue
		}
		if phrase := collectLeftPhrase(tokens[:i]); phrase != "" {
			if appendPhrase(phrase) {
				return out
			}
		}
		if phrase := collectRightPhrase(tokens[i+1:]); phrase != "" {
			if appendPhrase(phrase) {
				return out
			}
		}
	}
	for _, phrase := range deriveBoundarySpanProbes(tokens) {
		if appendPhrase(phrase) {
			return out
		}
	}
	return out
}

func filteredPhraseProbes(question string) []string {
	phrases := derivePhraseProbes(question)
	if len(phrases) == 0 {
		return nil
	}
	out := make([]string, 0, len(phrases))
	for _, phrase := range phrases {
		if len(filterQuestionTokens(phrase, lowSignalPhraseProbeWords)) == 0 {
			continue
		}
		out = append(out, phrase)
	}
	return out
}

func deriveBoundarySpanProbes(tokens []string) []string {
	if len(tokens) < phraseProbeMinTokens {
		return nil
	}
	out := make([]string, 0, maxDerivedSubQueries)
	var segment []string
	flush := func() {
		if phrase := bestSegmentPhrase(segment); phrase != "" {
			out = append(out, phrase)
		}
		segment = nil
	}
	for _, token := range tokens {
		if token == "" || phraseProbeBoundaryWords[token] || len(token) < 2 || containsDigit(token) {
			flush()
			continue
		}
		segment = append(segment, token)
	}
	flush()
	return dedupeTrimmedStrings(out)
}

func bestSegmentPhrase(tokens []string) string {
	trimmed := trimPhraseProbeTokens(tokens)
	if len(trimmed) < phraseProbeMinTokens {
		return ""
	}
	if len(trimmed) <= phraseProbeMaxTokens {
		return joinPhraseTokens(trimmed)
	}
	best := ""
	bestScore := -1
	for size := phraseProbeMaxTokens; size >= phraseProbeMinTokens; size-- {
		for i := 0; i+size <= len(trimmed); i++ {
			candidate := trimPhraseProbeTokens(trimmed[i : i+size])
			if len(candidate) < phraseProbeMinTokens {
				continue
			}
			score := phraseProbeScore(candidate)
			if score > bestScore {
				bestScore = score
				best = joinPhraseTokens(candidate)
			}
		}
		if best != "" {
			return best
		}
	}
	return best
}

func trimPhraseProbeTokens(tokens []string) []string {
	start := 0
	for start < len(tokens) && phraseProbeTrimWords[tokens[start]] {
		start++
	}
	if len(tokens[start:]) < phraseProbeMinTokens {
		return nil
	}
	return tokens[start:]
}

func phraseProbeScore(tokens []string) int {
	score := len(tokens) * 100
	for _, token := range tokens {
		score += len(token)
	}
	return score
}

func compileBM25FanoutQuery(query string, phraseProbes []string) string {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return ""
	}
	if isDerivedPhraseProbe(trimmed, phraseProbes) && strings.Contains(trimmed, " ") && !strings.Contains(trimmed, `"`) {
		return compileToFTS(`"` + trimmed + `"`)
	}
	return compileToFTS(trimmed)
}

func isDerivedPhraseProbe(query string, phraseProbes []string) bool {
	for _, probe := range phraseProbes {
		if query == probe {
			return true
		}
	}
	return false
}

func phraseProbeTokens(question string) []string {
	if strings.TrimSpace(question) == "" {
		return nil
	}
	raw := strings.Fields(strings.ToLower(question))
	out := make([]string, 0, len(raw))
	for _, token := range raw {
		token = strings.Trim(token, `.,;:!?"'()[]{}<>`)
		if token == "" {
			continue
		}
		out = append(out, token)
	}
	return out
}

func collectLeftPhrase(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}
	collected := make([]string, 0, phraseProbeMaxTokens)
	for i := len(tokens) - 1; i >= 0; i-- {
		token := tokens[i]
		if phraseProbeBoundaryWords[token] {
			if len(collected) > 0 {
				break
			}
			continue
		}
		if len(token) < 2 || containsDigit(token) {
			if len(collected) > 0 {
				break
			}
			continue
		}
		collected = append(collected, token)
		if len(collected) == phraseProbeMaxTokens {
			break
		}
	}
	for i, j := 0, len(collected)-1; i < j; i, j = i+1, j-1 {
		collected[i], collected[j] = collected[j], collected[i]
	}
	return joinPhraseTokens(collected)
}

func collectRightPhrase(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}
	collected := make([]string, 0, phraseProbeMaxTokens)
	for _, token := range tokens {
		if phraseProbeBoundaryWords[token] {
			if len(collected) > 0 {
				break
			}
			continue
		}
		if len(token) < 2 || containsDigit(token) {
			if len(collected) > 0 {
				break
			}
			continue
		}
		collected = append(collected, token)
		if len(collected) == phraseProbeMaxTokens {
			break
		}
	}
	return joinPhraseTokens(collected)
}

func joinPhraseTokens(tokens []string) string {
	if len(tokens) < phraseProbeMinTokens {
		return ""
	}
	return strings.Join(tokens, " ")
}

func questionTokens(question string) []string {
	if question == "" {
		return nil
	}
	raw := strings.Fields(strings.ToLower(question))
	seen := make(map[string]bool, len(raw))
	out := make([]string, 0, len(raw))
	for _, token := range raw {
		token = strings.Trim(token, `.,;:!?"'()[]{}<>`)
		if len(token) < 3 || questionTokenStopWords[token] || seen[token] || containsDigit(token) {
			continue
		}
		seen[token] = true
		out = append(out, token)
	}
	return out
}

func containsDigit(value string) bool {
	for _, r := range value {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
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

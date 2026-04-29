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
	maxBM25FanoutQueries    = 8
	maxVectorFanoutQueries  = 3
	maxDerivedSubQueries    = 2
	maxCoverageFacetQueries = 4
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
	"different": true, "type": true, "types": true,
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

var coverageFacetSkipWords = map[string]bool{
	"amount": true, "current": true, "currently": true, "different": true,
	"day": true, "days": true, "event": true, "events": true, "exact": true, "first": true,
	"many": true, "much": true, "number": true, "previous": true,
	"pass": true, "passed": true, "recent": true, "same": true, "specific": true, "total": true,
	"type": true, "types": true,
}

var episodicDescriptorSkipWords = map[string]bool{
	"back": true, "chat": true, "conversation": true, "created": true, "create": true,
	"drafted": true, "earlier": true, "generated": true, "gave": true, "looking": true,
	"made": true, "previous": true, "remind": true, "remember": true, "said": true,
	"two": true, "wrote": true,
}

var capitalisedFacetGlueWords = map[string]bool{
	"of":  true,
	"the": true,
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

var (
	countTotalIncludingRe    = regexp.MustCompile(`(?i)\bincluding\s+(.+)$`)
	dateArithmeticQueryRe    = regexp.MustCompile(`(?i)\bhow many days?\b.*\b(?:when|between|before|after)\b`)
	typeContextQueryRe       = regexp.MustCompile(`(?i)\b(?:different\s+)?(?:types?|kinds?|varieties)\s+of\s+(.+?)\s+(?:have|has|had|do|does|did|am|are|is|was|were|use|used|using|try|tried|trying|make|made|cook|cooked|in|for|with)\b`)
	typeContextScopeRe       = regexp.MustCompile(`(?i)\b(?:in|for|with)\s+(?:my|our|the|a|an)?\s*(.+?)(?:[?.!]|$)`)
	recallTitleProbeRe       = regexp.MustCompile(`\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){1,4}\b`)
	dateArithmeticBoundaryRe = regexp.MustCompile(`(?i)^(?:when|while|after|before|between|and|then|ago|days?|day)$`)
	dateArithmeticWhenRe     = regexp.MustCompile(`(?i)\bhow many days?\s+ago\s+did\s+i\s+(.+?)\s+when\s+i\s+(.+?)(?:[?.!]|$)`)
	exactListRecallRe        = regexp.MustCompile(`(?i)\b(?:specific|exact|precise)\b.*\b(?:recommend(?:ed|ation|ations)?|suggest(?:ed|ion|ions)?|options?|list|items?|languages?)\b|\b(?:recommend(?:ed|ation|ations)?|suggest(?:ed|ion|ions)?|options?|list|items?|languages?)\b.*\b(?:specific|exact|precise)\b`)
	rawMessageMarkerRe       = regexp.MustCompile(`^\s*\[([A-Za-z][A-Za-z0-9_-]*)\]\s*:\s*(.*)$`)
	rawSectionHeadingRe      = regexp.MustCompile(`(?i)^\s*(verse|chorus|bridge|outro|chapter|section|ingredients|instructions|steps?|code|subject|table|row|option)\b(?:\s+\d+)?\s*:`)
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

	searchReq := req
	filtersWithDate, generatedDateBound := filtersWithQuestionDate(req.Filters, req.QuestionDate)
	searchReq.Filters = filtersWithDate

	// -- BM25 leg with retry ladder on zero hits or weak hits. --
	bmCandidates, bmAttempts, usedRetry, qualityRetry, bmErr := r.runBM25Leg(ctx, searchReq, candidateK, generatedDateBound, req.Filters)
	attempts = append(attempts, bmAttempts...)
	trace.UsedRetry = usedRetry
	trace.QualityRetry = qualityRetry
	if bmErr != nil {
		return Response{}, bmErr
	}
	trace.BM25Hits = len(bmCandidates)

	// -- Vector leg (only when the mode requests it). --
	var vecCandidates []rrfCandidate
	if r.embedder != nil && (mode == ModeHybrid || mode == ModeSemantic || mode == ModeHybridRerank) {
		hits, err := r.runVectorLeg(ctx, searchReq, candidateK)
		if err == nil && len(hits) > 0 {
			trace.EmbedderUsed = true
			vecCandidates = hits
		} else if err != nil {
			trace.VectorSkipReason = "vector_error"
		} else {
			trace.VectorSkipReason = "no_vector_hits"
		}
	} else if mode == ModeHybrid || mode == ModeSemantic || mode == ModeHybridRerank {
		trace.VectorSkipReason = "no_embedder"
	}
	trace.VectorHits = len(vecCandidates)
	if requestedMode == ModeAuto && mode == ModeHybrid && len(vecCandidates) == 0 {
		mode = ModeBM25
		trace.EffectiveMode = ModeBM25
	}

	// -- Fuse according to mode. --
	fused := r.fuse(mode, bmCandidates, vecCandidates)
	fused = r.hydrateBodies(ctx, fused)
	trace.FusedHits = len(fused)
	fused = r.expandSameSessionNeighbours(ctx, searchReq, fused, topK, &trace)

	// -- Intent-aware reweighting (English-only). --
	intent := detectRetrievalIntent(req.Query)
	trace.Intent = intent.label()
	fused = reweightSharedMemoryRanking(req.Query, fused)
	fused = r.expandEpisodicRecall(ctx, searchReq, fused, topK, &trace)

	// -- Optional rerank pass. --
	final := r.maybeRerank(ctx, req, mode, fused, bmCandidates, vecCandidates, rerankTopN, &trace)
	final = reweightTemporalRanking(req.Query, req.QuestionDate, final)
	final = dedupeNearDuplicateChunks(final)
	var aggregateTrace aggregateEvidenceTrace
	final, aggregateTrace = groupAggregateEvidence(req.Query, final)
	trace.AggregateEvidenceGroups = aggregateTrace.Groups
	trace.AggregateEvidenceSuppressed = aggregateTrace.Suppressed
	var stateTrace statePromotionTrace
	final, stateTrace = promoteCurrentStateEvidence(req.Query, req.QuestionDate, final)
	trace.StateIntent = stateTrace.Intent
	trace.StatePromotions = stateTrace.Promotions

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

// runBM25Leg runs the initial BM25 call and, when it returns zero or
// low-quality hits, walks rungs 1-5 in order. Returns the accepted
// candidate list, the attempt log, whether retry ran, and whether
// quality retry caused it.
func (r *retriever) runBM25Leg(ctx context.Context, req Request, candidateK int, generatedDateBound bool, originalFilters Filters) ([]rrfCandidate, []Attempt, bool, bool, error) {
	attempts := make([]Attempt, 0, 6)

	initialExprs := buildBM25FanoutExprs(req.Query, req.QuestionDate)
	candidates, initialExpr, err := r.runBM25Fanout(ctx, initialExprs, candidateK, req.Filters)
	if err != nil {
		return nil, attempts, false, false, fmt.Errorf("retrieval: bm25 leg: %w", err)
	}
	initialQuality := assessBM25CandidateQuality(req.Query, req.QuestionDate, candidates)
	attempts = append(attempts, Attempt{
		Rung:         0,
		Mode:         ModeBM25,
		TopK:         candidateK,
		Reason:       "initial",
		Query:        initialExpr,
		Chunks:       len(candidates),
		Quality:      initialQuality.label,
		QualityScore: initialQuality.score,
		DateBounded:  generatedDateBound,
	})

	if req.SkipRetryLadder {
		return candidates, attempts, false, false, nil
	}
	if len(candidates) > 0 && !initialQuality.shouldRetry {
		return candidates, attempts, false, false, nil
	}

	if len(candidates) > 0 {
		retryCandidates, retryAttempts, accepted, err := r.runBM25RetryRungs(ctx, req, candidateK, generatedDateBound, candidates, initialQuality)
		attempts = append(attempts, retryAttempts...)
		if err != nil {
			return nil, attempts, true, true, err
		}
		if accepted {
			return mergeRetryWithBaselineCandidates(retryCandidates, candidates), attempts, true, true, nil
		}
		return candidates, attempts, true, true, nil
	}

	retryCandidates, retryAttempts, accepted, err := r.runBM25RetryRungs(ctx, req, candidateK, generatedDateBound, nil, initialQuality)
	attempts = append(attempts, retryAttempts...)
	if err != nil {
		return nil, attempts, true, false, err
	}
	if accepted {
		return retryCandidates, attempts, true, false, nil
	}
	if generatedDateBound {
		hits, expr, err := r.runBM25Fanout(ctx, buildBM25FanoutExprs(req.Query, req.QuestionDate), candidateK, originalFilters)
		if err != nil {
			return nil, attempts, true, false, fmt.Errorf("retrieval: relaxed date bound bm25: %w", err)
		}
		quality := assessBM25CandidateQuality(req.Query, req.QuestionDate, hits)
		attempts = append(attempts, Attempt{
			Rung:         6,
			Mode:         ModeBM25,
			TopK:         candidateK,
			Reason:       "relaxed_date_bound",
			Query:        expr,
			Chunks:       len(hits),
			Quality:      quality.label,
			QualityScore: quality.score,
		})
		if len(hits) > 0 {
			return hits, attempts, true, false, nil
		}
	}

	return nil, attempts, true, false, nil
}

func (r *retriever) runBM25RetryRungs(ctx context.Context, req Request, candidateK int, dateBounded bool, baseline []rrfCandidate, baselineQuality bm25Quality) ([]rrfCandidate, []Attempt, bool, error) {
	attempts := make([]Attempt, 0, 5)
	acceptHits := func(hits []rrfCandidate, quality bm25Quality) bool {
		if len(hits) == 0 {
			return false
		}
		if len(baseline) == 0 {
			return true
		}
		return bm25QualityImproved(quality, baselineQuality)
	}

	// Rung 1: strongest term. Skipped silently when strongest
	// matches the raw trimmed lowered query (no new information).
	loweredRaw := strings.ToLower(strings.TrimSpace(req.Query))
	strongest := strongestTerm(req.Query)
	if strongest != "" && strongest != loweredRaw {
		exprs := buildBM25FanoutExprs(strongest, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, false, fmt.Errorf("retrieval: rung 1 bm25: %w", err)
		}
		quality := assessBM25CandidateQuality(req.Query, req.QuestionDate, hits)
		attempts = append(attempts, Attempt{
			Rung:         1,
			Mode:         ModeBM25,
			TopK:         candidateK,
			Reason:       "strongest_term",
			Query:        expr,
			Chunks:       len(hits),
			Quality:      quality.label,
			QualityScore: quality.score,
			DateBounded:  dateBounded,
		})
		if acceptHits(hits, quality) {
			return hits, attempts, true, nil
		}
	}

	// Rung 2: refresh the backing index when the source can do so.
	// No trace row is emitted because the following rungs carry the
	// refreshed_* labels that matter to callers and eval traces.
	if err := r.refreshIndex(ctx); err != nil {
		return nil, attempts, false, fmt.Errorf("retrieval: rung 2 refresh index: %w", err)
	}

	// Rung 3: refreshed sanitised query.
	sanitised := sanitiseQuery(req.Query)
	if sanitised != "" {
		exprs := buildBM25FanoutExprs(sanitised, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, false, fmt.Errorf("retrieval: rung 3 bm25: %w", err)
		}
		quality := assessBM25CandidateQuality(req.Query, req.QuestionDate, hits)
		attempts = append(attempts, Attempt{
			Rung:         3,
			Mode:         ModeBM25,
			TopK:         candidateK,
			Reason:       "refreshed_sanitised",
			Query:        expr,
			Chunks:       len(hits),
			Quality:      quality.label,
			QualityScore: quality.score,
			DateBounded:  dateBounded,
		})
		if acceptHits(hits, quality) {
			return hits, attempts, true, nil
		}
	}

	// Rung 4: refreshed strongest term.
	if s := strongestTerm(sanitised); s != "" {
		exprs := buildBM25FanoutExprs(s, req.QuestionDate)
		hits, expr, err := r.runBM25Fanout(ctx, exprs, candidateK, req.Filters)
		if err != nil {
			return nil, attempts, false, fmt.Errorf("retrieval: rung 4 bm25: %w", err)
		}
		quality := assessBM25CandidateQuality(req.Query, req.QuestionDate, hits)
		attempts = append(attempts, Attempt{
			Rung:         4,
			Mode:         ModeBM25,
			TopK:         candidateK,
			Reason:       "refreshed_strongest",
			Query:        expr,
			Chunks:       len(hits),
			Quality:      quality.label,
			QualityScore: quality.score,
			DateBounded:  dateBounded,
		})
		if acceptHits(hits, quality) {
			return hits, attempts, true, nil
		}
	}

	// Rung 5: trigram fuzzy fallback.
	tokens := queryTokens(req.Query)
	if len(tokens) > 0 {
		idx := r.ensureTrigramIndex(ctx)
		if idx != nil {
			searchLimit := candidateK
			if req.Filters.HasAny() {
				searchLimit = max(searchLimit*10, 200)
			}
			fuzzy := idx.search(tokens, searchLimit)
			candidates := make([]rrfCandidate, 0, len(fuzzy))
			for _, h := range fuzzy {
				if !trigramHitPassesFilters(h, req.Filters) {
					continue
				}
				candidates = append(candidates, rrfCandidate{
					id:           h.ID,
					path:         h.Path,
					title:        h.Title,
					summary:      h.Summary,
					content:      h.Content,
					bm25Rank:     len(candidates),
					haveBM25Rank: true,
				})
				if len(candidates) >= candidateK {
					break
				}
			}
			quality := assessBM25CandidateQuality(req.Query, req.QuestionDate, candidates)
			attempts = append(attempts, Attempt{
				Rung:         5,
				Mode:         ModeBM25,
				TopK:         candidateK,
				Reason:       "trigram_fuzzy",
				Query:        strings.Join(tokens, " "),
				Chunks:       len(candidates),
				Quality:      quality.label,
				QualityScore: quality.score,
				DateBounded:  dateBounded,
			})
			if acceptHits(candidates, quality) {
				return candidates, attempts, true, nil
			}
		}
	}

	return nil, attempts, false, nil
}

func (r *retriever) refreshIndex(ctx context.Context) error {
	if r == nil || r.source == nil {
		return nil
	}
	source, ok := r.source.(RefreshSource)
	if !ok {
		return nil
	}
	return source.Refresh(ctx)
}

func trigramHitPassesFilters(hit trigramHit, filters Filters) bool {
	if !filters.MatchesPath(hit.Path) {
		return false
	}
	if !sessionMatchesFilter(hit.SessionID, filters.SessionIDs) {
		return false
	}
	if !scopeMatchesFilter(hit.Scope, filters.Scope) {
		return false
	}
	if !projectMatchesFilter(hit.Scope, hit.Project, filters.Project) {
		return false
	}
	if !dateMatchesFilter(hit.Session, filters) {
		return false
	}
	if len(filters.Tags) == 0 {
		return true
	}
	hitTags := tagSet(hit.Tags)
	for _, want := range filters.Tags {
		tag := normaliseTag(want)
		if tag == "" {
			continue
		}
		if !hitTags[tag] {
			return false
		}
	}
	return true
}

func mergeRetryWithBaselineCandidates(retryCandidates, baseline []rrfCandidate) []rrfCandidate {
	if len(retryCandidates) == 0 || len(baseline) == 0 {
		return retryCandidates
	}
	merged := make([]rrfCandidate, 0, len(retryCandidates)+len(baseline))
	seen := make(map[string]struct{}, len(retryCandidates)+len(baseline))
	appendUnique := func(c rrfCandidate) {
		key := pickID(c.id, c.path)
		if key == "" {
			return
		}
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		merged = append(merged, c)
	}
	for _, c := range retryCandidates {
		appendUnique(c)
	}
	for _, c := range baseline {
		appendUnique(c)
	}
	for i := range merged {
		if merged[i].haveBM25Rank {
			merged[i].bm25Rank = i
		}
	}
	return merged
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
	queries := buildVectorFanoutQueries(req.Query)
	if len(queries) == 0 {
		return nil, nil
	}
	vectors, err := r.embedder.Embed(ctx, queries)
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, nil
	}
	lists := make([][]rrfCandidate, 0, min(len(queries), len(vectors)))
	for i, vector := range vectors {
		if i >= len(queries) || len(vector) == 0 {
			continue
		}
		hits, err := r.source.SearchVector(ctx, vector, k, req.Filters)
		if err != nil {
			return nil, err
		}
		if len(hits) == 0 {
			continue
		}
		out := make([]rrfCandidate, 0, len(hits))
		for _, h := range hits {
			out = append(out, rrfCandidate{
				id:               pickID(h.ID, h.Path),
				path:             h.Path,
				title:            h.Title,
				summary:          h.Summary,
				content:          h.Content,
				vectorSimilarity: h.Similarity,
				haveVectorSim:    true,
			})
		}
		lists = append(lists, out)
	}
	switch len(lists) {
	case 0:
		return nil, nil
	case 1:
		return lists[0], nil
	default:
		fused := reciprocalRankFusion(lists, r.rrfK)
		out := make([]rrfCandidate, 0, len(fused))
		for _, h := range fused {
			out = append(out, rrfCandidate{
				id:               pickID(h.ChunkID, h.Path),
				path:             h.Path,
				title:            h.Title,
				summary:          h.Summary,
				content:          h.Text,
				vectorSimilarity: h.VectorSimilarity,
				haveVectorSim:    h.VectorSimilarity != 0,
			})
		}
		return out, nil
	}
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
		out[i] = applyIndexedRowToChunk(out[i], row)
	}
	return out
}

func (r *retriever) expandSameSessionNeighbours(ctx context.Context, req Request, chunks []RetrievedChunk, topK int, trace *Trace) []RetrievedChunk {
	if len(chunks) == 0 || len(normalisePathList(req.Filters.Paths)) > 0 || !shouldExpandSameSessionNeighbours(req.Query) {
		return chunks
	}
	neighbours, ok := r.source.(SessionNeighbourSource)
	if !ok {
		return chunks
	}
	headLimit := min(len(chunks), min(max(topK, 1), 5))
	if headLimit == 0 {
		return chunks
	}

	seeds := make([]SessionNeighbourSeed, 0, 3)
	seedBySession := make(map[string]RetrievedChunk, 3)
	for i := 0; i < headLimit; i++ {
		chunk := chunks[i]
		sessionID := chunkSessionID(chunk)
		if sessionID == "" {
			continue
		}
		if _, exists := seedBySession[sessionID]; exists {
			continue
		}
		seedBySession[sessionID] = chunk
		seeds = append(seeds, SessionNeighbourSeed{
			Path:      chunk.Path,
			SessionID: sessionID,
			Score:     chunk.Score,
			Rank:      i,
		})
		if len(seeds) >= 3 {
			break
		}
	}
	if len(seeds) == 0 {
		return chunks
	}

	maxRowsTotal := min(12, max(topK, 10))
	rows, err := neighbours.SessionNeighbours(ctx, seeds, SessionNeighbourOptions{
		MaxSessions:       len(seeds),
		MaxRowsPerSession: 3,
		MaxRowsTotal:      maxRowsTotal,
		Filters:           req.Filters,
	})
	if err != nil || len(rows) == 0 {
		return chunks
	}

	existingPaths := make(map[string]bool, len(chunks))
	signatures := make([]duplicateChunkSignature, 0, len(chunks)+len(rows))
	for _, chunk := range chunks {
		if chunk.Path != "" {
			existingPaths[chunk.Path] = true
		}
		signatures = append(signatures, duplicateSignatureForChunk(chunk))
	}

	perSession := make(map[string]int, len(seeds))
	additions := make([]RetrievedChunk, 0, len(rows))
	for _, row := range rows {
		if row.Path == "" || existingPaths[row.Path] {
			continue
		}
		text := stripFrontmatterBody(row.Content)
		if strings.TrimSpace(text) == "" {
			continue
		}
		sessionID := strings.TrimSpace(row.SessionID)
		if sessionID == "" {
			sessionID = firstFrontmatterValue(row.Content, "session_id")
		}
		seed, ok := seedBySession[sessionID]
		if !ok {
			continue
		}
		chunk := indexedRowToRetrievedChunk(row)
		chunk.Score = sameSessionNeighbourScore(seed, chunk, perSession[sessionID])
		meta := cloneChunkMetadata(chunk.Metadata)
		meta["expanded"] = true
		meta["expanded_from"] = seed.Path
		meta["expansion"] = "same_session"
		chunk.Metadata = meta

		sig := duplicateSignatureForChunk(chunk)
		if duplicateSignatureMatches(sig, signatures) {
			continue
		}
		additions = append(additions, chunk)
		signatures = append(signatures, sig)
		existingPaths[row.Path] = true
		perSession[sessionID]++
	}
	if len(additions) == 0 {
		return chunks
	}
	sort.SliceStable(additions, func(i, j int) bool {
		if additions[i].Score != additions[j].Score {
			return additions[i].Score > additions[j].Score
		}
		return additions[i].Path < additions[j].Path
	})

	combined := make([]RetrievedChunk, 0, len(chunks)+len(additions))
	combined = append(combined, chunks...)
	combined = append(combined, additions...)
	sort.SliceStable(combined, func(i, j int) bool {
		if combined[i].Score != combined[j].Score {
			return combined[i].Score > combined[j].Score
		}
		return combined[i].Path < combined[j].Path
	})
	trace.SessionExpansions = len(additions)
	return combined
}

func (r *retriever) expandEpisodicRecall(ctx context.Context, req Request, chunks []RetrievedChunk, topK int, trace *Trace) []RetrievedChunk {
	profile := detectEpisodicRecallProfile(req.Query)
	if !profile.enabled || !episodicRecallFiltersAllowed(req.Filters) || episodicRecallSatisfiedByMemory(profile, chunks) {
		return chunks
	}

	exprs := buildEpisodicRecallExprs(req.Query, req.QuestionDate, profile)
	if len(exprs) == 0 {
		return chunks
	}
	rawFilters := Filters{
		Scope:    "raw",
		DateFrom: req.Filters.DateFrom,
		DateTo:   req.Filters.DateTo,
	}
	hits, err := r.runEpisodicRecallBM25(ctx, exprs, 32, rawFilters)
	if err != nil || len(hits) == 0 {
		return chunks
	}

	rawChunks := r.hydrateBodies(ctx, singleList(hits, r.rrfK))
	existingPaths := make(map[string]bool, len(chunks))
	signatures := make([]duplicateChunkSignature, 0, len(chunks)+len(rawChunks))
	for _, chunk := range chunks {
		if chunk.Path != "" {
			existingPaths[chunk.Path] = true
		}
		signatures = append(signatures, duplicateSignatureForChunk(chunk))
	}

	topScore := 1.0 / float64(max(r.rrfK, 1)+1)
	if len(chunks) > 0 && chunks[0].Score > 0 {
		topScore = chunks[0].Score
	}
	type recallCandidate struct {
		chunk   RetrievedChunk
		snippet string
		quality int
		index   int
	}
	candidates := make([]recallCandidate, 0, min(8, len(rawChunks)))
	for _, chunk := range rawChunks {
		if chunk.Path == "" || existingPaths[chunk.Path] {
			continue
		}
		snippet, ok := episodicRecallSnippet(profile, chunk.Text)
		if !ok {
			continue
		}
		candidates = append(candidates, recallCandidate{
			chunk:   chunk,
			snippet: snippet,
			quality: episodicRecallCandidateQuality(profile, chunk.Text, snippet),
			index:   len(candidates),
		})
		if len(candidates) >= 8 {
			break
		}
	}
	if len(candidates) == 0 {
		return chunks
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].quality != candidates[j].quality {
			return candidates[i].quality > candidates[j].quality
		}
		if candidates[i].chunk.Score != candidates[j].chunk.Score {
			return candidates[i].chunk.Score > candidates[j].chunk.Score
		}
		return candidates[i].index < candidates[j].index
	})

	additions := make([]RetrievedChunk, 0, min(3, len(candidates)))
	for _, candidate := range candidates {
		chunk := candidate.chunk
		chunk.Text = candidate.snippet
		chunk.Score = topScore * episodicRecallScoreMultiplier(profile, len(additions))
		meta := cloneChunkMetadata(chunk.Metadata)
		meta["expanded"] = true
		meta["expansion"] = "episodic_recall"
		meta["recall_scope"] = "raw"
		meta["recall_reason"] = profile.reason
		chunk.Metadata = meta

		sig := duplicateSignatureForChunk(chunk)
		if duplicateSignatureMatches(sig, signatures) {
			continue
		}
		additions = append(additions, chunk)
		signatures = append(signatures, sig)
		existingPaths[chunk.Path] = true
		if len(additions) >= 3 {
			break
		}
	}
	if len(additions) == 0 {
		return chunks
	}

	combined := make([]RetrievedChunk, 0, len(chunks)+len(additions))
	combined = append(combined, chunks...)
	combined = append(combined, additions...)
	sort.SliceStable(combined, func(i, j int) bool {
		if combined[i].Score != combined[j].Score {
			return combined[i].Score > combined[j].Score
		}
		return combined[i].Path < combined[j].Path
	})
	trace.EpisodicRecall = true
	trace.EpisodicRecallHits = len(additions)
	trace.EpisodicRecallReason = profile.reason
	return combined
}

func (r *retriever) runEpisodicRecallBM25(ctx context.Context, exprs []string, k int, filters Filters) ([]rrfCandidate, error) {
	lists := make([][]rrfCandidate, 0, len(exprs))
	for _, expr := range exprs {
		hits, err := r.runBM25(ctx, expr, k, filters)
		if err != nil {
			return nil, err
		}
		if len(hits) > 0 {
			lists = append(lists, hits)
		}
	}
	switch len(lists) {
	case 0:
		return nil, nil
	case 1:
		return lists[0], nil
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
		return out, nil
	}
}

type episodicRecallProfile struct {
	enabled         bool
	reason          string
	ordinal         int
	artifactCue     bool
	sectionTerms    []string
	artifactTerms   []string
	descriptorTerms []string
}

func detectEpisodicRecallProfile(query string) episodicRecallProfile {
	normalised := strings.ToLower(query)
	if !hasConversationRecallCue(normalised) {
		return episodicRecallProfile{}
	}
	profile := episodicRecallProfile{
		ordinal:         queryOrdinal(normalised),
		sectionTerms:    querySectionTerms(normalised),
		artifactTerms:   queryArtifactTerms(normalised),
		descriptorTerms: queryDescriptorTerms(normalised),
	}
	profile.artifactCue = len(profile.artifactTerms) > 0 ||
		strings.Contains(normalised, "you created") ||
		strings.Contains(normalised, "you generated") ||
		strings.Contains(normalised, "you wrote") ||
		strings.Contains(normalised, "you drafted") ||
		strings.Contains(normalised, "you gave me") ||
		strings.Contains(normalised, "you made")
	if !profile.artifactCue && len(profile.sectionTerms) == 0 {
		return episodicRecallProfile{}
	}
	profile.enabled = true
	switch {
	case profile.ordinal > 0 && len(profile.sectionTerms) > 0:
		profile.reason = "ordinal_section_artifact"
	case profile.ordinal > 0:
		profile.reason = "ordinal_artifact"
	case len(profile.sectionTerms) > 0:
		profile.reason = "section_artifact"
	default:
		profile.reason = "artifact"
	}
	return profile
}

func episodicRecallFiltersAllowed(filters Filters) bool {
	if strings.TrimSpace(filters.PathPrefix) != "" || len(normalisePathList(filters.Paths)) > 0 || len(filters.Tags) > 0 {
		return false
	}
	scope := strings.ToLower(strings.TrimSpace(filters.Scope))
	return scope == "" || scope == "memory"
}

func episodicRecallSatisfiedByMemory(profile episodicRecallProfile, chunks []RetrievedChunk) bool {
	if len(chunks) == 0 {
		return false
	}
	limit := min(len(chunks), 8)
	for i := 0; i < limit; i++ {
		text := strings.ToLower(retrievalResultText(chunks[i]))
		if text == "" {
			continue
		}
		if profile.ordinal > 0 && !containsOrdinalMarker(text, profile.ordinal) {
			continue
		}
		if len(profile.sectionTerms) > 0 && !containsAnyTerm(text, profile.sectionTerms) {
			continue
		}
		if len(profile.artifactTerms) > 0 && !containsAnyTerm(text, profile.artifactTerms) {
			continue
		}
		return true
	}
	return false
}

func buildEpisodicRecallExprs(query, questionDate string, profile episodicRecallProfile) []string {
	queries := buildBM25FanoutQueries(query, questionDate)
	if len(profile.artifactTerms) > 0 || len(profile.sectionTerms) > 0 {
		queries = append(queries, joinLimitedTerms(appendUniqueStrings(profile.artifactTerms, profile.sectionTerms), 4))
	}
	if len(profile.descriptorTerms) > 0 && (len(profile.artifactTerms) > 0 || len(profile.sectionTerms) > 0) {
		queries = append(queries, joinLimitedTerms(appendUniqueStrings(profile.descriptorTerms, appendUniqueStrings(profile.artifactTerms, profile.sectionTerms)), 4))
		queries = append(queries, joinLimitedTerms(appendUniqueStrings(profile.descriptorTerms, profile.artifactTerms), 4))
	}
	if len(profile.sectionTerms) > 0 {
		queries = append(queries, joinLimitedTerms(profile.sectionTerms, 2))
	}
	queries = dedupeTrimmedStrings(queries)

	phraseProbes := derivePhraseProbes(query)
	exprs := make([]string, 0, len(queries))
	seen := make(map[string]bool, len(queries))
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

func episodicRecallScoreMultiplier(profile episodicRecallProfile, offset int) float64 {
	base := 0.94
	if profile.ordinal > 0 && len(profile.sectionTerms) > 0 {
		base = 0.985
	} else if profile.ordinal > 0 || len(profile.sectionTerms) > 0 {
		base = 0.96
	}
	if offset <= 0 {
		return base
	}
	decay := base - float64(offset)*0.08
	if decay < 0.72 {
		return 0.72
	}
	return decay
}

func episodicRecallCandidateQuality(profile episodicRecallProfile, raw, snippet string) int {
	rawLower := strings.ToLower(raw)
	snippetLower := strings.ToLower(snippet)
	quality := 0
	if len(profile.sectionTerms) > 0 && containsAnyTerm(snippetLower, profile.sectionTerms) {
		quality += 10
	}
	if len(profile.descriptorTerms) > 0 && containsAnyTerm(rawLower, profile.descriptorTerms) {
		quality += 7
	}
	if len(profile.artifactTerms) > 0 && containsAnyTerm(rawLower, profile.artifactTerms) {
		quality += 3
	}
	if profile.ordinal > 0 && len(rawMessageBlocks(raw)) >= profile.ordinal*2 {
		quality += 4
	}
	if len([]rune(snippet)) <= 1200 {
		quality += 2
	}
	return quality
}

type rawMessageBlock struct {
	role string
	text string
}

func episodicRecallSnippet(profile episodicRecallProfile, raw string) (string, bool) {
	body := stripFrontmatterBody(raw)
	if strings.TrimSpace(body) == "" {
		return "", false
	}
	blocks := rawMessageBlocks(body)
	if len(blocks) > 0 {
		if snippet, ok := episodicRecallSnippetFromBlocks(profile, blocks); ok {
			return snippet, true
		}
	}
	return episodicRecallSnippetFromText(profile, body)
}

func episodicRecallSnippetFromBlocks(profile episodicRecallProfile, blocks []rawMessageBlock) (string, bool) {
	candidates := make([]rawMessageBlock, 0, len(blocks))
	for _, block := range blocks {
		if block.role != "assistant" {
			continue
		}
		lower := strings.ToLower(block.text)
		if len(profile.artifactTerms) > 0 && !containsAnyTerm(lower, profile.artifactTerms) && !containsAnyTerm(lower, profile.sectionTerms) {
			continue
		}
		candidates = append(candidates, block)
	}
	if len(candidates) == 0 {
		return "", false
	}
	index := 0
	if profile.ordinal > 0 {
		if len(candidates) < profile.ordinal {
			return "", false
		}
		index = profile.ordinal - 1
	}
	block := candidates[index]
	if snippet, ok := extractLabelledSection(profile, block.text); ok {
		return labelEpisodicRecallSnippet(profile, snippet), true
	}
	return labelEpisodicRecallSnippet(profile, limitRecallSnippet(strings.TrimSpace(block.text))), true
}

func episodicRecallSnippetFromText(profile episodicRecallProfile, text string) (string, bool) {
	if snippet, ok := extractLabelledSection(profile, text); ok {
		return snippet, true
	}
	lines := splitNonEmptyLines(text)
	if len(lines) == 0 {
		return "", false
	}
	best := -1
	bestScore := 0
	for i, line := range lines {
		lower := strings.ToLower(line)
		score := 0
		if containsAnyTerm(lower, profile.sectionTerms) {
			score += 5
		}
		if containsAnyTerm(lower, profile.artifactTerms) {
			score += 3
		}
		if profile.ordinal > 0 && containsOrdinalMarker(lower, profile.ordinal) {
			score += 4
		}
		if score > bestScore {
			best = i
			bestScore = score
		}
	}
	if best < 0 {
		return "", false
	}
	start := max(0, best-4)
	end := min(len(lines), best+12)
	return limitRecallSnippet(strings.Join(lines[start:end], "\n")), true
}

func rawMessageBlocks(text string) []rawMessageBlock {
	lines := strings.Split(text, "\n")
	blocks := make([]rawMessageBlock, 0)
	currentRole := ""
	var current []string
	flush := func() {
		if currentRole == "" || len(current) == 0 {
			current = nil
			return
		}
		body := strings.TrimSpace(strings.Join(current, "\n"))
		if body != "" {
			blocks = append(blocks, rawMessageBlock{role: currentRole, text: body})
		}
		current = nil
	}
	for _, line := range lines {
		match := rawMessageMarkerRe.FindStringSubmatch(line)
		if len(match) == 3 {
			flush()
			currentRole = strings.ToLower(strings.TrimSpace(match[1]))
			if strings.TrimSpace(match[2]) != "" {
				current = append(current, strings.TrimSpace(match[2]))
			}
			continue
		}
		if currentRole != "" {
			current = append(current, line)
		}
	}
	flush()
	return blocks
}

func extractLabelledSection(profile episodicRecallProfile, text string) (string, bool) {
	if len(profile.sectionTerms) == 0 {
		return "", false
	}
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if !lineMatchesSectionTerm(line, profile.sectionTerms) {
			continue
		}
		end := i + 1
		for end < len(lines) {
			if end > i+1 && rawSectionHeadingRe.MatchString(lines[end]) {
				break
			}
			if rawMessageMarkerRe.MatchString(lines[end]) {
				break
			}
			end++
		}
		return limitRecallSnippet(strings.TrimSpace(strings.Join(lines[i:end], "\n"))), true
	}
	return "", false
}

func lineMatchesSectionTerm(line string, terms []string) bool {
	lower := strings.ToLower(strings.TrimSpace(line))
	if lower == "" {
		return false
	}
	for _, term := range terms {
		term = strings.ToLower(strings.TrimSpace(term))
		if term == "" {
			continue
		}
		if strings.HasPrefix(lower, term+":") || strings.HasPrefix(lower, term+" ") || strings.Contains(lower, term+":") {
			return true
		}
	}
	return false
}

func splitNonEmptyLines(text string) []string {
	raw := strings.Split(text, "\n")
	lines := make([]string, 0, len(raw))
	for _, line := range raw {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			lines = append(lines, trimmed)
		}
	}
	return lines
}

func limitRecallSnippet(text string) string {
	const maxRunes = 1600
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	runes := []rune(trimmed)
	if len(runes) <= maxRunes {
		return trimmed
	}
	return strings.TrimSpace(string(runes[:maxRunes]))
}

func labelEpisodicRecallSnippet(profile episodicRecallProfile, snippet string) string {
	trimmed := strings.TrimSpace(snippet)
	if trimmed == "" || profile.ordinal <= 0 {
		return trimmed
	}
	return fmt.Sprintf("Assistant artefact %d:\n%s", profile.ordinal, trimmed)
}

func queryOrdinal(normalised string) int {
	padded := " " + strings.TrimSpace(normalised) + " "
	switch {
	case strings.Contains(padded, " first ") || strings.Contains(padded, " 1st "):
		return 1
	case strings.Contains(padded, " second ") || strings.Contains(padded, " 2nd "):
		return 2
	case strings.Contains(padded, " third ") || strings.Contains(padded, " 3rd "):
		return 3
	case strings.Contains(padded, " fourth ") || strings.Contains(padded, " 4th "):
		return 4
	default:
		return 0
	}
}

func querySectionTerms(normalised string) []string {
	candidates := []string{
		"chorus", "verse", "bridge", "outro", "chapter", "section", "ingredients",
		"instructions", "steps", "code", "subject", "table", "row", "option",
	}
	out := make([]string, 0, 3)
	for _, candidate := range candidates {
		if strings.Contains(normalised, candidate) {
			out = append(out, candidate)
		}
	}
	return out
}

func queryArtifactTerms(normalised string) []string {
	candidates := []string{
		"song", "songs", "poem", "poems", "lyrics", "chord", "chords", "progression",
		"progressions", "notes", "recipe", "recipes", "itinerary", "schedule", "outline",
		"email", "letter", "draft", "table", "code", "answer", "response", "option",
	}
	out := make([]string, 0, 4)
	for _, candidate := range candidates {
		if strings.Contains(normalised, candidate) {
			out = append(out, strings.TrimSuffix(candidate, "s"))
		}
	}
	return dedupeTrimmedStrings(out)
}

func queryDescriptorTerms(normalised string) []string {
	tokens := phraseProbeTokens(normalised)
	if len(tokens) == 0 {
		return nil
	}
	artifactTerms := queryArtifactTerms(normalised)
	sectionTerms := querySectionTerms(normalised)
	out := make([]string, 0, 3)
	for _, token := range tokens {
		if len(token) < 3 ||
			questionTokenStopWords[token] ||
			phraseProbeBoundaryWords[token] ||
			coverageFacetSkipWords[token] ||
			episodicDescriptorSkipWords[token] ||
			containsDigit(token) ||
			containsAnyTerm(token, artifactTerms) ||
			containsAnyTerm(token, sectionTerms) ||
			containsOrdinalMarker(token, queryOrdinal(normalised)) {
			continue
		}
		out = append(out, token)
		if len(out) == 3 {
			break
		}
	}
	return dedupeTrimmedStrings(out)
}

func containsOrdinalMarker(text string, ordinal int) bool {
	switch ordinal {
	case 1:
		return strings.Contains(text, "first") || strings.Contains(text, "1st") || strings.Contains(text, "artefact 1") || strings.Contains(text, "artifact 1") || strings.Contains(text, "option 1")
	case 2:
		return strings.Contains(text, "second") || strings.Contains(text, "2nd") || strings.Contains(text, "artefact 2") || strings.Contains(text, "artifact 2") || strings.Contains(text, "option 2")
	case 3:
		return strings.Contains(text, "third") || strings.Contains(text, "3rd") || strings.Contains(text, "artefact 3") || strings.Contains(text, "artifact 3") || strings.Contains(text, "option 3")
	case 4:
		return strings.Contains(text, "fourth") || strings.Contains(text, "4th") || strings.Contains(text, "artefact 4") || strings.Contains(text, "artifact 4") || strings.Contains(text, "option 4")
	default:
		return false
	}
}

func containsAnyTerm(text string, terms []string) bool {
	if text == "" || len(terms) == 0 {
		return false
	}
	for _, term := range terms {
		term = strings.ToLower(strings.TrimSpace(term))
		if term != "" && strings.Contains(text, term) {
			return true
		}
	}
	return false
}

func appendUniqueStrings(base []string, extra []string) []string {
	seen := make(map[string]bool, len(base)+len(extra))
	out := make([]string, 0, len(base)+len(extra))
	for _, values := range [][]string{base, extra} {
		for _, value := range values {
			trimmed := strings.TrimSpace(value)
			if trimmed == "" || seen[trimmed] {
				continue
			}
			seen[trimmed] = true
			out = append(out, trimmed)
		}
	}
	return out
}

func joinLimitedTerms(terms []string, limit int) string {
	if limit <= 0 || len(terms) == 0 {
		return ""
	}
	out := make([]string, 0, min(limit, len(terms)))
	for _, term := range terms {
		trimmed := strings.TrimSpace(term)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
		if len(out) == limit {
			break
		}
	}
	return strings.Join(out, " ")
}

func shouldExpandSameSessionNeighbours(query string) bool {
	normalised := strings.ToLower(query)
	if normalised == "" {
		return false
	}
	if isExactListRecallQuery(normalised) {
		return false
	}
	return hasConversationRecallCue(normalised) || isContextualNeighbourExpansionQuery(query, normalised)
}

func isExactListRecallQuery(normalised string) bool {
	return exactListRecallRe.MatchString(normalised)
}

func isContextualNeighbourExpansionQuery(query, normalised string) bool {
	if isTypeContextCountQuery(query) {
		return true
	}
	if strings.Contains(normalised, "when did") && len(deriveActionDateProbes(query)) > 0 {
		return true
	}
	return false
}

func hasConversationRecallCue(normalised string) bool {
	recallCues := []string{
		"previous chat",
		"previous conversation",
		"last conversation",
		"our last chat",
		"last time",
		"looking back",
		"going through our previous",
		"we talked about",
		"we discussed",
		"follow up on our previous",
	}
	for _, cue := range recallCues {
		if strings.Contains(normalised, cue) {
			return true
		}
	}
	if strings.Contains(normalised, "remind me") {
		return strings.Contains(normalised, "previous") ||
			strings.Contains(normalised, "last") ||
			strings.Contains(normalised, "earlier")
	}
	return false
}

func sameSessionNeighbourScore(seed, neighbour RetrievedChunk, sessionOffset int) float64 {
	multiplier := 0.72
	if sessionOffset > 0 {
		multiplier = 0.55
	}
	if seedRole := chunkSourceRole(seed); seedRole != "" && seedRole == chunkSourceRole(neighbour) {
		multiplier *= 1.05
	}
	if sameChunkEvidenceDate(seed, neighbour) {
		multiplier *= 1.08
	}
	score := seed.Score * multiplier
	maxScore := seed.Score * 0.95
	if score > maxScore {
		return maxScore
	}
	return score
}

func sameChunkEvidenceDate(a, b RetrievedChunk) bool {
	aDate := firstChunkMetadataString(a, "event_date", "eventDate", "observed_on", "observedOn", "session_date", "sessionDate")
	bDate := firstChunkMetadataString(b, "event_date", "eventDate", "observed_on", "observedOn", "session_date", "sessionDate")
	return aDate != "" && aDate == bDate
}

func firstChunkMetadataString(chunk RetrievedChunk, keys ...string) string {
	for _, key := range keys {
		if value, ok := chunk.Metadata[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func indexedRowToRetrievedChunk(row search.IndexedRow) RetrievedChunk {
	chunk := RetrievedChunk{
		ChunkID:    row.Path,
		DocumentID: row.Path,
		Path:       row.Path,
		Text:       stripFrontmatterBody(row.Content),
		Title:      row.Title,
		Summary:    row.Summary,
	}
	return applyIndexedRowToChunk(chunk, row)
}

func applyIndexedRowToChunk(chunk RetrievedChunk, row search.IndexedRow) RetrievedChunk {
	if row.Content != "" {
		chunk.Text = stripFrontmatterBody(row.Content)
	}
	if chunk.Title == "" && row.Title != "" {
		chunk.Title = row.Title
	}
	if chunk.Summary == "" && row.Summary != "" {
		chunk.Summary = row.Summary
	}
	if row.Scope == "" &&
		row.ProjectSlug == "" &&
		row.SessionDate == "" &&
		row.SessionID == "" &&
		row.ObservedOn == "" &&
		row.Modified == "" &&
		row.SourceRole == "" &&
		row.EventDate == "" &&
		row.EvidenceKind == "" &&
		row.EvidenceGroup == "" &&
		row.StateKey == "" &&
		row.StateKind == "" &&
		row.StateSubject == "" &&
		row.StateValue == "" &&
		row.ClaimKey == "" &&
		row.ClaimStatus == "" &&
		row.ValidFrom == "" &&
		row.ValidTo == "" &&
		row.ArtefactType == "" &&
		row.ArtefactOrdinal == "" &&
		row.ArtefactSection == "" &&
		row.ArtefactDescriptor == "" &&
		row.Content == "" {
		return chunk
	}
	meta := cloneChunkMetadata(chunk.Metadata)
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
	sessionID := row.SessionID
	if sessionID == "" {
		sessionID = firstFrontmatterValue(row.Content, "session_id")
	}
	if sessionID != "" {
		meta["sessionId"] = sessionID
		meta["session_id"] = sessionID
	}
	observedOn := row.ObservedOn
	if observedOn == "" {
		observedOn = firstFrontmatterValue(row.Content, "observed_on")
	}
	if observedOn != "" {
		meta["observedOn"] = observedOn
		meta["observed_on"] = observedOn
	}
	modified := row.Modified
	if modified == "" {
		modified = firstFrontmatterValue(row.Content, "modified")
	}
	if modified != "" {
		meta["modified"] = modified
	}
	sourceRole := row.SourceRole
	if sourceRole == "" {
		sourceRole = firstFrontmatterValue(row.Content, "source_role")
	}
	if sourceRole != "" {
		meta["sourceRole"] = sourceRole
		meta["source_role"] = sourceRole
	}
	eventDate := row.EventDate
	if eventDate == "" {
		eventDate = firstFrontmatterValue(row.Content, "event_date")
	}
	if eventDate != "" {
		meta["eventDate"] = eventDate
		meta["event_date"] = eventDate
	}
	applyMetadataAlias(meta, row.EvidenceKind, "evidenceKind", "evidence_kind")
	applyMetadataAlias(meta, row.EvidenceGroup, "evidenceGroup", "evidence_group")
	applyMetadataAlias(meta, row.StateKey, "stateKey", "state_key")
	applyMetadataAlias(meta, row.StateKind, "stateKind", "state_kind")
	applyMetadataAlias(meta, row.StateSubject, "stateSubject", "state_subject")
	applyMetadataAlias(meta, row.StateValue, "stateValue", "state_value")
	applyMetadataAlias(meta, row.ClaimKey, "claimKey", "claim_key")
	applyMetadataAlias(meta, row.ClaimStatus, "claimStatus", "claim_status")
	applyMetadataAlias(meta, row.ValidFrom, "validFrom", "valid_from")
	applyMetadataAlias(meta, row.ValidTo, "validTo", "valid_to")
	applyMetadataAlias(meta, row.ArtefactType, "artefactType", "artefact_type")
	applyMetadataAlias(meta, row.ArtefactOrdinal, "artefactOrdinal", "artefact_ordinal")
	applyMetadataAlias(meta, row.ArtefactSection, "artefactSection", "artefact_section")
	applyMetadataAlias(meta, row.ArtefactDescriptor, "artefactDescriptor", "artefact_descriptor")
	if len(meta) > 0 {
		chunk.Metadata = meta
	} else {
		chunk.Metadata = nil
	}
	return chunk
}

func applyMetadataAlias(meta map[string]any, value string, keys ...string) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	for _, key := range keys {
		meta[key] = value
	}
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

type duplicateChunkSignature struct {
	sessionID string
	exactKey  string
	tokens    map[string]struct{}
}

func dedupeNearDuplicateChunks(chunks []RetrievedChunk) []RetrievedChunk {
	if len(chunks) <= 1 {
		return chunks
	}
	out := make([]RetrievedChunk, 0, len(chunks))
	signatures := make([]duplicateChunkSignature, 0, len(chunks))
	for _, chunk := range chunks {
		sig := duplicateSignatureForChunk(chunk)
		if duplicateSignatureMatches(sig, signatures) {
			continue
		}
		out = append(out, chunk)
		signatures = append(signatures, sig)
	}
	return out
}

func duplicateSignatureMatches(candidate duplicateChunkSignature, existing []duplicateChunkSignature) bool {
	for _, current := range existing {
		sameSession := candidate.sessionID != "" && current.sessionID != "" && candidate.sessionID == current.sessionID
		if sameSession {
			if candidate.exactKey != "" && candidate.exactKey == current.exactKey {
				return true
			}
			if nearDuplicateTokenSets(candidate.tokens, current.tokens) {
				return true
			}
			continue
		}
		if candidate.sessionID == "" && current.sessionID == "" && candidate.exactKey != "" && candidate.exactKey == current.exactKey {
			return true
		}
	}
	return false
}

func duplicateSignatureForChunk(chunk RetrievedChunk) duplicateChunkSignature {
	text := chunk.Text
	if text == "" {
		text = strings.Join([]string{chunk.Title, chunk.Summary}, "\n")
	}
	text = duplicateComparableText(text)
	return duplicateChunkSignature{
		sessionID: chunkSessionID(chunk),
		exactKey:  normaliseDuplicateExactText(text),
		tokens:    duplicateTokenSet(text),
	}
}

func duplicateComparableText(text string) string {
	text = stripFrontmatterBody(text)
	trimmed := strings.TrimSpace(text)
	for strings.HasPrefix(trimmed, "[") {
		end := strings.Index(trimmed, "]")
		if end <= 0 || end > 180 {
			break
		}
		label := strings.ToLower(trimmed[1:end])
		if !strings.Contains(label, "date") &&
			!strings.Contains(label, "observed") &&
			!strings.Contains(label, "modified") &&
			!strings.Contains(label, "source") &&
			!strings.Contains(label, "event") {
			break
		}
		trimmed = strings.TrimSpace(trimmed[end+1:])
	}
	return trimmed
}

func chunkSessionID(chunk RetrievedChunk) string {
	for _, key := range []string{"session_id", "sessionId"} {
		if value, ok := chunk.Metadata[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	if sessionID := firstFrontmatterValue(chunk.Text, "session_id"); sessionID != "" {
		return sessionID
	}
	return ""
}

func normaliseDuplicateExactText(text string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(text))), " ")
}

func duplicateTokenSet(text string) map[string]struct{} {
	out := make(map[string]struct{})
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		token := b.String()
		b.Reset()
		if len(token) < 3 || duplicateTokenStopWords[token] {
			return
		}
		out[token] = struct{}{}
	}
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return out
}

var duplicateTokenStopWords = map[string]bool{
	"and": true, "are": true, "but": true, "for": true, "from": true,
	"has": true, "have": true, "how": true, "into": true, "not": true,
	"now": true, "that": true, "the": true, "their": true, "there": true,
	"this": true, "was": true, "were": true, "what": true, "when": true,
	"where": true, "which": true, "why": true, "with": true, "you": true,
	"your": true, "user": true, "assistant": true, "observed": true,
	"date": true, "fact": true, "facts": true, "session": true,
}

func nearDuplicateTokenSets(a, b map[string]struct{}) bool {
	if len(a) < 6 || len(b) < 6 {
		return false
	}
	intersection := 0
	for token := range a {
		if _, ok := b[token]; ok {
			intersection++
		}
	}
	if intersection < 5 {
		return false
	}
	smaller := min(len(a), len(b))
	larger := max(len(a), len(b))
	containment := float64(intersection) / float64(smaller)
	jaccard := float64(intersection) / float64(larger+smaller-intersection)
	return containment >= 0.84 && jaccard >= 0.7
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
	priorityQueries := buildPriorityBM25Queries(trimmed)
	coverageQueries := deriveCoverageFacetQueries(trimmed)
	augmented := augmentQueryWithTemporal(trimmed, questionDate)
	if shouldUsePriorityOnlyBM25(trimmed) && len(priorityQueries) >= 2 {
		queries := []string{trimmed, augmented}
		queries = append(queries, coverageQueries...)
		queries = append(queries, priorityQueries...)
		queries = append(queries, filteredPhraseProbes(trimmed)...)
		queries = dedupeTrimmedStrings(queries)
		return limitBM25FanoutQueries(queries, trimmed, augmented)
	}
	queries := []string{trimmed, augmented}
	queries = append(queries, coverageQueries...)
	queries = append(queries, priorityQueries...)
	for _, sub := range deriveSubQueries(trimmed) {
		queries = append(queries, sub)
	}
	queries = dedupeTrimmedStrings(queries)
	return limitBM25FanoutQueries(queries, trimmed, augmented)
}

func buildVectorFanoutQueries(raw string) []string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return nil
	}
	queries := []string{trimmed}
	queries = append(queries, deriveTypeContextProbes(trimmed)...)
	queries = append(queries, deriveCoverageFacetQueries(trimmed)...)
	for _, phrase := range filteredPhraseProbes(trimmed) {
		queries = append(queries, phrase)
	}
	queries = dedupeTrimmedStrings(queries)
	if len(queries) > maxVectorFanoutQueries {
		return queries[:maxVectorFanoutQueries]
	}
	return queries
}

func limitBM25FanoutQueries(queries []string, required ...string) []string {
	if len(queries) > maxBM25FanoutQueries {
		kept := make([]string, 0, maxBM25FanoutQueries)
		seen := make(map[string]bool, maxBM25FanoutQueries)
		for _, value := range required {
			trimmed := strings.TrimSpace(value)
			if trimmed == "" || seen[trimmed] {
				continue
			}
			seen[trimmed] = true
			kept = append(kept, trimmed)
			if len(kept) == maxBM25FanoutQueries {
				return kept
			}
		}
		for _, value := range queries {
			trimmed := strings.TrimSpace(value)
			if trimmed == "" || seen[trimmed] {
				continue
			}
			seen[trimmed] = true
			kept = append(kept, trimmed)
			if len(kept) == maxBM25FanoutQueries {
				return kept
			}
		}
		return kept
	}
	return queries
}

func buildPriorityBM25Queries(question string) []string {
	queries := make([]string, 0, maxBM25FanoutQueries)
	queries = append(queries, deriveTypeContextProbes(question)...)
	queries = append(queries, deriveDateArithmeticProbes(question)...)
	queries = append(queries, deriveCountTotalProbes(question)...)
	queries = append(queries, derivePrioritySubQueries(question)...)
	queries = append(queries, deriveRecallTitleProbes(question)...)
	if len(queries) == 0 {
		return nil
	}
	queries = dedupeTrimmedStrings(queries)
	if len(queries) > maxBM25FanoutQueries {
		return queries[:maxBM25FanoutQueries]
	}
	return queries
}

func shouldUsePriorityOnlyBM25(question string) bool {
	return len(deriveActionDateContextProbes(question)) > 0
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

func deriveCoverageFacetQueries(question string) []string {
	trimmed := strings.TrimSpace(question)
	if trimmed == "" || !shouldDeriveCoverageFacets(trimmed) {
		return nil
	}
	out := make([]string, 0, maxCoverageFacetQueries)
	seen := map[string]bool{strings.ToLower(trimmed): true}
	add := func(candidate string) bool {
		candidate = normaliseCoverageFacet(candidate)
		if candidate == "" || seen[candidate] {
			return false
		}
		seen[candidate] = true
		out = append(out, candidate)
		return len(out) == maxCoverageFacetQueries
	}
	for _, phrase := range quotedCoverageFacets(trimmed) {
		if add(phrase) {
			return out
		}
	}
	for _, phrase := range capitalisedCoverageFacets(trimmed) {
		if add(phrase) {
			return out
		}
	}
	for _, phrase := range segmentedCoverageFacets(trimmed) {
		if add(phrase) {
			return out
		}
	}
	return out
}

func shouldDeriveCoverageFacets(question string) bool {
	lowered := strings.ToLower(question)
	return strings.Contains(lowered, " and ") ||
		strings.Contains(lowered, " or ") ||
		strings.Contains(lowered, " including ") ||
		strings.Contains(lowered, " between ") ||
		strings.Contains(lowered, " from ") ||
		strings.Contains(question, ",") ||
		strings.Contains(question, "'") ||
		strings.Contains(question, `"`)
}

func quotedCoverageFacets(question string) []string {
	matches := regexp.MustCompile(`["'“”‘’]([^"'“”‘’]{2,80})["'“”‘’]`).FindAllStringSubmatch(question, -1)
	if len(matches) == 0 {
		return nil
	}
	out := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		out = append(out, match[1])
	}
	return out
}

func capitalisedCoverageFacets(question string) []string {
	spans := regexp.MustCompile(`\b[A-Za-z0-9][A-Za-z0-9'’.-]*\b`).FindAllStringIndex(question, -1)
	if len(spans) == 0 {
		return nil
	}
	out := make([]string, 0, maxCoverageFacetQueries)
	for i := 0; i < len(spans); i++ {
		start, end := spans[i][0], spans[i][1]
		word := question[start:end]
		startToken := normaliseCoverageToken(word)
		if !startsWithUppercaseLetter(word) ||
			phraseProbeBoundaryWords[startToken] ||
			questionTokenStopWords[startToken] ||
			coverageFacetSkipWords[startToken] {
			continue
		}
		tokens := []string{word}
		j := i + 1
		for ; j < len(spans); j++ {
			next := spans[j]
			rawToken := question[next[0]:next[1]]
			token := normaliseCoverageToken(rawToken)
			if phraseProbeConnectors[token] || phraseProbeBoundaryWords[token] || questionTokenStopWords[token] {
				if capitalisedFacetGlueWords[token] && nextSpanStartsUppercase(question, spans, j) {
					tokens = append(tokens, rawToken)
					continue
				}
				break
			}
			if startsWithUppercaseLetter(rawToken) {
				tokens = append(tokens, rawToken)
				continue
			}
			if len(token) >= 3 && !coverageFacetSkipWords[token] && !containsDigit(token) {
				tokens = append(tokens, rawToken)
			}
			break
		}
		i = j - 1
		phrase := strings.Join(tokens, " ")
		if phrase == "" || isLowSignalCapitalisedFacet(phrase) {
			continue
		}
		out = append(out, phrase)
		if len(out) == maxCoverageFacetQueries {
			return out
		}
	}
	return out
}

func startsWithUppercaseLetter(value string) bool {
	for _, r := range value {
		return unicode.IsUpper(r)
	}
	return false
}

func nextSpanStartsUppercase(question string, spans [][]int, current int) bool {
	next := current + 1
	if next >= len(spans) {
		return false
	}
	start, end := spans[next][0], spans[next][1]
	return startsWithUppercaseLetter(question[start:end])
}

func isLowSignalCapitalisedFacet(phrase string) bool {
	tokens := phraseProbeTokens(phrase)
	if len(tokens) == 0 {
		return true
	}
	for _, token := range tokens {
		if !phraseProbeBoundaryWords[token] && !questionTokenStopWords[token] && !coverageFacetSkipWords[token] {
			return false
		}
	}
	return true
}

func nextCoverageTailToken(raw string) string {
	tokens := phraseProbeTokens(raw)
	for _, token := range tokens {
		if phraseProbeBoundaryWords[token] || phraseProbeConnectors[token] || coverageFacetSkipWords[token] {
			if token == "and" || token == "or" {
				return ""
			}
			continue
		}
		if len(token) < 3 || containsDigit(token) {
			return ""
		}
		return token
	}
	return ""
}

func segmentedCoverageFacets(question string) []string {
	parts := regexp.MustCompile(`(?i)\b(?:and|or|including|between|from|to)\b|[,;:]`).Split(question, -1)
	if len(parts) <= 1 {
		return nil
	}
	out := make([]string, 0, maxCoverageFacetQueries)
	head := coverageHeadToken(parts[0])
	for _, part := range parts {
		probe := coverageFacetFromSegment(part)
		if probe == "" {
			continue
		}
		out = append(out, probe)
		if head != "" && len(questionTokens(probe)) == 1 {
			out = append(out, head+" "+probe)
		}
		if len(out) >= maxCoverageFacetQueries {
			return out[:maxCoverageFacetQueries]
		}
	}
	return out
}

func coverageFacetFromSegment(segment string) string {
	tokens := phraseProbeTokens(segment)
	if len(tokens) == 0 {
		return ""
	}
	filtered := make([]string, 0, len(tokens))
	for _, token := range tokens {
		token = normaliseCoverageToken(token)
		if phraseProbeBoundaryWords[token] || questionTokenStopWords[token] || coverageFacetSkipWords[token] || containsDigit(token) {
			continue
		}
		filtered = append(filtered, token)
	}
	if len(filtered) == 0 {
		return ""
	}
	if len(filtered) == 1 {
		return filtered[0]
	}
	return bestSegmentPhrase(filtered)
}

func coverageHeadToken(segment string) string {
	tokens := phraseProbeTokens(segment)
	for i := len(tokens) - 1; i >= 0; i-- {
		token := normaliseCoverageToken(tokens[i])
		if phraseProbeBoundaryWords[token] || questionTokenStopWords[token] || coverageFacetSkipWords[token] || containsDigit(token) {
			continue
		}
		return singularProbeEntityToken(token)
	}
	return ""
}

func normaliseCoverageFacet(candidate string) string {
	tokens := phraseProbeTokens(candidate)
	if len(tokens) == 0 {
		return ""
	}
	filtered := make([]string, 0, len(tokens))
	for _, token := range tokens {
		token = normaliseCoverageToken(token)
		if phraseProbeBoundaryWords[token] || questionTokenStopWords[token] || coverageFacetSkipWords[token] || containsDigit(token) {
			continue
		}
		filtered = append(filtered, token)
	}
	if len(filtered) == 0 {
		return ""
	}
	if len(filtered) > phraseProbeMaxTokens {
		filtered = filtered[:phraseProbeMaxTokens]
	}
	if len(filtered) == 1 && len(filtered[0]) < 4 {
		return ""
	}
	return strings.Join(filtered, " ")
}

func normaliseCoverageToken(token string) string {
	token = strings.TrimSpace(strings.ToLower(token))
	token = strings.TrimSuffix(token, "'s")
	token = strings.TrimSuffix(token, "’s")
	return token
}

func deriveSpecificRecommendationProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !specificRecommendationQueryRe.MatchString(lowered) {
		return nil
	}
	if !strings.Contains(lowered, "recommend") && !strings.Contains(lowered, "remind me") {
		return nil
	}
	return filteredPhraseProbes(question)
}

func deriveTypeContextProbes(question string) []string {
	if !isTypeContextCountQuery(question) {
		return nil
	}
	categoryMatch := typeContextQueryRe.FindStringSubmatch(question)
	scopeMatch := typeContextScopeRe.FindStringSubmatch(question)
	if len(categoryMatch) < 2 || len(scopeMatch) < 2 {
		return nil
	}

	category := firstCoverageToken(categoryMatch[1])
	scope := firstCoverageToken(scopeMatch[1])
	if category == "" || scope == "" || category == scope {
		return nil
	}

	return dedupeTrimmedStrings([]string{
		scope + " " + category,
		category + " " + scope,
	})
}

func firstCoverageToken(raw string) string {
	for _, token := range phraseProbeTokens(raw) {
		token = singularProbeEntityToken(normaliseCoverageToken(token))
		if token == "" ||
			phraseProbeBoundaryWords[token] ||
			phraseProbeConnectors[token] ||
			questionTokenStopWords[token] ||
			coverageFacetSkipWords[token] ||
			containsDigit(token) {
			continue
		}
		return token
	}
	return ""
}

func isTypeContextCountQuery(question string) bool {
	lowered := strings.ToLower(strings.TrimSpace(question))
	return lowered != "" && enumerationOrTotalQueryRe.MatchString(lowered) && typeContextQueryRe.MatchString(question)
}

func deriveDateArithmeticProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !dateArithmeticQueryRe.MatchString(lowered) {
		return nil
	}

	return eventProbeWindows(question, maxBM25FanoutQueries)
}

func eventProbeWindows(question string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	out := make([]string, 0, limit)
	seen := make(map[string]bool, limit)
	appendProbe := func(tokens []string) bool {
		tokens = trimPhraseProbeTokens(tokens)
		if len(tokens) == 0 {
			return false
		}
		if len(tokens) > phraseProbeMaxTokens {
			tokens = tokens[:phraseProbeMaxTokens]
		}
		candidates := []string{strings.Join(tokens, " ")}
		if len(tokens) >= 2 {
			past := inflectProbeVerbPast(tokens[0])
			if past != "" && past != tokens[0] {
				inflected := append([]string{past}, tokens[1:]...)
				candidates = append(candidates, strings.Join(inflected, " "))
			}
		}
		for _, candidate := range candidates {
			if candidate == "" || seen[candidate] {
				continue
			}
			seen[candidate] = true
			out = append(out, candidate)
			if len(out) == limit {
				return true
			}
		}
		return false
	}
	if match := dateArithmeticWhenRe.FindStringSubmatch(question); len(match) >= 3 {
		for _, clause := range match[1:] {
			if appendProbe(questionTokens(clause)) {
				return out
			}
		}
		if len(out) > 0 {
			return out
		}
	}

	rawTokens := phraseProbeTokens(question)
	var window []string
	flush := func() bool {
		if len(window) == 0 {
			return false
		}
		done := appendProbe(window)
		window = nil
		return done
	}
	for _, token := range rawTokens {
		if dateArithmeticBoundaryRe.MatchString(token) || phraseProbeBoundaryWords[token] || len(token) < 2 || containsDigit(token) {
			if flush() {
				return out
			}
			continue
		}
		window = append(window, token)
	}
	flush()
	return out
}

func inflectProbeVerbPast(token string) string {
	switch strings.ToLower(strings.TrimSpace(token)) {
	case "buy":
		return "bought"
	case "begin":
		return "began"
	case "catch":
		return "caught"
	case "drive":
		return "drove"
	case "ride":
		return "rode"
	case "see":
		return "saw"
	case "sell":
		return "sold"
	case "take":
		return "took"
	case "travel":
		return "travelled"
	}
	switch {
	case strings.HasSuffix(token, "e"):
		return token + "d"
	case strings.HasSuffix(token, "y") && len(token) > 1:
		return strings.TrimSuffix(token, "y") + "ied"
	default:
		return token + "ed"
	}
}

func singularProbeEntityToken(token string) string {
	token = strings.ToLower(strings.TrimSpace(token))
	switch {
	case strings.HasSuffix(token, "ss") || strings.HasSuffix(token, "us"):
		return token
	case strings.HasSuffix(token, "ies") && len(token) > 4:
		return strings.TrimSuffix(token, "ies") + "y"
	case strings.HasSuffix(token, "oes") && len(token) > 4:
		return strings.TrimSuffix(token, "es")
	case strings.HasSuffix(token, "ches") && len(token) > 5:
		return strings.TrimSuffix(token, "es")
	case strings.HasSuffix(token, "shes") && len(token) > 5:
		return strings.TrimSuffix(token, "es")
	case strings.HasSuffix(token, "xes") && len(token) > 4:
		return strings.TrimSuffix(token, "es")
	case strings.HasSuffix(token, "ses") && len(token) > 4:
		return strings.TrimSuffix(token, "es")
	case strings.HasSuffix(token, "s") && len(token) > 3:
		return strings.TrimSuffix(token, "s")
	default:
		return token
	}
}

func deriveCountTotalProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" || !enumerationOrTotalQueryRe.MatchString(lowered) || moneyEventQueryRe.MatchString(lowered) {
		return nil
	}
	match := countTotalIncludingRe.FindStringSubmatch(question)
	if len(match) != 2 {
		return nil
	}
	return cleanAggregateListProbes(match[1], maxBM25FanoutQueries)
}

func cleanAggregateListProbes(raw string, limit int) []string {
	cleaned := strings.TrimSpace(raw)
	cleaned = strings.TrimRight(cleaned, "?.! ")
	if cleaned == "" || limit <= 0 {
		return nil
	}
	cleaned = regexp.MustCompile(`(?i)\b(?:since|from|across|over|during|when|where)\b.*$`).ReplaceAllString(cleaned, "")
	cleaned = strings.ReplaceAll(cleaned, ", and ", ", ")
	parts := strings.Split(cleaned, ",")
	out := make([]string, 0, limit)
	seen := make(map[string]bool, limit)
	appendProbe := func(part string) {
		probe := cleanAggregateListProbe(part)
		if probe == "" || seen[probe] {
			return
		}
		seen[probe] = true
		out = append(out, probe)
	}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.Contains(strings.ToLower(part), " and ") && !strings.Contains(strings.ToLower(part), "flea and tick") {
			for _, sub := range regexp.MustCompile(`(?i)\s+and\s+`).Split(part, -1) {
				appendProbe(sub)
				if len(out) == limit {
					return out
				}
			}
			continue
		}
		appendProbe(part)
		if len(out) == limit {
			return out
		}
	}
	return out
}

func cleanAggregateListProbe(raw string) string {
	probe := strings.ToLower(strings.TrimSpace(raw))
	if probe == "" {
		return ""
	}
	probe = regexp.MustCompile(`(?i)\bpieces?\s+for\s+(?:the\s+)?`).ReplaceAllString(probe, "")
	probe = regexp.MustCompile(`(?i)\b(?:pieces?|items?|parts?|entries?|tasks?|things?)\s+of\s+`).ReplaceAllString(probe, "")
	tokens := questionTokens(probe)
	filtered := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if phraseProbeBoundaryWords[token] || countProbeSkipWords[token] {
			continue
		}
		filtered = append(filtered, token)
	}
	if len(filtered) == 0 {
		return ""
	}
	if len(filtered) > phraseProbeMaxTokens {
		filtered = filtered[len(filtered)-phraseProbeMaxTokens:]
	}
	return strings.Join(filtered, " ")
}

var countProbeSkipWords = map[string]bool{
	"completed": true,
	"including": true,
	"number":    true,
	"piece":     true,
	"pieces":    true,
}

func deriveRecallTitleProbes(question string) []string {
	lowered := strings.ToLower(strings.TrimSpace(question))
	if lowered == "" ||
		(!strings.Contains(lowered, "previous chat") &&
			!strings.Contains(lowered, "previous conversation") &&
			!strings.Contains(lowered, "remind me") &&
			!strings.Contains(lowered, "specific")) {
		return nil
	}
	matches := recallTitleProbeRe.FindAllString(question, -1)
	if len(matches) == 0 {
		return nil
	}
	out := make([]string, 0, maxDerivedSubQueries)
	seen := make(map[string]bool, maxDerivedSubQueries)
	for _, match := range matches {
		tokens := strings.Fields(match)
		filtered := make([]string, 0, len(tokens))
		for _, token := range tokens {
			cleaned := strings.Trim(token, `.,;:!?"'()[]{}<>`)
			cleaned = strings.TrimSuffix(cleaned, "'s")
			cleaned = strings.TrimSuffix(cleaned, "’s")
			loweredToken := strings.ToLower(cleaned)
			if loweredToken == "" || phraseProbeBoundaryWords[loweredToken] || questionTokenStopWords[loweredToken] {
				continue
			}
			filtered = append(filtered, loweredToken)
		}
		if len(filtered) < 2 {
			continue
		}
		probe := strings.Join(filtered, " ")
		if seen[probe] {
			continue
		}
		seen[probe] = true
		out = append(out, probe)
		if len(out) == maxDerivedSubQueries {
			break
		}
	}
	return out
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
	return dedupeTrimmedStrings([]string{
		focus + " inspiration",
		focus + " examples",
	})
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

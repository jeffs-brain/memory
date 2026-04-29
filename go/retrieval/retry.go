// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/jeffs-brain/memory/go/query"
)

// retryStopWords mirrors the spec's STOP_WORDS set in
// retrieval/retry.ts. Deliberately duplicated here so the retry
// ladder keeps the same English + small Dutch overlap whether or not
// the caller wires up the larger search.Stopwords payload.
var retryStopWords = map[string]bool{
	"the":             true,
	"a":               true,
	"an":              true,
	"and":             true,
	"or":              true,
	"but":             true,
	"is":              true,
	"are":             true,
	"was":             true,
	"what":            true,
	"who":             true,
	"when":            true,
	"where":           true,
	"why":             true,
	"how":             true,
	"you":             true,
	"for":             true,
	"from":            true,
	"about":           true,
	"advice":          true,
	"any":             true,
	"been":            true,
	"can":             true,
	"choose":          true,
	"current":         true,
	"decide":          true,
	"deciding":        true,
	"feeling":         true,
	"find":            true,
	"getting":         true,
	"help":            true,
	"helpful":         true,
	"idea":            true,
	"ideas":           true,
	"interesting":     true,
	"ive":             true,
	"look":            true,
	"looking":         true,
	"make":            true,
	"making":          true,
	"might":           true,
	"need":            true,
	"needs":           true,
	"noticed":         true,
	"planning":        true,
	"recent":          true,
	"recently":        true,
	"recommend":       true,
	"recommendation":  true,
	"recommendations": true,
	"should":          true,
	"some":            true,
	"soon":            true,
	"suggest":         true,
	"suggestion":      true,
	"suggestions":     true,
	"sure":            true,
	"thinking":        true,
	"tips":            true,
	"together":        true,
	"trying":          true,
	"upcoming":        true,
	"useful":          true,
	"want":            true,
	"weekend":         true,
	"with":            true,
	"would":           true,
	"again":           true,
	"becoming":        true,
	"bit":             true,
	"combined":        true,
	"having":          true,
	"items":           true,
	"keep":            true,
	"keeping":         true,
	"kind":            true,
	"kinds":           true,
	"lately":          true,
	"many":            true,
	"seen":            true,
	"show":            true,
	"tonight":         true,
	"trouble":         true,
	"type":            true,
	"types":           true,
	"watch":           true,
	"have":            true,
	"has":             true,
	"had":             true,
	"de":              true,
	"het":             true,
	"een":             true,
	"en":              true,
	"of":              true,
}

// trigramJaccardThreshold is fixed by the spec. Raising it drops
// recall sharply; lowering it lets slug noise leak through.
const trigramJaccardThreshold = 0.3

// sanitisePunctRe matches Unicode punctuation and symbols.
// Equivalent to the TypeScript `/[\p{P}\p{S}]+/gu`.
var sanitisePunctRe = regexp.MustCompile(`[\p{P}\p{S}]+`)

// sanitiseQuery strips punctuation so the "refreshed sanitised"
// rung retries a cleaner form of the raw input. Mirrors the Go
// reference and the TS port bit-for-bit.
func sanitiseQuery(q string) string {
	if q == "" {
		return ""
	}
	stripped := sanitisePunctRe.ReplaceAllString(q, " ")
	return strings.Join(strings.Fields(stripped), " ")
}

// strongestTerm returns the longest non-stop-word token of at least
// three characters from the sanitised, lowercased query. Empty
// string when nothing survives filtering.
func strongestTerm(q string) string {
	best := ""
	for _, tok := range normaliseRetryTokens(q) {
		if len(tok) < 3 {
			continue
		}
		if retryStopWords[tok] {
			continue
		}
		if len(tok) > len(best) {
			best = tok
		}
	}
	return best
}

// queryTokens returns every non-stop-word token of at least three
// characters, deduplicated and lowercased. The trigram fallback
// feeds this in so every surviving token gets a chance to match a
// slug.
func queryTokens(q string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0)
	for _, tok := range normaliseRetryTokens(q) {
		if len(tok) < 3 {
			continue
		}
		if retryStopWords[tok] {
			continue
		}
		if seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	return out
}

func normaliseRetryTokens(q string) []string {
	cleaned := sanitiseQuery(q)
	if cleaned == "" {
		return nil
	}
	lowered := strings.ToLower(cleaned)
	return strings.Fields(lowered)
}

func filtersWithQuestionDate(filters Filters, questionDate string) (Filters, bool) {
	anchor, ok := parseCandidateTime(questionDate)
	if !ok {
		return filters, false
	}
	anchor = endOfQuestionDate(anchor)
	out := filters
	if out.DateTo.IsZero() || anchor.Before(out.DateTo) {
		out.DateTo = anchor
		return out, true
	}
	return out, false
}

func endOfQuestionDate(anchor time.Time) time.Time {
	year, month, day := anchor.Date()
	return time.Date(year, month, day, 23, 59, 59, int(time.Second-time.Nanosecond), time.UTC)
}

type bm25Quality struct {
	label       string
	score       float64
	shouldRetry bool
}

func assessBM25CandidateQuality(question, questionDate string, candidates []rrfCandidate) bm25Quality {
	if len(candidates) == 0 {
		return bm25Quality{label: "empty", shouldRetry: true}
	}
	if dateMismatched(question, questionDate, candidates) {
		return bm25Quality{label: "date_mismatch", score: 0.1, shouldRetry: true}
	}

	tokens := queryTokens(question)
	if len(tokens) == 0 {
		return bm25Quality{label: "good", score: 1}
	}

	text := qualityText(candidates, 3)
	matched := 0
	for _, token := range tokens {
		if strings.Contains(text, token) {
			matched++
		}
	}
	coverage := float64(matched) / float64(len(tokens))
	score := coverage

	strongest := strongestTerm(question)
	if strongest != "" && strings.Contains(text, strongest) {
		score += 0.15
	}
	if score > 1 {
		score = 1
	}

	switch {
	case matched == 0:
		return bm25Quality{label: "generic", score: score, shouldRetry: true}
	case len(tokens) >= 4 && matched < 2:
		return bm25Quality{label: "generic", score: score, shouldRetry: true}
	case len(tokens) >= 3 && coverage < 0.34:
		return bm25Quality{label: "weak", score: score, shouldRetry: true}
	default:
		return bm25Quality{label: "good", score: score}
	}
}

func bm25QualityImproved(candidate, baseline bm25Quality) bool {
	if candidate.label == "good" && baseline.label != "good" && candidate.score >= baseline.score {
		return true
	}
	return candidate.score >= baseline.score+0.2
}

func qualityText(candidates []rrfCandidate, limit int) string {
	if len(candidates) < limit {
		limit = len(candidates)
	}
	var b strings.Builder
	for i := 0; i < limit; i++ {
		c := candidates[i]
		b.WriteByte(' ')
		b.WriteString(c.path)
		b.WriteByte(' ')
		b.WriteString(c.title)
		b.WriteByte(' ')
		b.WriteString(c.summary)
		b.WriteByte(' ')
		b.WriteString(c.content)
	}
	return strings.ToLower(b.String())
}

func dateMismatched(question, questionDate string, candidates []rrfCandidate) bool {
	anchor, hasAnchor := parseCandidateTime(questionDate)
	hints := temporalHintTimes(question, questionDate)

	dated := 0
	future := 0
	nearestHintDays := 1e9
	for i, c := range candidates {
		if i >= 5 {
			break
		}
		when, ok := extractTimeFromText(strings.Join([]string{c.title, c.summary, c.content}, "\n"))
		if !ok {
			continue
		}
		dated++
		if hasAnchor && when.After(anchor) {
			future++
		}
		for _, hint := range hints {
			days := absDurationDays(when.Sub(hint))
			if days < nearestHintDays {
				nearestHintDays = days
			}
		}
	}
	if dated == 0 {
		return false
	}
	if hasAnchor && future == dated {
		return true
	}
	return len(hints) > 0 && nearestHintDays > 45
}

func temporalHintTimes(question, questionDate string) []time.Time {
	expansion := query.ExpandTemporal(question, questionDate)
	if !expansion.Resolved {
		return nil
	}
	out := make([]time.Time, 0, len(expansion.DateHints))
	for _, hint := range expansion.DateHints {
		if parsed, ok := parseCandidateTime(hint); ok {
			out = append(out, parsed)
		}
	}
	return out
}

func absDurationDays(d time.Duration) float64 {
	if d < 0 {
		d = -d
	}
	return d.Hours() / 24
}

// trigramChunk is the minimum payload the trigram index needs. Paths
// are required; title / summary / content are optional but used to
// hydrate the hit so callers can read the match without another
// round-trip.
type trigramChunk struct {
	ID        string
	Path      string
	Title     string
	Summary   string
	Content   string
	Tags      string
	Scope     string
	Project   string
	Session   string
	SessionID string
}

type trigramHit struct {
	ID         string
	Path       string
	Similarity float64
	Title      string
	Summary    string
	Content    string
	Tags       string
	Scope      string
	Project    string
	Session    string
	SessionID  string
}

type trigramIndex struct {
	entries []trigramEntry
	byGram  map[string][]int
}

type trigramEntry struct {
	id        string
	path      string
	grams     map[string]struct{}
	title     string
	summary   string
	content   string
	tags      string
	scope     string
	project   string
	session   string
	sessionID string
}

// buildTrigramIndex constructs a slug trigram index over the supplied
// chunks. Duplicate IDs are collapsed to the first entry, matching
// the TS reference.
func buildTrigramIndex(chunks []trigramChunk) *trigramIndex {
	idx := &trigramIndex{
		entries: make([]trigramEntry, 0, len(chunks)),
		byGram:  make(map[string][]int),
	}
	seen := make(map[string]bool)
	for _, c := range chunks {
		if c.ID == "" || seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		grams := computeTrigrams(slugTextFor(c.Path))
		entry := trigramEntry{
			id:        c.ID,
			path:      c.Path,
			grams:     grams,
			title:     c.Title,
			summary:   c.Summary,
			content:   c.Content,
			tags:      c.Tags,
			scope:     c.Scope,
			project:   c.Project,
			session:   c.Session,
			sessionID: c.SessionID,
		}
		pos := len(idx.entries)
		idx.entries = append(idx.entries, entry)
		for g := range grams {
			idx.byGram[g] = append(idx.byGram[g], pos)
		}
	}
	return idx
}

// search runs a Jaccard search over every query token; the best
// similarity per entry wins. Limit <= 0 returns nothing.
func (t *trigramIndex) search(tokens []string, limit int) []trigramHit {
	if t == nil || len(tokens) == 0 || len(t.entries) == 0 || limit <= 0 {
		return nil
	}
	type bestHit struct {
		pos        int
		similarity float64
	}
	best := make(map[string]bestHit)
	for _, tok := range tokens {
		qGrams := computeTrigrams(tok)
		if len(qGrams) == 0 {
			continue
		}
		seen := make(map[int]struct{})
		for g := range qGrams {
			for _, pos := range t.byGram[g] {
				seen[pos] = struct{}{}
			}
		}
		for pos := range seen {
			entry := t.entries[pos]
			if len(entry.grams) == 0 {
				continue
			}
			sim := jaccard(qGrams, entry.grams)
			if sim < trigramJaccardThreshold {
				continue
			}
			prev, ok := best[entry.id]
			if !ok || sim > prev.similarity {
				best[entry.id] = bestHit{pos: pos, similarity: sim}
			}
		}
	}

	hits := make([]trigramHit, 0, len(best))
	for _, bh := range best {
		e := t.entries[bh.pos]
		hits = append(hits, trigramHit{
			ID:         e.id,
			Path:       e.path,
			Similarity: bh.similarity,
			Title:      e.title,
			Summary:    e.summary,
			Content:    e.content,
			Tags:       e.tags,
			Scope:      e.scope,
			Project:    e.project,
			Session:    e.session,
			SessionID:  e.sessionID,
		})
	}

	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Similarity != hits[j].Similarity {
			return hits[i].Similarity > hits[j].Similarity
		}
		return hits[i].Path < hits[j].Path
	})

	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits
}

// computeTrigrams returns the $-padded 3-gram set for text,
// lowercased, with non-alphanumerics squashed to spaces.
func computeTrigrams(text string) map[string]struct{} {
	out := make(map[string]struct{})
	if text == "" {
		return out
	}
	cleaned := collapseNonAlnum(strings.ToLower(text))
	for _, word := range strings.Fields(cleaned) {
		padded := "$" + word + "$"
		if len([]rune(padded)) < 3 {
			continue
		}
		runes := []rune(padded)
		for i := 0; i+3 <= len(runes); i++ {
			out[string(runes[i:i+3])] = struct{}{}
		}
	}
	return out
}

// slugTextFor keeps only the filename stem so single-word queries
// match slugs without being drowned out by parent-directory noise.
// Mirrors the Go reference implementation.
func slugTextFor(p string) string {
	s := strings.ToLower(p)
	if idx := strings.LastIndex(s, "/"); idx >= 0 {
		s = s[idx+1:]
	}
	if strings.HasSuffix(s, ".md") {
		s = s[:len(s)-3]
	}
	return strings.TrimSpace(collapseNonAlnum(s))
}

// collapseNonAlnum replaces runs of non-letter, non-digit runes with
// a single space. Unicode-aware so accented characters survive.
func collapseNonAlnum(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	lastSpace := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastSpace = false
			continue
		}
		if !lastSpace {
			b.WriteByte(' ')
			lastSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}

func jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	small, large := a, b
	if len(b) < len(a) {
		small, large = b, a
	}
	intersection := 0
	for g := range small {
		if _, ok := large[g]; ok {
			intersection++
		}
	}
	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

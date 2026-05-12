// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"regexp"
	"sort"
	"strings"
	"time"
)

// tokenPattern matches alphanumeric tokens used for text search.
var tokenPattern = regexp.MustCompile(`[a-z0-9][a-z0-9._:-]*`)

// matchesEpisodeFilters returns true if the episode matches all
// non-zero filter criteria.
func matchesEpisodeFilters(ep EpisodeRecord, opts EpisodeListOptions) bool {
	if opts.ActorID != "" && ep.ActorID != opts.ActorID {
		return false
	}
	if opts.Scope != "" && ep.Scope != opts.Scope {
		return false
	}
	if opts.Outcome != "" && ep.Outcome != opts.Outcome {
		return false
	}
	if opts.SessionID != "" && ep.SessionID != opts.SessionID {
		return false
	}

	if len(opts.Tags) > 0 {
		required := make(map[string]bool, len(opts.Tags))
		for _, tag := range opts.Tags {
			normalised := normaliseTagStr(tag)
			if normalised != "" {
				required[normalised] = true
			}
		}
		if len(required) > 0 {
			actual := make(map[string]bool, len(ep.Tags))
			for _, tag := range ep.Tags {
				actual[normaliseTagStr(tag)] = true
			}
			for tag := range required {
				if !actual[tag] {
					return false
				}
			}
		}
	}

	ts := episodeTimestampMs(ep)
	if opts.From != nil && ts < opts.From.UnixMilli() {
		return false
	}
	if opts.To != nil && ts > opts.To.UnixMilli() {
		return false
	}
	return true
}

// episodeTimestampMs returns the best-available timestamp for an
// episode in milliseconds since epoch. It prefers endedAt, then
// startedAt, then modified, then created.
func episodeTimestampMs(ep EpisodeRecord) int64 {
	for _, s := range []string{ep.EndedAt, ep.StartedAt, ep.Modified, ep.Created} {
		if s == "" {
			continue
		}
		t, err := time.Parse(time.RFC3339, s)
		if err != nil {
			t, err = time.Parse(time.RFC3339Nano, s)
		}
		if err == nil {
			return t.UnixMilli()
		}
	}
	return 0
}

// sortEpisodesNewestFirst sorts episodes by timestamp descending, then
// by path ascending for deterministic ordering.
func sortEpisodesNewestFirst(episodes []EpisodeRecord) {
	sort.Slice(episodes, func(i, j int) bool {
		ti := episodeTimestampMs(episodes[i])
		tj := episodeTimestampMs(episodes[j])
		if ti != tj {
			return ti > tj
		}
		return string(episodes[i].Path) < string(episodes[j].Path)
	})
}

// sortQueryHits sorts query hits by score descending, then by
// timestamp descending, then by path ascending.
func sortQueryHits(hits []EpisodeQueryHit) {
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		ti := episodeTimestampMs(hits[i].EpisodeRecord)
		tj := episodeTimestampMs(hits[j].EpisodeRecord)
		if ti != tj {
			return ti > tj
		}
		return string(hits[i].Path) < string(hits[j].Path)
	})
}

// scoreEpisode computes a relevance score for an episode against a
// query string and its tokens.
func scoreEpisode(ep EpisodeRecord, trimmedQuery string, queryTokens []string) int {
	score := 0
	summaryTokens := tokeniseText(ep.Summary, 24)
	tagTokens := tokeniseText(strings.Join(ep.Tags, " "), 24)

	var heuristicParts []string
	for _, h := range ep.Heuristics {
		heuristicParts = append(heuristicParts,
			h.Rule, h.Context, h.Category, h.Confidence, h.Scope)
		if h.AntiPattern {
			heuristicParts = append(heuristicParts, "anti-pattern")
		} else {
			heuristicParts = append(heuristicParts, "pattern")
		}
	}
	heuristicTokens := tokeniseText(strings.Join(heuristicParts, " "), 48)

	supportParts := []string{ep.SessionID, ep.ActorID, string(ep.Outcome), ep.RetryFeedback}
	supportParts = append(supportParts, ep.OpenQuestions...)
	supportTokens := tokeniseText(strings.Join(supportParts, " "), 48)

	score += countOverlap(queryTokens, summaryTokens) * 4
	score += countOverlap(queryTokens, tagTokens) * 3
	score += countOverlap(queryTokens, heuristicTokens) * 2
	score += countOverlap(queryTokens, supportTokens)

	if trimmedQuery != "" {
		searchable := buildSearchText(ep)
		if strings.Contains(searchable, trimmedQuery) {
			score += 2
		}
		if strings.ToLower(ep.SessionID) == trimmedQuery {
			score += 8
		}
		if strings.ToLower(string(ep.Outcome)) == trimmedQuery {
			score += 2
		}
	}

	return score
}

// buildSearchText concatenates all searchable fields of an episode
// into a single lowercase string.
func buildSearchText(ep EpisodeRecord) string {
	parts := []string{
		string(ep.Path),
		ep.SessionID,
		ep.ActorID,
		ep.Summary,
		string(ep.Outcome),
		ep.RetryFeedback,
	}
	parts = append(parts, ep.OpenQuestions...)
	parts = append(parts, ep.Tags...)
	for _, h := range ep.Heuristics {
		parts = append(parts, h.Rule, h.Context, h.Category, h.Confidence, h.Scope)
		if h.AntiPattern {
			parts = append(parts, "anti-pattern")
		} else {
			parts = append(parts, "pattern")
		}
	}
	return strings.ToLower(strings.Join(parts, "\n"))
}

// tokeniseText extracts unique, stemmed tokens from text.
func tokeniseText(text string, limit int) []string {
	matches := tokenPattern.FindAllString(strings.ToLower(text), -1)
	seen := make(map[string]bool, len(matches))
	out := make([]string, 0, limit)
	for _, match := range matches {
		token := stemToken(match)
		if len(token) < 2 {
			continue
		}
		if seen[token] {
			continue
		}
		seen[token] = true
		out = append(out, token)
		if len(out) >= limit {
			break
		}
	}
	return out
}

// stemToken and countOverlap are defined in text_util.go.

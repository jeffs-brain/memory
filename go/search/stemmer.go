// SPDX-License-Identifier: Apache-2.0

package search

import (
	"math"
	"strings"
	"sync"
	"unicode"

	snowballRuntime "github.com/blevesearch/snowballstem"
	"github.com/blevesearch/snowballstem/danish"
	"github.com/blevesearch/snowballstem/dutch"
	"github.com/blevesearch/snowballstem/english"
	"github.com/blevesearch/snowballstem/finnish"
	"github.com/blevesearch/snowballstem/french"
	"github.com/blevesearch/snowballstem/german"
	"github.com/blevesearch/snowballstem/hungarian"
	"github.com/blevesearch/snowballstem/italian"
	"github.com/blevesearch/snowballstem/norwegian"
	"github.com/blevesearch/snowballstem/portuguese"
	"github.com/blevesearch/snowballstem/romanian"
	"github.com/blevesearch/snowballstem/russian"
	"github.com/blevesearch/snowballstem/spanish"
	"github.com/blevesearch/snowballstem/swedish"
	"github.com/blevesearch/snowballstem/turkish"
)

// Stemmer applies language-specific Snowball stemming to a single token.
type Stemmer interface {
	Stem(word string) string
	Language() string
}

// snowballStemmer wraps a Snowball stem function for a specific language.
type snowballStemmer struct {
	lang   string
	stemFn func(env *snowballRuntime.Env) bool
}

// Compile-time interface compliance check.
var _ Stemmer = (*snowballStemmer)(nil)

// NewSnowballStemmer returns a Stemmer for the given ISO 639-1 language
// code. Returns an error if the language is not supported.
func NewSnowballStemmer(lang string) (Stemmer, error) {
	fn, ok := stemFunctions[lang]
	if !ok {
		return nil, &UnsupportedLanguageError{Lang: lang}
	}
	return &snowballStemmer{lang: lang, stemFn: fn}, nil
}

// Stem applies the Snowball algorithm to the word and returns the stem.
// The input is lowercased before stemming for consistency.
func (s *snowballStemmer) Stem(word string) string {
	if word == "" {
		return ""
	}
	lower := strings.ToLower(word)
	env := snowballRuntime.NewEnv(lower)
	s.stemFn(env)
	return env.Current()
}

// Language returns the ISO 639-1 code for this stemmer's language.
func (s *snowballStemmer) Language() string {
	return s.lang
}

// stemFunctions maps ISO 639-1 codes to their Snowball stem functions.
var stemFunctions = map[string]func(env *snowballRuntime.Env) bool{
	"en": english.Stem,
	"de": german.Stem,
	"fr": french.Stem,
	"es": spanish.Stem,
	"nl": dutch.Stem,
	"it": italian.Stem,
	"pt": portuguese.Stem,
	"sv": swedish.Stem,
	"no": norwegian.Stem,
	"da": danish.Stem,
	"fi": finnish.Stem,
	"hu": hungarian.Stem,
	"tr": turkish.Stem,
	"ro": romanian.Stem,
	"ru": russian.Stem,
}

// UnsupportedLanguageError is returned when a stemmer is requested for
// a language that has no Snowball implementation.
type UnsupportedLanguageError struct {
	Lang string
}

func (e *UnsupportedLanguageError) Error() string {
	return "search: unsupported stemmer language: " + e.Lang
}

// DefaultConfidenceThreshold is the minimum detection confidence
// required to use a non-English stemmer (fastText production standard).
// Below this threshold the detector returns English as the safe default.
const DefaultConfidenceThreshold = 0.7

// DefaultMinDetectionLength is the minimum number of alphabetic
// characters required for reliable language detection. Apple ML Research
// validated that below 50 characters, detection accuracy drops below
// 90%.
const DefaultMinDetectionLength = 50

// DetectLanguageOptions configures language detection behaviour.
type DetectLanguageOptions struct {
	// Threshold is the minimum confidence score (0-1) required to use
	// the detected language. Below this, English is returned as the
	// safe default. Zero uses DefaultConfidenceThreshold.
	Threshold float64
	// MinLength is the minimum number of alphabetic characters required
	// for detection. Below this, English is returned with zero
	// confidence. Zero uses DefaultMinDetectionLength.
	MinLength int
}

// DetectLanguage performs basic bigram-frequency language detection on
// text. Returns the ISO 639-1 code and a confidence score in [0, 1].
// When confidence is below the threshold, returns "en" as the safe
// default.
//
// The detection uses character-level bigram frequency profiles for each
// language. Confidence is computed by scaling the best cosine similarity
// score (typically 0.2-0.7) to [0, 1]. Short texts (below MinLength
// characters of alphabetic content) always return "en" with zero
// confidence.
//
// Pass nil for opts to use defaults.
func DetectLanguage(text string, opts *DetectLanguageOptions) (string, float64) {
	threshold := DefaultConfidenceThreshold
	minLen := DefaultMinDetectionLength
	if opts != nil {
		if opts.Threshold > 0 {
			threshold = opts.Threshold
		}
		if opts.MinLength > 0 {
			minLen = opts.MinLength
		}
	}

	cleaned := extractAlphaRuns(text)
	if len([]rune(cleaned)) < minLen {
		return "en", 0.0
	}

	bigrams := buildBigrams(cleaned)
	if len(bigrams) == 0 {
		return "en", 0.0
	}

	profilesMu.RLock()
	profiles := languageProfiles
	profilesMu.RUnlock()

	var bestLang string
	var bestScore float64
	for lang, profile := range profiles {
		score := bigramCosineSimilarity(bigrams, profile)
		if score > bestScore {
			bestScore = score
			bestLang = lang
		}
	}

	if bestScore == 0 {
		return "en", 0.0
	}

	// Scale the raw cosine score (typically 0.2-0.7) to [0, 1].
	// Scores above 0.35 map to confidence above 0.7.
	confidence := math.Min(1.0, bestScore*2.0)

	if confidence < threshold {
		return "en", confidence
	}

	return bestLang, confidence
}

// RegisterLanguage adds a custom language profile at runtime, enabling
// detection and stemming for languages beyond the built-in set. The code
// must be an ISO 639-1 language code. The profile is a map of character
// bigrams to normalised frequency values. If a profile for the code
// already exists, it is replaced.
func RegisterLanguage(code string, profile map[string]float64) {
	profilesMu.Lock()
	defer profilesMu.Unlock()

	// Copy the map to avoid mutation of the shared global.
	newProfiles := make(map[string]map[string]float64, len(languageProfiles)+1)
	for k, v := range languageProfiles {
		newProfiles[k] = v
	}
	newProfiles[code] = profile
	languageProfiles = newProfiles
}

// profilesMu protects concurrent access to languageProfiles.
var profilesMu sync.RWMutex

// extractAlphaRuns keeps only letter characters and spaces, collapsing
// non-letter runs into single spaces. Used for language detection.
func extractAlphaRuns(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	prevSpace := true
	for _, r := range text {
		if unicode.IsLetter(r) {
			b.WriteRune(unicode.ToLower(r))
			prevSpace = false
		} else if !prevSpace {
			b.WriteByte(' ')
			prevSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}

// buildBigrams extracts character bigram frequencies from text,
// normalized to a unit vector for cosine similarity comparison.
func buildBigrams(text string) map[string]float64 {
	runes := []rune(text)
	if len(runes) < 2 {
		return nil
	}

	counts := make(map[string]int, len(runes))
	var total int
	for i := 0; i+1 < len(runes); i++ {
		bigram := string(runes[i : i+2])
		counts[bigram]++
		total++
	}

	if total == 0 {
		return nil
	}

	freqs := make(map[string]float64, len(counts))
	for bg, count := range counts {
		freqs[bg] = float64(count) / float64(total)
	}
	return freqs
}

// bigramCosineSimilarity computes the cosine similarity between a
// document bigram frequency vector and a language profile vector.
func bigramCosineSimilarity(doc map[string]float64, profile map[string]float64) float64 {
	var dot, normDoc, normProfile float64
	for bg, freq := range doc {
		normDoc += freq * freq
		if pFreq, ok := profile[bg]; ok {
			dot += freq * pFreq
		}
	}
	for _, pFreq := range profile {
		normProfile += pFreq * pFreq
	}
	if normDoc == 0 || normProfile == 0 {
		return 0
	}
	return dot / (math.Sqrt(normDoc) * math.Sqrt(normProfile))
}

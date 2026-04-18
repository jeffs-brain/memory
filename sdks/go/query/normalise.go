// SPDX-License-Identifier: Apache-2.0

package query

import (
	"strings"
	"unicode"
)

// Common English stop words for significance detection.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true, "need": true,
	"to": true, "of": true, "in": true, "for": true, "on": true,
	"with": true, "at": true, "by": true, "from": true, "as": true,
	"into": true, "through": true, "about": true, "up": true, "out": true,
	"and": true, "but": true, "or": true, "nor": true, "not": true,
	"so": true, "yet": true, "both": true, "either": true, "neither": true,
	"this": true, "that": true, "these": true, "those": true,
	"i": true, "me": true, "my": true, "we": true, "our": true,
	"you": true, "your": true, "he": true, "she": true, "it": true,
	"they": true, "them": true, "its": true, "his": true, "her": true,
	"what": true, "which": true, "who": true, "whom": true, "how": true,
	"when": true, "where": true, "why": true,
}

// countTokens returns a rough whitespace-based token count.
func countTokens(s string) int {
	count := 0
	inToken := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if inToken {
				count++
				inToken = false
			}
		} else {
			inToken = true
		}
	}
	if inToken {
		count++
	}
	return count
}

// countSignificantTerms counts non-stop-word tokens in the input.
func countSignificantTerms(s string) int {
	count := 0
	for _, word := range strings.Fields(s) {
		w := strings.ToLower(strings.TrimFunc(word, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}))
		if w != "" && !stopWords[w] {
			count++
		}
	}
	return count
}

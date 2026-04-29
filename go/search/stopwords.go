// SPDX-License-Identifier: Apache-2.0

package search

import (
	_ "embed"
	"encoding/json"
)

// Mirror of spec/fixtures/stopwords — update both if changing. The spec
// directory lives above the Go module root so //go:embed cannot reach
// it directly; the files are duplicated here and verified in sync by
// the stopwords sync test in the spec conformance suite.
//
//go:embed stopwords/en.json
var enStopwordsJSON []byte

//go:embed stopwords/nl.json
var nlStopwordsJSON []byte

// stopWords is the union of the canonical English and Dutch stopword
// sets loaded at package init time. Parity with the TypeScript and
// Python SDKs is guaranteed because all three load the same JSON
// fixtures under spec/fixtures/stopwords.
var stopWords = loadStopWords()

func loadStopWords() map[string]bool {
	out := make(map[string]bool, 256)
	for _, raw := range [][]byte{enStopwordsJSON, nlStopwordsJSON} {
		var list []string
		if err := json.Unmarshal(raw, &list); err != nil {
			// The embed payload is committed alongside the package, so
			// any parse failure is a developer error at build time.
			panic("search: malformed stopwords fixture: " + err.Error())
		}
		for _, tok := range list {
			if tok == "" {
				continue
			}
			out[tok] = true
		}
	}
	return out
}

// isStopWord returns true when the lowercase token is in the curated
// English or Dutch stop word set. Tokens of two or fewer characters
// are also treated as stop words so "me", "to", "in" fall out
// alongside the curated list.
func isStopWord(token string) bool {
	if len(token) <= 2 {
		return true
	}
	return stopWords[token]
}

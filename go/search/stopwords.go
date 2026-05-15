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

//go:embed stopwords/de.json
var deStopwordsJSON []byte

//go:embed stopwords/fr.json
var frStopwordsJSON []byte

//go:embed stopwords/es.json
var esStopwordsJSON []byte

//go:embed stopwords/it.json
var itStopwordsJSON []byte

//go:embed stopwords/pt.json
var ptStopwordsJSON []byte

//go:embed stopwords/ru.json
var ruStopwordsJSON []byte

//go:embed stopwords/zh.json
var zhStopwordsJSON []byte

//go:embed stopwords/ja.json
var jaStopwordsJSON []byte

// stopWordsByLang holds per-language stopword sets, keyed by ISO 639-1
// code. Loaded at package init time from embedded JSON fixtures.
var stopWordsByLang = loadStopWordsByLang()

// stopWords is the union of the canonical English and Dutch stopword
// sets loaded at package init time. Parity with the TypeScript and
// Python SDKs is guaranteed because all three load the same JSON
// fixtures under spec/fixtures/stopwords.
var stopWords = buildUnionStopWords()

func loadStopWordsByLang() map[string]map[string]struct{} {
	langData := map[string][]byte{
		"en": enStopwordsJSON,
		"nl": nlStopwordsJSON,
		"de": deStopwordsJSON,
		"fr": frStopwordsJSON,
		"es": esStopwordsJSON,
		"it": itStopwordsJSON,
		"pt": ptStopwordsJSON,
		"ru": ruStopwordsJSON,
		"zh": zhStopwordsJSON,
		"ja": jaStopwordsJSON,
	}

	result := make(map[string]map[string]struct{}, len(langData))
	for lang, raw := range langData {
		var list []string
		if err := json.Unmarshal(raw, &list); err != nil {
			panic("search: malformed stopwords fixture for " + lang + ": " + err.Error())
		}
		set := make(map[string]struct{}, len(list))
		for _, tok := range list {
			if tok != "" {
				set[tok] = struct{}{}
			}
		}
		result[lang] = set
	}
	return result
}

func buildUnionStopWords() map[string]bool {
	out := make(map[string]bool, 256)
	for _, lang := range []string{"en", "nl"} {
		for tok := range stopWordsByLang[lang] {
			out[tok] = true
		}
	}
	return out
}

// StopWords returns the stopword set for the given ISO 639-1 language
// code. Returns nil for unsupported languages.
func StopWords(lang string) map[string]struct{} {
	return stopWordsByLang[lang]
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

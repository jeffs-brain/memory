// SPDX-License-Identifier: Apache-2.0

package search

import (
	"strings"
	"unicode"
)

// cjkTrigramSize is the number of codepoints per CJK trigram token.
const cjkTrigramSize = 3

// IsCJK reports whether the rune is in a CJK script range: CJK
// Unified Ideographs, Hiragana, Katakana (including prolonged sound
// mark and other modifier characters in the Katakana block), or Hangul
// Syllables. These scripts lack explicit word boundaries and require
// n-gram tokenization for effective full-text search.
func IsCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r) ||
		isKatakanaBlock(r)
}

// isKatakanaBlock checks whether r falls in the Katakana Unicode block
// (U+30A0..U+30FF) which includes modifier marks like the prolonged
// sound mark (ー, U+30FC) that unicode.Katakana does not cover.
func isKatakanaBlock(r rune) bool {
	return r >= 0x30A0 && r <= 0x30FF
}

// TokenizeCJK splits text into tokens suitable for CJK full-text
// search. CJK runs are converted into overlapping 3-character trigrams;
// non-CJK segments are split on whitespace and returned as regular
// word tokens.
//
// Time: O(N) where N = number of runes in text.
// Space: O(N) for the output slice.
func TokenizeCJK(text string) []string {
	if text == "" {
		return nil
	}

	runes := []rune(text)
	tokens := make([]string, 0, len(runes)/2)

	var cjkRun []rune
	var latinRun strings.Builder

	flushCJK := func() {
		if len(cjkRun) == 0 {
			return
		}
		if len(cjkRun) < cjkTrigramSize {
			tokens = append(tokens, string(cjkRun))
		} else {
			for i := 0; i+cjkTrigramSize <= len(cjkRun); i++ {
				tokens = append(tokens, string(cjkRun[i:i+cjkTrigramSize]))
			}
		}
		cjkRun = cjkRun[:0]
	}

	flushLatin := func() {
		if latinRun.Len() == 0 {
			return
		}
		for _, word := range strings.Fields(latinRun.String()) {
			tokens = append(tokens, strings.ToLower(word))
		}
		latinRun.Reset()
	}

	for _, r := range runes {
		switch {
		case IsCJK(r):
			flushLatin()
			cjkRun = append(cjkRun, r)
		case unicode.IsSpace(r) || unicode.IsPunct(r):
			flushCJK()
			if latinRun.Len() > 0 {
				latinRun.WriteRune(r)
			}
		default:
			flushCJK()
			latinRun.WriteRune(r)
		}
	}

	flushCJK()
	flushLatin()

	if len(tokens) == 0 {
		return nil
	}
	return tokens
}

// ContainsCJK reports whether text contains any CJK codepoints,
// indicating that trigram tokenization should be used alongside or
// instead of whitespace-based splitting.
func ContainsCJK(text string) bool {
	for _, r := range text {
		if IsCJK(r) {
			return true
		}
	}
	return false
}

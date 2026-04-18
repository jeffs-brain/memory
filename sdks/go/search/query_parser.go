// SPDX-License-Identifier: Apache-2.0

package search

import (
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// TokenKind is the shape of a single unit produced by [ParseQuery].
type TokenKind int

const (
	// TokTerm is a bare word that should go through the FTS5
	// stemmer.
	TokTerm TokenKind = iota
	// TokPhrase is a multi-word string that should match exactly
	// (wrapped in double quotes on the FTS5 side).
	TokPhrase
	// TokPrefix is a word followed by `*` for prefix matching.
	TokPrefix
)

// String returns the spec name for the token kind. Mirrors the
// `kind` field on the AST nodes in spec/fixtures/query-parser/cases.json
// so the golden test can compare directly.
func (k TokenKind) String() string {
	switch k {
	case TokPhrase:
		return "phrase"
	case TokPrefix:
		return "prefix"
	default:
		return "term"
	}
}

// Token is one unit produced by [ParseQuery]. Operator is only
// populated when the user wrote an explicit uppercase boolean
// (`AND`/`OR`/`NOT`) immediately before the token; otherwise the
// default operator configured in [BuildFTS5Expr] applies.
type Token struct {
	Kind     TokenKind
	Text     string
	Operator string
}

// AST is the full parse result. Raw is the NFC + whitespace-collapsed
// form of the input. HasOperators records whether the input carried
// any quote or explicit uppercase AND/OR/NOT; per spec/QUERY-DSL.md
// that flag disables stopword filtering.
type AST struct {
	Raw          string
	Tokens       []Token
	HasOperators bool
}

// ftsTermReplacer strips FTS5 special characters and natural-language
// punctuation from bare terms so [BuildFTS5Expr] never emits a
// malformed expression. Phrases are exempt: their content is wrapped
// in double quotes, which FTS5 treats as a literal phrase. The set
// mirrors spec/QUERY-DSL.md ("Token stripping" section).
var ftsTermReplacer = strings.NewReplacer(
	"*", "",
	"(", "",
	")", "",
	":", "",
	"^", "",
	"+", "",
	`"`, "",
	"-", "",
	"?", "",
	"!", "",
	".", "",
	",", "",
	";", "",
	"/", "",
	"\\", "",
	"[", "",
	"]", "",
	"{", "",
	"}", "",
	"<", "",
	">", "",
	"|", "",
	"&", "",
	"'", "",
	"$", "",
	"#", "",
	"@", "",
	"%", "",
	"=", "",
	"~", "",
	"`", "",
)

// ftsPhraseReplacer scrubs the small number of characters that would
// prematurely close a phrase or break the FTS5 parser inside quoted
// content.
var ftsPhraseReplacer = strings.NewReplacer(
	`"`, "",
)

// normaliseInput applies the spec pre-parse normalisation: Unicode
// NFC, strip zero-width and BOM characters, map NBSP to regular
// space, collapse whitespace, trim. Returns the canonical `raw` form
// that the golden fixture compares against.
func normaliseInput(raw string) string {
	if raw == "" {
		return ""
	}
	n := norm.NFC.String(raw)
	var b strings.Builder
	for _, r := range n {
		switch r {
		case '\u200B', '\u200C', '\u200D', '\uFEFF':
			// Strip zero-width joiners / non-joiners / BOM.
			continue
		case '\u00A0':
			// NBSP mapped to regular space.
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	// Collapse runs of whitespace and trim.
	return strings.Join(strings.Fields(b.String()), " ")
}

// hasOperatorSignals reports whether the raw input contains at least
// one double quote or at least one whitespace-bounded uppercase
// boolean operator. When true the parser disables stopword filtering,
// preserving power-user intent verbatim.
func hasOperatorSignals(raw string) bool {
	if strings.ContainsRune(raw, '"') {
		return true
	}
	for _, field := range strings.Fields(raw) {
		if field == "AND" || field == "OR" || field == "NOT" {
			return true
		}
	}
	return false
}

// Parse walks raw and produces the full [AST]: the normalised `raw`
// string, the token stream, and the `hasOperators` flag documented in
// spec/QUERY-DSL.md.
//
// TODO(spec): jeff's upstream parser (apps/jeff/internal/search)
// always applied stopword filtering to bare terms regardless of
// whether the query contained quotes or explicit uppercase AND /
// OR / NOT. This port follows the spec's "disable stopword
// filtering when any operator or phrase is present" rule so the
// golden fixture passes; callers that relied on jeff's stricter
// filtering will see the stopword list survive in operator-bearing
// queries. The spec is the source of truth for cross-SDK
// determinism.
func Parse(raw string) AST {
	normalised := normaliseInput(raw)
	ast := AST{Raw: normalised}
	if normalised == "" {
		return ast
	}

	ast.HasOperators = hasOperatorSignals(normalised)
	filterStopwords := !ast.HasOperators

	runes := []rune(normalised)
	i := 0
	pendingOp := ""

	for i < len(runes) {
		r := runes[i]

		// Skip whitespace.
		if unicode.IsSpace(r) {
			i++
			continue
		}

		// Quoted phrase.
		if r == '"' {
			i++ // consume opening quote
			start := i
			for i < len(runes) && runes[i] != '"' {
				i++
			}
			phrase := string(runes[start:i])
			if i < len(runes) {
				i++ // consume closing quote
			}
			phrase = strings.TrimSpace(ftsPhraseReplacer.Replace(phrase))
			phrase = strings.ToLower(phrase)
			if phrase != "" {
				ast.Tokens = append(ast.Tokens, Token{
					Kind:     TokPhrase,
					Text:     phrase,
					Operator: pendingOp,
				})
				pendingOp = ""
			}
			continue
		}

		// Bare word: run until whitespace or quote.
		start := i
		for i < len(runes) && !unicode.IsSpace(runes[i]) && runes[i] != '"' {
			i++
		}
		word := string(runes[start:i])
		if word == "" {
			continue
		}

		// Uppercase boolean operator outside a phrase.
		if word == "AND" || word == "OR" || word == "NOT" {
			pendingOp = word
			continue
		}

		// Prefix token: trailing `*` survives the scrub; everything
		// else is cleaned.
		isPrefix := strings.HasSuffix(word, "*")
		if isPrefix {
			word = strings.TrimSuffix(word, "*")
		}

		// Keep the pre-scrub form around so the alias map can be
		// consulted with the user's original surface token. The FTS5
		// scrub strips hyphens, which would otherwise turn `a-ware`
		// into `aware` before alias lookup and silently miss the
		// entry.
		preScrubLower := strings.ToLower(strings.TrimSpace(word))

		cleaned := ftsTermReplacer.Replace(word)
		cleaned = strings.TrimSpace(cleaned)
		if cleaned == "" {
			pendingOp = ""
			continue
		}

		lower := strings.ToLower(cleaned)
		if filterStopwords {
			// Lowercase `and`, `or`, `not` are natural-language
			// filler in bare-word queries per spec/QUERY-DSL.md.
			// Drop them before consulting the stopword set.
			if lower == "and" || lower == "or" || lower == "not" {
				pendingOp = ""
				continue
			}
			if isStopWord(lower) {
				pendingOp = ""
				continue
			}
		}

		kind := TokTerm
		if isPrefix {
			kind = TokPrefix
		}

		// Alias expansion only applies to bare terms: phrases are
		// literal, prefix tokens already cover a family of stems, and
		// aliasing either of them would silently break user intent.
		if kind == TokTerm {
			expanded := expandAlias(preScrubLower)
			if len(expanded) == 1 && preScrubLower != lower {
				expanded = expandAlias(lower)
			}
			if len(expanded) > 1 {
				emitted := 0
				seen := map[string]bool{}
				for _, alt := range expanded {
					altToken, ok := aliasToken(alt)
					if !ok {
						continue
					}
					dedupeKey := fmt.Sprintf("%d|%s", altToken.Kind, altToken.Text)
					if seen[dedupeKey] {
						continue
					}
					seen[dedupeKey] = true

					if emitted == 0 {
						altToken.Operator = pendingOp
					}
					ast.Tokens = append(ast.Tokens, altToken)
					emitted++
				}
				if emitted > 0 {
					pendingOp = ""
					continue
				}
			}
		}

		ast.Tokens = append(ast.Tokens, Token{
			Kind:     kind,
			Text:     lower,
			Operator: pendingOp,
		})
		pendingOp = ""
	}

	return ast
}

// ParseQuery is the jeff-style token-stream entry point kept for
// compatibility with the ported tests. Prefer [Parse] for the full
// AST shape.
func ParseQuery(raw string) []Token {
	return Parse(raw).Tokens
}

// expandAlias consults the package-wide alias map and returns the
// alternatives for token, lowercased and deduplicated. When no alias
// map is installed or the token has no match, the original token is
// returned as a single-element slice so callers can branch on length.
func expandAlias(token string) []string {
	m := getAliasMap()
	if m == nil {
		return []string{token}
	}
	return m.Expand(token)
}

// aliasToken normalises a single alias alternative into a Token
// ready for [BuildFTS5Expr]. Multi-word or hyphenated alternatives
// become phrase tokens so the FTS5 tokenizer (porter+unicode61,
// which splits on non-alphanumeric) still matches them against the
// stored documents. Single-word alternatives stay as bare terms so
// they benefit from the stemmer. Returns (zero, false) when the
// alternative collapses to nothing after trimming.
func aliasToken(alt string) (Token, bool) {
	alt = strings.ToLower(strings.TrimSpace(alt))
	if alt == "" {
		return Token{}, false
	}

	normalised := strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return r
		}
		return ' '
	}, alt)
	fields := strings.Fields(normalised)
	if len(fields) == 0 {
		return Token{}, false
	}
	if len(fields) == 1 {
		return Token{Kind: TokTerm, Text: fields[0]}, true
	}
	return Token{Kind: TokPhrase, Text: strings.Join(fields, " ")}, true
}

// BuildFTS5Expr converts parsed tokens into a valid FTS5 MATCH
// expression. Default operator between bare terms is `OR`. Explicit
// operators from the user override the default. Phrase tokens are
// wrapped in double quotes (the one place quoting is legitimate in
// the new pipeline). Prefix tokens keep their trailing `*`.
//
// FTS5 does not support `OR NOT`, so any `NOT` token preceded by the
// default `OR` is rewritten to `AND NOT`. Likewise an explicit `NOT`
// operator is always emitted as `AND NOT` when it is not the first
// token in the expression. A leading `NOT` is illegal in FTS5; the
// compiler drops the operator and keeps the operand.
func BuildFTS5Expr(tokens []Token) string {
	if len(tokens) == 0 {
		return ""
	}

	var b strings.Builder
	for _, tok := range tokens {
		piece := renderToken(tok)
		if piece == "" {
			continue
		}

		if b.Len() == 0 {
			// First token: an explicit leading NOT is not valid in
			// FTS5, so drop the operator prefix.
			b.WriteString(piece)
			continue
		}

		op := tok.Operator
		if op == "" {
			op = "OR"
		}
		if op == "NOT" {
			// Rewrite bare `NOT` into `AND NOT` because FTS5 only
			// accepts the latter after the first operand.
			op = "AND NOT"
		}

		b.WriteString(" ")
		b.WriteString(op)
		b.WriteString(" ")
		b.WriteString(piece)
	}

	return strings.TrimSpace(b.String())
}

// renderToken produces the FTS5 surface form for a single [Token].
// Phrase tokens are wrapped in double quotes; prefix tokens keep
// their trailing `*`. Returns an empty string for pathological
// tokens (e.g. whitespace-only phrases) so [BuildFTS5Expr] can skip
// them cleanly.
func renderToken(tok Token) string {
	text := strings.TrimSpace(tok.Text)
	if text == "" {
		return ""
	}
	switch tok.Kind {
	case TokPhrase:
		return `"` + text + `"`
	case TokPrefix:
		return text + "*"
	default:
		return text
	}
}

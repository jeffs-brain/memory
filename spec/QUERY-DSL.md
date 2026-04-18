# Query DSL

The Jeffs Brain query DSL is a small, user-facing query language that compiles to a SQLite FTS5 `MATCH` expression. The reference implementation lives in `packages/memory/src/query/`. Every SDK must produce an identical AST and FTS5 expression for the same input so that retrieval is deterministic across languages.

## Pipeline

1. **Normalise** raw input: Unicode NFC, strip zero-width and BOM characters, map non-breaking space to regular space, collapse whitespace, trim.
2. **Parse** into a token AST. Quoted phrases, explicit `AND` / `OR` / `NOT`, and prefix wildcards are preserved.
3. **Stopword filter** (bare-word queries only; disabled when any quote or explicit operator is present).
4. **Alias expansion** (optional; driven by a caller-supplied `AliasTable`).
5. **Compile** to an FTS5 `MATCH` string.

## Grammar (EBNF)

```
query        = { whitespace | term | phrase | operator } ;
whitespace   = ? Unicode whitespace ? ;
operator     = "AND" | "OR" | "NOT" ;               (* case-sensitive, whitespace-bounded *)
phrase       = '"' , { any-char-except-quote } , '"' ;
term         = bare-word [ "*" ] ;                  (* trailing "*" marks a prefix wildcard *)
bare-word    = { letter | digit | other-non-whitespace-non-quote } ;
```

Notes:

- Operators are recognised only when they appear uppercase and whitespace-bounded. Lowercase `and`, `or`, `not` are natural-language filler and are dropped from bare queries.
- A trailing `*` on a bare term emits a `prefix` token. Internal `*` characters are stripped by the term cleaner.
- Quoted phrases are literal: stopword filtering and operator detection do not descend into them.

## Token stripping

Bare terms pass through a strip set that removes FTS5 control and punctuation characters:

```
* ( ) : ^ + " - ? ! . , ; / \ [ ] { } < > | & ' $ # @ % = ~ `
```

A bare word that reduces to the empty string after stripping is dropped silently. The cleaned token is then lowercased via `toLocaleLowerCase('en')` and compared against the stopword set. Phrases pass through only a quote-stripping pass and are not lowercased beyond `toLocaleLowerCase('en')` of the full phrase.

## Operator semantics

The AST attaches the nearest preceding explicit operator to the next emitted token via a `pendingOp` slot. If two operators appear back-to-back with no intervening term, only the later one survives.

Compilation to FTS5:

- Consecutive tokens join with `OR` by default.
- A token's explicit operator overrides the default.
- `NOT` rewrites to `AND NOT` on any non-first token (FTS5 requires `AND NOT`, not bare `NOT`, after the first operand).
- A leading `NOT` is illegal in FTS5; the compiler drops the operator and keeps the operand.

## Stopword filtering

Stopword filtering applies to **bare-word queries only**. Any of the following disables it for the whole query:

- The presence of at least one double quote
- The presence of at least one whitespace-bounded uppercase `AND`, `OR`, or `NOT`

When filtering is active:

- Tokens with length ≤ 2 are dropped.
- Tokens present in the combined EN + NL stopword set are dropped.

The canonical stopword lists live in `spec/fixtures/stopwords/en.json` and `spec/fixtures/stopwords/nl.json`.

## Alias expansion

`expand(ast, aliasTable)` substitutes each bare `term` token with one or more alternatives from the table. Rules:

- Only `term` tokens are expanded. `phrase` and `prefix` tokens are literal by design.
- Alternative strings are lowercased and split on anything other than Unicode letters or digits.
- Single-word alternatives become bare `term` tokens so they benefit from stemming.
- Multi-word or punctuation-bearing alternatives become `phrase` tokens.
- Deduplication keys on `(kind, text)` so overlapping alternatives collapse.
- If the original token had an explicit `operator`, that operator is attached to the first expanded token.

## FTS5 compilation target

The compiler walks the (optionally alias-expanded) AST and renders each token back into its surface form:

- `term` → the text itself
- `phrase` → `"<text>"`
- `prefix` → `<text>*`

Compiled pieces join with spaces. The default connective is `OR`; explicit operators override. `NOT` becomes `AND NOT`. Leading `NOT` is dropped.

## Worked examples

### Bare query with filler

Input: `The Kubernetes Deployment`

AST:

```json
{
  "raw": "The Kubernetes Deployment",
  "tokens": [
    { "kind": "term", "text": "kubernetes" },
    { "kind": "term", "text": "deployment" }
  ],
  "hasOperators": false
}
```

FTS5: `kubernetes OR deployment`

### Quoted phrase plus bare term

Input: `"hello world" kube*`

AST:

```json
{
  "raw": "\"hello world\" kube*",
  "tokens": [
    { "kind": "phrase", "text": "hello world" },
    { "kind": "prefix", "text": "kube" }
  ],
  "hasOperators": true
}
```

FTS5: `"hello world" OR kube*`

### Explicit boolean query

Input: `foo AND bar NOT baz`

AST:

```json
{
  "raw": "foo AND bar NOT baz",
  "tokens": [
    { "kind": "term", "text": "foo" },
    { "kind": "term", "text": "bar", "operator": "AND" },
    { "kind": "term", "text": "baz", "operator": "NOT" }
  ],
  "hasOperators": true
}
```

FTS5: `foo AND bar AND NOT baz`

### Leading NOT

Input: `NOT alpha bravo`

AST:

```json
{
  "raw": "NOT alpha bravo",
  "tokens": [
    { "kind": "term", "text": "alpha", "operator": "NOT" },
    { "kind": "term", "text": "bravo" }
  ],
  "hasOperators": true
}
```

FTS5: `alpha OR bravo` (leading NOT stripped by the compiler)

### Bare query that collapses to nothing

Input: `to do list`

All three tokens are stopword noise after the length-2 and stopword-list checks. AST: `tokens: []`. FTS5: empty string; retrieval layer must treat that as "no FTS leg" and fall back to other signals.

## Temporal expansion

Temporal expansion is an **optional augmentation** that runs before retrieval, not inside the parser. The reference implementation lives in `query/temporal.ts` and is used by callers who pass both a question and a `questionDate` anchor. Expansion does not modify the query AST; it produces a rewritten query string (and, separately, a set of date-search tokens) that the retrieval layer concatenates onto the user's input before parsing. The parser in `query/parser.ts` is temporally blind by design.

The expander is **English-locale only**. Dutch and other languages are out of scope for v1.0: non-English phrases pass through unchanged and expansion is reported as `resolved: false`. Go and Python SDKs MUST mirror the three recognisers below bit-for-bit and MUST NOT introduce locale-specific variants without a spec update.

### Recognised phrases

Three independent recognisers, applied in order. All matching is case-insensitive.

| Recogniser | Pattern (surface form) | Behaviour |
| --- | --- | --- |
| Relative time | `<N> day[s] ago`, `<N> week[s] ago`, `<N> month[s] ago` | Replace with the original phrase suffixed by `(around <YYYY/MM/DD>)`. |
| Last weekday | `last monday` .. `last sunday` | Walk back at most 7 days from the anchor until weekday matches, suffix with `(<YYYY/MM/DD>)`. |
| Ordering hint | case-insensitive substring match on `first`, `earlier`, `before`, `most recent`, `latest`, `last time` | Append either `[Note: look for the earliest dated event]` or `[Note: look for the most recently dated event]` to the query. |

Phrases not matched by any recogniser (including `today`, `yesterday`, `this week`, `last month`, `this quarter`, month names, ISO dates, and natural-language constructions like `the week of 12 March`) are **not** expanded. The list above is exhaustive for v1.0. Additional recognisers are tracked as future work and MUST land in the spec before being shipped in any SDK.

### Anchor handling

- The caller supplies `questionDate` as a string. Accepted surface forms, all parsed in UTC via `parseQuestionDate`:
  - `YYYY-MM-DD` or `YYYY/MM/DD`
  - Either of the above with an optional `(DOW)` suffix (e.g. `2026-04-18 (Sat)`) which is ignored
  - Either of the above with an optional `HH:MM[:SS]` suffix
  - Any string that `new Date()` can parse (e.g. full ISO-8601) as a fallback
- When `questionDate` is absent, undefined, or unparseable, `expandTemporal` returns `{ resolved: false }` and the query string is passed through unchanged.
- All date arithmetic uses UTC (`getUTCDate`, `setUTCDate`, `setUTCMonth`). There is no local-timezone conversion, no DST handling, and no IANA timezone database lookup. Callers that care about local time must normalise the anchor to UTC before passing it in.
- "Now" is not a distinct anchor. The current wall-clock time is never read inside `temporal.ts`. Any sense of "now" is whatever `questionDate` resolves to.

### Month-boundary and day-arithmetic edge cases

- `N months ago` uses `setUTCMonth(getUTCMonth() - N)`. This inherits JavaScript `Date` semantics: if the resulting month has fewer days than the anchor day, the date rolls forward into the next month. For example, anchor `2026-03-31`, `1 month ago` resolves to `2026-03-03`, not `2026-02-28`. SDK implementers MUST reproduce this behaviour even in languages with a stricter default (subtract N months, then let an overflow day roll forward).
- `N weeks ago` and `N days ago` are simple UTC-day subtraction (`setUTCDate(getUTCDate() - N)` and `… - N * 7`). No weekend or working-day awareness.
- `last <weekday>` walks back exactly one UTC day at a time and returns as soon as `getUTCDay()` matches. The anchor day itself is never returned; "last monday" on a Monday resolves to seven days earlier.

### Emitted date formats

Expansion appends ordinary text to the surface query, which is then re-fed to the parser. Dates are emitted in two forms so either the slash or the hyphen form matches indexed chunk text:

- `YYYY/MM/DD` (used inside the `(around …)` / `(…)` suffixes).
- `YYYY-MM-DD` (added separately by `augmentQueryWithTemporal` alongside the slash form as extra search tokens).

`dateSearchTokens(value)` additionally derives the four-digit year, the weekday name, and the month name (all in English) from a parsed anchor, so retrieval can match documents that spell the date out in prose.

### Worked example

Input (anchor `2026-04-18 (Sat)`, question `what did I watch 2 weeks ago last Friday?`):

```json
{
  "originalQuery": "what did I watch 2 weeks ago last Friday?",
  "expandedQuery": "what did I watch 2 weeks ago (around 2026/04/04) last Friday (2026/04/17)? [Note: look for the most recently dated event]",
  "dateHints": ["2026/04/04", "2026/04/17"],
  "resolved": true
}
```

`augmentQueryWithTemporal` for the same input yields:

```
what did I watch 2 weeks ago last Friday? 2026/04/04 2026-04-04 2026/04/17 2026-04-17
```

The result is an ordinary query string: it is re-parsed and re-compiled through the standard pipeline, so stopword filtering and alias expansion still apply.

## Alias tables

The `AliasTable` type is an **in-memory contract** only. The reference declaration in `query/aliases.ts` is:

```ts
export type AliasTable = ReadonlyMap<string, readonly string[]>
```

A persisted on-disk representation is **reserved for a future spec version**. The v1.0 TS SDK does not ship a loader, a writer, or a file-format schema: callers construct the map programmatically (typically by iterating their domain-specific entity store) and hand it to `createRetrieval({ aliases })` or call `expandAliases(ast, table)` directly. Go and Python SDKs MUST expose an equivalent in-memory type and MUST NOT invent an on-disk format unilaterally.

### Runtime semantics (normative)

- Keys are lowercase surface tokens. The lookup in `query/aliases.ts` accepts either the exact key or the `toLocaleLowerCase('en')` form.
- Values are an ordered list of alternative surface strings. Alternatives are lowercased and split on non-(letter|digit) runs at expansion time. Single-word survivors become `term` tokens; multi-word survivors join with a single space and become `phrase` tokens.
- Expansion applies to **`term` tokens only**. `phrase` and `prefix` tokens pass through unchanged.
- If the original `term` carried an explicit `operator`, that operator is copied onto the first emitted alternative.
- Deduplication keys on `(kind, text)` so overlapping alternatives collapse to a single emitted token.

### When a file-backed schema lands

The expected shape is flagged here so downstream implementers can plan for it, but **none of the SDKs read or write this file today**. Any SDK that ships support for persisted alias tables before the spec is updated is non-conformant.

- Scope: **per-brain**, one file per brain, stored at an SDK-defined path under the brain's storage root (candidate path `aliases.json` sibling to the brain's knowledge index). Global cross-brain aliases are out of scope for v1.0.
- Format: JSON object, top-level keys are lowercase surface tokens, values are `string[]` of alternative surface strings. No TTL, no versioning field, no per-entry metadata.
- Loading: SDKs MUST construct the `AliasTable` from the file at retrieval construction time. Mutations require a rebuild; there is no hot-reload contract.
- Updates: out of scope. The v1.0 contract is read-only from the SDK's perspective; writes come from whatever upstream process maintains the file.

Until this section is upgraded from "reserved" to "normative", SDK implementers MUST treat alias-table persistence as a caller concern and MUST NOT assume cross-language compatibility of any file they happen to ship.

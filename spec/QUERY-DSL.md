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

## TODOs / ambiguities

- **TODO**: Document temporal expansion in `packages/memory/src/query/temporal.ts`. That module augments the AST with date-range tokens when the user asks about `yesterday`, `last week`, etc.; it is not yet captured in this spec.
- **TODO**: Decide whether Go and Python SDKs should use the same `AliasTable` source format. The TS `AliasTable` is an in-memory map; a persisted representation (JSON/YAML) will need a schema.

# Stopwords fixtures

These JSON files are the **canonical** English and Dutch stopword sets used by the Jeffs Brain query parser when filtering bare-word queries before compilation to FTS5. Every SDK (TypeScript, Go, Python) must load these same lists so that query parsing is byte-for-byte identical across language implementations.

## Files

- `en.json`: English stopword set, sorted ascending. Combines articles, determiners, conjunctions, pronouns, common verbs, question words, prepositions, and a curated set of natural-language noise terms.
- `nl.json`: Dutch stopword set, sorted ascending. Covers articles, determiners, conjunctions, pronouns, common verbs, question words, prepositions, and affirmation noise.

## Keeping implementations in sync

The canonical stopword data lives in these JSON files. Every SDK loads
them identically:

- TypeScript: `sdks/ts/memory/src/query/stopwords.ts`.
- Go: `go/search/stopwords` helpers.
- Python: `sdks/py/src/jeffs_brain_memory/search/stopwords.py`.

Any change to either list must land in both the JSON and every SDK's
copy in the same pull request. A sync check will be added to CI that
fails when the JSON and any SDK's constants diverge.

## Runtime behaviour

The parser only applies stopword filtering to bare-word queries. Quoted phrases and queries containing explicit `AND` / `OR` / `NOT` operators bypass filtering entirely so power-user intent survives verbatim. Tokens two characters or shorter are also treated as noise regardless of language. See `spec/QUERY-DSL.md` for the full rule set.

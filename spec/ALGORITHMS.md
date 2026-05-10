# Retrieval algorithms

This document describes the hybrid retrieval pipeline every Jeffs Brain SDK must implement. The TypeScript reference lives in `packages/memory/src/retrieval/` and `packages/memory/src/rerank/`.

## Reciprocal Rank Fusion (RRF)

Cormack, Clarke, Buettcher (SIGIR 2009). Merges an arbitrary number of ranked lists into a single ranking.

```
RRF_DEFAULT_K = 60

function reciprocalRankFusion(lists, k = RRF_DEFAULT_K):
    safeK = k > 0 ? k : RRF_DEFAULT_K
    buckets = {}                                     # keyed by candidate.id
    for list in lists:
        for rank (0-indexed), candidate in list:
            contribution = 1 / (safeK + rank + 1)    # 1-indexed inside the formula
            if candidate.id not in buckets:
                buckets[candidate.id] = {
                    id, path, title, summary, content,
                    bm25Rank: candidate.bm25Rank,
                    vectorSimilarity: candidate.vectorSimilarity,
                    score: contribution,
                }
            else:
                existing = buckets[candidate.id]
                # Fill-in merge: only fill blanks from later lists.
                if existing.title is empty and candidate.title non-empty:
                    existing.title = candidate.title
                ... same for summary, content, bm25Rank, vectorSimilarity
                existing.score += contribution
    sort buckets by score descending, break ties by path ascending
    return buckets as list
```

Properties:

- Order of input lists is irrelevant to the final score: lists commute.
- Metadata merging is a one-way fill: the first list to introduce a candidate seeds title/summary/content; later lists only fill fields that were empty.
- Output is stable: equal scores break to path ascending.

## Hybrid pipeline

Driven by `createRetrieval({ index, embedder?, reranker?, aliases?, trigramChunks? })`.

```
function searchRaw(req):
    ast         = parseQuery(req.query)
    expanded    = aliases ? expandAliases(ast, aliases) : ast
    compiled    = compileToFTS(expanded)

    mode = resolveMode(req.mode ?? 'auto', embedder != null)
        # 'auto' with embedder   -> 'hybrid'
        # 'auto' without embedder-> 'bm25'
        # 'semantic'/'hybrid' without embedder -> falls back to 'bm25', records fellBackToBM25

    # -- BM25 leg (with retry ladder on zero hits) --
    bmCandidates = runBM25(compiled)
    if bmCandidates empty and not req.skipRetryLadder:
        bmCandidates = retryLadder(req.query)        # see retry ladder section

    # -- Vector leg --
    vecCandidates = []
    if embedder != null and mode in ('hybrid', 'semantic'):
        embedding = embedder.embed([req.query])[0]
        if embedding non-empty:
            vecCandidates = index.searchVector(embedding, candidateK)

    # -- Fuse --
    if mode == 'bm25':
        fused = bmCandidates mapped with score 1 / (k + rank + 1)
    else if mode == 'semantic':
        fused = vecCandidates mapped with score 1 / (k + rank + 1)
    else:   # hybrid
        fused = reciprocalRankFusion([bmCandidates, vecCandidates], RRF_DEFAULT_K)

    fused = reweightSharedMemoryRanking(req.query, fused)    # intent-aware multipliers

    # -- Unanimity shortcut + optional rerank --
    if fused empty or not rerankEnabled or reranker == null:
        final = fused
    else:
        shortcut = unanimityShortcut(bmCandidates top-3, vecCandidates top-3)
        if shortcut != null:
            final = fused         # skip the reranker, trace.rerankSkippedReason = 'unanimity'
        else:
            head = fused[:rerankTopN]
            tail = fused[rerankTopN:]
            reranked = reranker.rerank({ query, documents: headAsRerankDocs })
            final = reranked ++ tail       # tail preserved untouched

    return { results: final[:topK], trace }
```

Defaults: `topK = 10`, `candidateK = 60`, `rerankTopN = 20`, `rerank = true`.

### Initial BM25 fanout

Before the retry ladder runs, the BM25 leg may issue up to four focused
queries derived from the raw question.

- Priority fanout: action-date questions and compound total questions start
  with focused subqueries plus filtered phrase probes, then dedupe and
  truncate to `maxBM25FanoutQueries = 4`.
- General fanout: other questions start with any priority subqueries, then
  add the raw query, the temporal-augmented raw query, and up to two derived
  subqueries before dedupe and truncation.

This means strongest-term style probes and phrase probes may already be part
of the first BM25 pass. The retry ladder is only the fallback after the whole
initial BM25 fanout still returns zero hits.

### Unanimity shortcut

```
function unanimityShortcut(bm25, vector, agreeMin = 2):
    window = 3
    if len(bm25) < window or len(vector) < window:
        return null
    agreements = count positions i in [0, window)
                 where bm25[i].id == vector[i].id
    if agreements < agreeMin:
        return null
    return { ids: bm25[:window].map(id), agreements }
```

The shortcut is a pure check. Its presence means BM25 and vector independently converged on the same head, so a reranker run is unlikely to change the outcome and is skipped.

### Rerank tail preservation

When the reranker runs, only the top `rerankTopN` candidates (`head`) are rescored. The `tail` (rank `rerankTopN` onwards) is appended unchanged after the reordered head. This preserves recall at rank > N without paying for LLM scoring on the long tail.

The reranker accepts a `{ query, documents: [{ id, text }] }` payload. Reference text composition is `title \n summary`, falling back to `title`, `summary`, or a trimmed `content` snippet capped at 280 characters.

### Intent-aware reweighting

`reweightSharedMemoryRanking` applies multiplicative score adjustments when the query pattern matches:

- **Preference query** (`recommend`, `suggest`, `tips`, `ideas`, `what should I`, `which should I`): boost `memory/global/...user-preference-*` notes by 2.35×, generic `memory/global/` preference notes by 2.1×. Penalise non-global generic notes (0.82×) and rollup notes (0.9×).
- **Concrete-fact query** (`how many`, `count`, `total`, `list`, or `did I / have I / was I / were I` combined with atomic event verbs): boost paths matching `user-fact-*`, `milestone-*`, or notes carrying `[date:` / `[observed on:` tags or atomic event phrasing (2.2×). Penalise rollup/overview notes (0.45×) and generic non-global notes (0.75×).

Scoring is multiplicative; when both intents match, the multipliers compose.

## Retry ladder

Runs only when the whole BM25 leg, including the initial fanout above,
returns zero and `req.skipRetryLadder !== true`. Each rung returns as soon as
it produces at least one hit.

```
1. strongest_term        - longest non-stopword token of length ≥ 3 from the raw query.
2. force-refresh FTS     - no-op for SQLite/WAL. Preserved for attempt-trace symmetry.
3. refreshed_sanitised   - rerun after stripping punctuation and symbols.
4. refreshed_strongest   - strongest term of the sanitised query.
5. trigram_fuzzy         - Jaccard ≥ 0.3 over padded 3-gram sets derived from slug text.
```

- Rung 1: `strongestTerm(query)`. If it matches the raw lowered query, rung 1 is skipped (no new information).
- Rung 2 is retained as a pass-through; the Go reference refreshed the FTS metadata table here, but SQLite FTS5 under better-sqlite3 is a live view so there is nothing to refresh.
- Rung 3 sanitises via `/[\p{P}\p{S}]+/` → space, collapses whitespace.
- Rung 4 is rung 1 applied to the sanitised text.
- Rung 5 builds (or reuses) a lazy `TrigramIndex` from chunk `id + path + title + summary + content` and returns all candidates with Jaccard similarity ≥ 0.3 against any of the query tokens. Slug text is derived from the last path segment with the `.md` extension stripped and non-alphanumerics collapsed to spaces.

Trigram Jaccard:

```
grams(word) = { padded_window of length 3 in '$' + word + '$' }     # skip words shorter than 3
jaccard(A, B) = |A ∩ B| / |A ∪ B|
TRIGRAM_JACCARD_THRESHOLD = 0.3
```

Results are sorted by similarity descending, then path ascending, and truncated to `candidateK`.

## Trace

Every `searchRaw` call returns a `HybridTrace` alongside the results, including the compiled query, mode, leg counts, timing per leg, whether the reranker ran (and why not if it did not), which rungs of the retry ladder fired, and any unanimity agreement count. SDKs should surface this trace as-is for eval consumers.

## Intent-aware reweighting (English-only)

The regex-driven reweighter in `retrieval/hybrid.ts` (`detectRetrievalIntent`, `retrievalIntentMultiplier`) is English-locale. Every trigger pattern is hand-tuned against English surface forms (including common contractions and tenses). **Non-English queries bypass intent reweighting and receive the base RRF score without modification. Localisation to additional languages is tracked as future work.**

Go and Python SDKs MUST mirror the regexes bit-for-bit, using the same pattern text, the same case-insensitive flag, and the same `\b` word boundaries. Additional locales are out of scope for v1.0; any SDK that introduces locale branches before the spec is updated is non-conformant.

### Intent categories

Two intents are detected. A single query may match neither, either, or both. When both match, the multipliers compose (plain multiplication); the detection step does not rank or deduplicate them.

| Intent | Matches when | Source |
| --- | --- | --- |
| `preferenceQuery` | `PREFERENCE_QUERY_RE` matches the lowercased query. | `retrieval/hybrid.ts` |
| `concreteFactQuery` | `ENUMERATION_OR_TOTAL_QUERY_RE` matches OR both `FIRST_PERSON_FACT_LOOKUP_RE` and `FACT_LOOKUP_VERB_RE` match. | `retrieval/hybrid.ts` |

### Query-side regexes

All patterns use the `i` (case-insensitive) flag and are defined in `retrieval/hybrid.ts`:

```
PREFERENCE_QUERY_RE           = /\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b/i
ENUMERATION_OR_TOTAL_QUERY_RE = /\b(?:how many|count|total|in total|sum|add up|list|what are all)\b/i
FIRST_PERSON_FACT_LOOKUP_RE   = /\b(?:did i|have i|was i|were i)\b/i
FACT_LOOKUP_VERB_RE           = /\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked)\b/i
```

### Document-side regexes

Applied to a concatenation of `path`, `title`, `summary`, and `content`, lowercased:

```
PREFERENCE_NOTE_RE   = /\b(?:prefer(?:s|red)?|like(?:s|d)?|love(?:s|d)?|want(?:s|ed)?|need(?:s|ed)?|avoid(?:s|ed)?|dislike(?:s|d)?|hate(?:s|d)?|enjoy(?:s|ed)?|interested in|looking for)\b/i
GENERIC_NOTE_RE      = /\b(?:tips?|advice|suggest(?:ion|ed)?s?|recommend(?:ation|ed)?s?|ideas?|options?|guide|tracking|tracker|checklist)\b/i
ROLLUP_NOTE_RE       = /\b(?:roll-?up|summary|recap|overview|aggregate|combined|overall|in total|totalled?|totalling)\b/i
ATOMIC_EVENT_NOTE_RE = /\b(?:i|we)\s+(?:picked up|bought|ordered|spent|earned|sold|drove|travelled|traveled|went|watched|visited|completed|finished|started|booked|got|took|submitted)\b/i
DATE_TAG_RE          = /\[(?:date|observed on):/i
```

### Multipliers

Applied multiplicatively to the fused RRF score. When both intents match, the multipliers compose (e.g. concrete-fact on a rollup note under a non-global path scores `0.45 × 0.75 = 0.3375`).

**Preference-query multipliers** (`preferenceIntentMultiplier`):

| Condition | Multiplier |
| --- | --- |
| Path contains `memory/global/` AND path contains `user-preference-` | 2.35 |
| Path contains `memory/global/` AND `PREFERENCE_NOTE_RE` matches the note text | 2.1 |
| Path does **not** contain `memory/global/` AND `GENERIC_NOTE_RE` matches | 0.82 |
| `ROLLUP_NOTE_RE` matches | 0.9 |
| Otherwise | 1.0 |

The 2.35 / 2.1 tiers are mutually exclusive (the `user-preference-` check wins). The 0.82 and 0.9 tiers can both fire on a single note when the path is not global and the text is both generic and a rollup.

**Concrete-fact multipliers** (`concreteFactIntentMultiplier`), with `isRollUp = ROLLUP_NOTE_RE.test(text)`:

| Condition | Multiplier |
| --- | --- |
| `path` contains `user-fact-` OR `milestone-` OR (NOT `isRollUp` AND (`DATE_TAG_RE` matches OR `ATOMIC_EVENT_NOTE_RE` matches)) | 2.2 |
| `isRollUp` | 0.45 |
| NOT the 2.2 branch AND path does not contain `memory/global/` AND `GENERIC_NOTE_RE` matches | 0.75 |

Multipliers compose within the concrete-fact path too: a rollup note with a matching `user-fact-` slug is scored `2.2 × 0.45 = 0.99` (the rollup penalty still applies).

### Determinism rules for SDK ports

- Use the exact regex source strings above. Do not refactor to Unicode property escapes or pre-compiled character classes.
- Apply the case-insensitive flag only. No `u`, `g`, or `s` flags.
- Evaluate intent once per query; re-use the boolean results across every candidate.
- Tie-break identical scores by original fused rank (stable sort). The TS implementation does this via `(index - index)` in the comparator; SDK ports MUST preserve this ordering.
- Non-English queries never hit the regexes, so they silently skip the reweight. This is the intended v1.0 behaviour, not a bug.

## `forceRefreshIndex` retry ladder

Runs only when the whole BM25 leg, including initial fanout, returns zero
hits and the caller did not pass `skipRetryLadder: true`. Each rung re-runs
BM25 against a rewritten query and returns as soon as at least one candidate
is produced; the pipeline then moves on to fusion with whatever the vector
leg found. Rungs are implemented in `retrieval/hybrid.ts` and
`retrieval/retry.ts`.

The five-rung shape is preserved verbatim from the Go reference (`apps/jeff/internal/knowledge/search.go:SearchWithOpts`) so attempt traces stay diffable across SDKs. `forceRefreshIndex` is retained as an explicit pass-through function so that attempt trace callers can see the boundary between rung 2 and rung 3 even on SQLite where no refresh is needed. SDK ports MUST keep all five rungs in order, even when a rung is a no-op, or the trace will drift.

### Rungs (ordered)

Each rung emits one `RetryAttempt` with `{ strategy, query, hits }`. A rung is skipped (not emitted) only when its input is vacuously empty; otherwise it runs and records its hit count (which may be zero).

```
0. initial              : the compiled query that triggered the ladder.
                          Always emitted first; `strategy: 'initial'`.
1. strongest_term       : strongestTerm(req.query).
                          Skipped silently when strongest equals the lowered,
                          trimmed raw query (no new information).
2. force-refresh        : forceRefreshIndex() is called. No trace row emitted;
                          the function is a documented no-op under SQLite WAL.
                          Present solely to preserve rung ordering for parity.
3. refreshed_sanitised  : runBM25(sanitiseQuery(req.query)).
                          Skipped when the sanitised string is empty.
4. refreshed_strongest  : runBM25(strongestTerm(sanitiseQuery(req.query))).
                          Skipped when no strongest term survives sanitisation.
5. trigram_fuzzy        : Lazy trigram index, Jaccard >= 0.3 over slug text.
                          Tokens come from queryTokens(req.query); a hit is any
                          chunk whose slug-text trigram set has Jaccard
                          similarity >= 0.3 against any query token's trigrams.
                          Truncated to candidateK.
```

`strongestTerm(query)` keeps the longest non-stop-word token of length ≥ 3 from `sanitiseQuery(query).toLowerCase().split(/\s+/)`. `sanitiseQuery(query)` replaces `/[\p{P}\p{S}]+/gu` with a single space and collapses whitespace. The stop-word set is defined inline in `retrieval/retry.ts` as `STOP_WORDS` and is English + a small Dutch overlap (`de`, `het`, `een`, `en`, `of`).

### Escalation conditions

The ladder does not retry on errors; it retries on **zero hits**. The escalation rules are strictly sequential:

1. Run the initial compiled query.
2. If `bmCandidates.length === 0` AND `req.skipRetryLadder !== true`, walk rungs 1-5 in order.
3. After each rung, if `bmCandidates.length > 0`, stop walking the ladder immediately. The last successful rung's results become the BM25 leg; all later rungs are skipped and no further trace rows are emitted for them.
4. If an exception is raised during the initial BM25 call, the entire ladder is skipped and the error is recorded on `trace.errorStage = 'bm25'`. The ladder is not a retry loop for infrastructure failures.

### Backoff and concurrency

There is **no backoff**. Rungs run synchronously in the same call, against the same SQLite connection. There is no sleep, no jitter, and no per-rung timeout beyond the outer `AbortSignal`. The ladder is a pure fallback sequence, not a retry mechanism in the network-failure sense.

### Terminal state

If all five rungs produce zero hits, the retry ladder exits with `bmCandidates = []` and `attempts` populated for every rung that ran (excluding silently-skipped ones). Fusion still runs: when the vector leg has candidates, `mode === 'hybrid'` fuses `[vecCandidates]` alone through `reciprocalRankFusion`; when both legs are empty, `fused = []` and `trace.rerankSkippedReason = 'empty_candidates'`.

### Trigram fallback details

- Built lazily on first fallback invocation per `Retrieval` instance. Either from the caller-supplied `trigramChunks` or by reading `id, path, title, summary, content` from `knowledge_chunks` via the index's SQLite handle.
- Slug text is `slugTextFor(path)`: lowercase the path, keep the last segment, strip the `.md` suffix, and collapse non-alphanumerics to spaces.
- Trigrams are `$`-padded 3-grams over each space-separated word of the slug text. Words shorter than 3 characters after padding are dropped.
- `jaccard(A, B) = |A ∩ B| / |A ∪ B|`. The threshold constant `TRIGRAM_JACCARD_THRESHOLD = 0.3` is fixed.
- Candidates are sorted by similarity descending, ties broken by path ascending, and truncated to `candidateK`. The attempt row reports the joined query tokens as its `query` field.

## Multi-Language Search

This section documents the multi-language stemmer, language detection, and CJK trigram tokenizer utilities available for BM25 full-text search. These components are standalone functions (not yet wired into the FTS5 schema); integration requires a schema_version bump and full index rebuild (tracked as a follow-up).

### Snowball Stemmers

Both SDKs implement Snowball stemming algorithms for 15 languages. The Snowball project (snowballstem.org) provides deterministic, well-tested stemming rules generated from a domain-specific language.

**Supported languages** (ISO 639-1 codes):

| Code | Language   | Go package                        | TS package             |
|------|------------|-----------------------------------|------------------------|
| en   | English    | blevesearch/snowballstem/english   | snowball-stemmers      |
| de   | German     | blevesearch/snowballstem/german    | snowball-stemmers      |
| fr   | French     | blevesearch/snowballstem/french    | snowball-stemmers      |
| es   | Spanish    | blevesearch/snowballstem/spanish   | snowball-stemmers      |
| nl   | Dutch      | blevesearch/snowballstem/dutch     | snowball-stemmers      |
| it   | Italian    | blevesearch/snowballstem/italian   | snowball-stemmers      |
| pt   | Portuguese | blevesearch/snowballstem/portuguese| snowball-stemmers      |
| sv   | Swedish    | blevesearch/snowballstem/swedish   | snowball-stemmers      |
| no   | Norwegian  | blevesearch/snowballstem/norwegian | snowball-stemmers      |
| da   | Danish     | blevesearch/snowballstem/danish    | snowball-stemmers      |
| fi   | Finnish    | blevesearch/snowballstem/finnish   | snowball-stemmers      |
| hu   | Hungarian  | blevesearch/snowballstem/hungarian | snowball-stemmers      |
| tr   | Turkish    | blevesearch/snowballstem/turkish   | snowball-stemmers      |
| ro   | Romanian   | blevesearch/snowballstem/romanian  | snowball-stemmers      |
| ru   | Russian    | blevesearch/snowballstem/russian   | snowball-stemmers      |

**Cross-SDK conformance**: Both SDKs use the official Snowball algorithms (Go: auto-generated from Snowball source; TS: jssnowball port). Given the same lowercased input, both produce identical stems. This is verified in the test suites.

**Interface** (both SDKs):

```
Stemmer.Stem(word) -> stemmed_word
Stemmer.Language() -> ISO 639-1 code
NewStemmer(lang) -> Stemmer or error if unsupported
```

Input is lowercased before stemming. Empty input returns empty output.

### Language Detection

A lightweight bigram-frequency language detector selects the appropriate stemmer at index and query time.

**Algorithm**:

1. Extract alphabetic runs from text (lowercase, non-letters become spaces).
2. Build a character bigram frequency vector (count / total).
3. Compute cosine similarity against each language's reference profile (top-40 bigrams per language).
4. The highest-scoring language wins.
5. Confidence = min(1.0, raw_cosine_score * 2.0).
6. If confidence < 0.5 OR text has fewer than 20 alphabetic characters, default to English.

**Properties**:

- Time: O(N) for bigram extraction, O(L * 40) for scoring against L language profiles.
- No external dependencies (no ML model, no network call).
- Handles Cyrillic script well (Russian bigrams share no overlap with Latin profiles).
- Romance languages (FR/ES/IT/PT) may produce lower margins between best and second-best, but the correct language consistently scores highest.
- CJK text is NOT detected via this mechanism; use `ContainsCJK()` / `containsCJK()` for script-level detection.

### CJK Trigram Tokenizer

Chinese, Japanese, and Korean (CJK) scripts do not use spaces to delimit words. The tokenizer produces overlapping 3-character trigrams from CJK runs, which serve as indexable tokens for BM25 search.

**Algorithm**:

```
function TokenizeCJK(text):
    tokens = []
    for each run in text:
        if run is CJK characters:
            if len(run) < 3:
                tokens.append(run)       # preserve short fragments
            else:
                for i in 0..len(run)-3:
                    tokens.append(run[i:i+3])
        else:
            tokens.extend(whitespace_split(run).map(lowercase))
    return tokens
```

**CJK detection ranges**:

- CJK Unified Ideographs (U+4E00..U+9FFF)
- CJK Extension A (U+3400..U+4DBF)
- CJK Extension B (U+20000..U+2A6DF)
- CJK Compatibility Ideographs (U+F900..U+FAFF)
- Hiragana (U+3040..U+309F)
- Katakana (U+30A0..U+30FF) including prolonged sound mark
- Katakana Phonetic Extensions (U+31F0..U+31FF)
- Hangul Syllables (U+AC00..U+D7AF)
- Hangul Jamo (U+1100..U+11FF)
- Hangul Compatibility Jamo (U+3130..U+318F)

**Properties**:

- Time: O(N) where N = codepoints in text.
- Space: O(N) for output tokens.
- Mixed CJK/Latin text is handled correctly: CJK runs produce trigrams, Latin runs produce whitespace-split tokens.
- Short CJK runs (< 3 characters) are emitted as-is to avoid information loss.

### FTS5 Integration (follow-up)

Wiring these utilities into the FTS5 custom tokenizer requires:

1. Bump `schema_version` in both Go and TS search index modules.
2. Register a custom FTS5 tokenizer via `sqlite3_fts5_tokenizer` API (CGo for Go, native addon for TS).
3. The tokenizer pipeline: detect language -> if CJK, trigram-tokenize; else stem with detected language's stemmer.
4. Full index rebuild on schema version mismatch (detected at index open).
5. Per-document language tag stored in `knowledge_chunks` metadata to avoid re-detection at query time.

This is tracked separately and NOT implemented in this ticket.

## Content Hashing

All document and chunk content hashing uses BLAKE3 with a 256-bit output, hex-encoded to a 64-character lowercase string. BLAKE3 was selected for its combination of speed on x86-64 platforms (4-10x faster than SHA-256 on single-threaded workloads with AVX2/AVX-512), streaming support, and equivalent collision resistance to SHA-256.

### Specification

- **Algorithm**: BLAKE3
- **Output**: 256-bit digest, lowercase hex-encoded (64 characters)
- **Input**: Raw byte content (UTF-8 for text documents)
- **Use cases**:
  - Document deduplication: identical content produces identical hash regardless of path or metadata
  - Chunk-level change detection: re-ingesting a modified document identifies which chunks changed
  - Document ID derivation: content hash is the seed for deterministic document identifiers

### Cross-SDK conformance

All SDK implementations (Go, TypeScript) MUST produce identical hex output for identical byte input. The canonical test vector is:

```
Input (UTF-8): "jeff's brain memory system"
BLAKE3-256:    e311e54b56b26bfef4e5c8501f04c708f1e02233106022f58a7e94b728b7265c
```

SDK conformance tests MUST assert this exact output. Any implementation that diverges from this value is non-conformant.

### Implementation references

| SDK | Package | Function |
| --- | --- | --- |
| Go | `github.com/zeebo/blake3` | `ingest.HashDocument`, `ingest.HashChunk`, `ingest.HashString` |
| TypeScript | `@noble/hashes/blake3` | `hashDocument`, `hashChunk`, `hashString` |

### Performance notes

- On x86-64 with AVX2/AVX-512: BLAKE3 is 4-10x faster than SHA-256 for inputs above 1KB.
- On ARM64 (Apple Silicon): SHA-256 benefits from dedicated hardware instructions and may outperform the software BLAKE3 implementation. This is an acceptable trade-off because production workloads target AMD64 infrastructure.
- Both algorithms are O(n) in input size with constant memory overhead (32 bytes for the digest).

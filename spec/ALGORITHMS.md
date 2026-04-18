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

- Order of input lists is irrelevant to the final score — lists commute.
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

Runs only when the BM25 leg returns zero and `req.skipRetryLadder !== true`. Each rung returns as soon as it produces at least one hit.

```
1. strongest_term        — longest non-stopword token of length ≥ 3 from the raw query.
2. force-refresh FTS     — no-op for SQLite/WAL. Preserved for attempt-trace symmetry.
3. refreshed_sanitised   — rerun after stripping punctuation and symbols.
4. refreshed_strongest   — strongest term of the sanitised query.
5. trigram_fuzzy         — Jaccard ≥ 0.3 over padded 3-gram sets derived from slug text.
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

## TODOs / ambiguities

- **TODO**: The `intent-aware reweighting` regexes are hand-tuned to English. Go and Python implementers will need equivalent fixtures; the multipliers are specified here but the trigger patterns deserve a language-neutral test set.
- **TODO**: `forceRefreshIndex` is retained only for shape parity. Consider collapsing the ladder to four rungs in a future spec version once eval reports no longer depend on the five-rung layout.

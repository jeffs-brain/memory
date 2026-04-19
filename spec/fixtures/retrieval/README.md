# Retrieval Golden Fixtures

Golden outputs from the reference retrieval pipeline (originally captured
from the `jeff` TypeScript implementation). Every language SDK
(TypeScript, Go, Python) re-runs the same corpus and query set against
its own pipeline and compares the result against these files to prove
cross-language parity.

## Background

These captures stack on top of three plan iterations that shipped in
`~/code/jeff/research/query-retrieval/`:

- **Layer 1** fixed the `sanitiseQuery` bug that joined every natural
  language token with implicit `AND`, stripped porter stemming via
  double-quoting, and ignored stop words. Post-Layer 1 fixtures capture
  the intermediate state before BM25 column weighting and the retry
  ladder landed.
- **Layer 2 / retry ladder** added `SearchWithTrace`, single-strongest-term
  fallbacks, and index refresh retries at the `kb.Search` layer.
- **v2 hybrid** extended the pipeline with Ollama `bge-m3` vector search,
  RRF fusion (k=60), entity alias expansion, trigram fuzzy fallback, and
  an optional local Gemma rerank pass over the top twenty candidates.
- **Distillation** is the next slice: LLM-side rewriting of noisy user
  input (error pastes, shallow prompts, anaphoric references, code-heavy
  questions) into a clean retrieval query before it reaches the pipeline.
  The `golden-distillation.yaml` set is the specification for that work.

## Fixture inventory

| Filename | Category | Mode | Purpose | Shape |
|---|---|---|---|---|
| `baseline-after.json` | baseline | hybrid | Final 100% recall report against `golden-realworld.yaml` after all Layer 1 + Layer 2 work. | Eval report (`retrieval.queries[]` with `id`, `question`, `passed`, `retrieved[]`, `missing_must`). |
| `baseline-post-layer1.json` | baseline | hybrid | Intermediate report captured after Layer 1 parser rewrite but before Layer 2 retry ladder; 40% recall. | Eval report (same schema). |
| `hybrid-baseline-bm25.json` | hybrid | bm25 | BM25-only pass over `golden-hybrid.yaml`, 70% baseline on semantic concept queries. | Eval report. |
| `hybrid-baseline-semantic.json` | hybrid | semantic | bge-m3 vector-only pass over `golden-hybrid.yaml`, 100%. | Eval report. |
| `hybrid-baseline-hybrid.json` | hybrid | hybrid | RRF fusion (BM25 + vector, k=60) over `golden-hybrid.yaml`, 100%. | Eval report. |
| `hybrid-baseline-rerank.json` | hybrid | hybrid-rerank | Hybrid plus Gemma rerank over `golden-hybrid.yaml`, 100%. | Eval report. |
| `realworld-baseline-rerank.json` | realworld | hybrid-rerank | Hybrid plus rerank over `golden-realworld.yaml`, sanity check that v2 did not regress v1. | Eval report. |
| `final-hybrid-bm25.json` | hybrid | bm25 | Final locked-in BM25 pass over `golden-hybrid.yaml` after the 50x FTS5 rank-config speed fix. | Eval report. |
| `final-hybrid-semantic.json` | hybrid | semantic | Final locked-in semantic-only pass. | Eval report. |
| `final-hybrid-hybrid.json` | hybrid | hybrid | Final locked-in RRF hybrid pass. | Eval report. |
| `final-hybrid-hybrid-rerank.json` | hybrid | hybrid-rerank | Final locked-in hybrid plus rerank pass. | Eval report. |
| `final-realworld-bm25.json` | realworld | bm25 | Final locked-in BM25 pass over `golden-realworld.yaml`. | Eval report. |
| `final-realworld-semantic.json` | realworld | semantic | Final locked-in semantic-only pass. | Eval report. |
| `final-realworld-hybrid.json` | realworld | hybrid | Final locked-in RRF hybrid pass. | Eval report. |
| `final-realworld-hybrid-rerank.json` | realworld | hybrid-rerank | Final locked-in hybrid plus rerank pass. | Eval report. |
| `golden-realworld.yaml` | realworld | n/a | v1 real-world golden query set (10 queries from actual TUI usage). | `queries[]` with `id`, `q`, `any_of[]` and/or `must_retrieve[]`, `notes`. |
| `golden-hybrid.yaml` | hybrid | n/a | v2 hybrid-challenge golden set (10 queries phrased so BM25 alone struggles). | Same schema as realworld. |
| `golden-distillation.yaml` | distillation | n/a | 50-query spec for the LLM distillation slice; noisy user inputs that benefit from rewriting. | Same schema, with longer multi-line `q:` blocks. |

All JSON reports share the same top-level schema:

```
{
  "generated_at": RFC3339 timestamp,
  "schema_version": 1,
  "brain": { "root": string, "backend": "fsstore" },
  "corpus": { "wiki_articles", "raw_files", "compiled_files",
              "uncompiled_files", "total_sources" },
  "retrieval": {
    "queries_run", "queries_passed", "recall_at_k", "k": 10,
    "with_synthesis": bool,
    "queries": [
      { "id", "question", "passed": bool,
        "retrieved": [ "path/to/article.md", ... ],
        "missing_must": [ "path" ] | null }
    ]
  }
}
```

The YAML golden sets drive the eval harness. Each query passes when any
`any_of` entry appears in the top-K retrieved list, or when every
`must_retrieve` entry appears.

## How every SDK consumes these

1. Load the matching YAML golden set (`golden-realworld.yaml`,
   `golden-hybrid.yaml`, or `golden-distillation.yaml`) into the SDK's
   retrieval test harness.
2. Run each query through the SDK's retrieval pipeline at the mode under
   test (`bm25`, `semantic`, `hybrid`, or `hybrid-rerank`).
3. Compare against the matching JSON fixture for that mode and golden set.
4. Pass criteria:
   - `retrieval.queries[].retrieved` top-K IDs match the fixture, or the
     pass / fail verdict agrees per query.
   - Normalised scores (when surfaced) agree within a documented tolerance
     (guidance: plus or minus 0.02 on RRF fused scores, plus or minus 0.5
     on rerank integer scores).
   - `recall_at_k` matches the fixture exactly.
5. Mode-specific notes: semantic and hybrid-rerank results depend on
   Ollama `bge-m3` and a local rerank model being available. If either
   is absent the SDK must degrade gracefully and skip rather than fail.

## Provenance

- Generated from the reference `jeff` TypeScript implementation at
  `~/code/jeff/research/query-retrieval/` on 2026-04-18 (sourced from
  artefacts dated 2026-04-11 and 2026-04-15).
- The corpus these reports were captured against was the production
  `~/.config/jeff` brain at the time: 5,373 wiki articles and 20,157 raw
  files for the first capture wave, growing to 5,822 / 20,265 for the
  final wave.
- Regeneration is intentional and manual. The referenced script that
  replays the eval suite and writes into this directory is TBD; until
  it lands, regeneration requires rerunning
  `jeff brain eval --suite retrieval --golden <golden>.yaml --mode <mode>`
  against a pinned corpus snapshot and copying the reports back in.

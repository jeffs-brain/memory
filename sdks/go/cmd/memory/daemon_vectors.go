// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/search"
)

// resolveEmbedModel returns the effective embedding model name so the
// VectorIndex and the embedder stay aligned on a single identifier.
// Order of precedence: JB_EMBED_MODEL, provider-specific default, or
// empty when no embedder is configured. Empty disables vector indexing
// downstream.
func resolveEmbedModel(getenv llm.Getenv, embedder llm.Embedder) string {
	if embedder == nil {
		return ""
	}
	if v := strings.TrimSpace(getenv(llm.EnvEmbedModel)); v != "" {
		return v
	}
	// Fall back to the provider-specific default. We cannot ask the
	// embedder interface directly without widening its surface, so we
	// mirror the defaults EmbedderFromEnv uses.
	provider := strings.ToLower(strings.TrimSpace(getenv(llm.EnvEmbedProvider)))
	if provider == "" {
		provider = strings.ToLower(strings.TrimSpace(getenv(llm.EnvProvider)))
	}
	switch provider {
	case "openai":
		return "text-embedding-3-small"
	case "ollama":
		return "bge-m3"
	}
	// Auto-detect (no explicit provider) leans on OpenAI's default when
	// an OpenAI key is present, ollama's default otherwise. Matches the
	// priority order in ProviderFromEnv.
	if strings.TrimSpace(getenv(llm.EnvOpenAIAPIKey)) != "" {
		return "text-embedding-3-small"
	}
	return "bge-m3"
}

// backfillVectors embeds every FTS-indexed path that lacks a vector for
// the configured embedding model and stores the result in vecIdx. Runs
// asynchronously so brain open is not blocked by remote embed calls.
//
// The FTS index is usable throughout; vector retrieval silently returns
// zero hits until the backfill persists entries. Batched embed calls
// keep round-trip overhead amortised and batch failures are skipped
// with a warn-level log so one bad document cannot stall the run.
func backfillVectors(
	ctx context.Context,
	brainID string,
	store brain.Store,
	idx *search.Index,
	vecIdx *search.VectorIndex,
	embedder llm.Embedder,
	model string,
	log *slog.Logger,
) {
	if idx == nil || vecIdx == nil || embedder == nil || model == "" || store == nil {
		return
	}

	paths, err := idx.IndexedPaths()
	if err != nil {
		log.Warn("vectors: list indexed paths", "brain", brainID, "err", err)
		return
	}
	if len(paths) == 0 {
		return
	}

	existing, err := vecIdx.LoadAll(ctx, model)
	if err != nil {
		log.Debug("vectors: loading existing", "brain", brainID, "model", model, "err", err)
	}
	have := make(map[string]bool, len(existing))
	for _, e := range existing {
		have[e.Path] = true
	}

	toEmbed := paths[:0]
	for _, p := range paths {
		if !have[p] {
			toEmbed = append(toEmbed, p)
		}
	}
	if len(toEmbed) == 0 {
		log.Info("vectors: up to date", "brain", brainID, "model", model, "total", len(paths))
		return
	}

	log.Info("vectors: backfill start", "brain", brainID, "model", model, "count", len(toEmbed), "have", len(have))
	started := time.Now()

	const batchSize = 100
	embedded := 0
	for i := 0; i < len(toEmbed); i += batchSize {
		if err := ctx.Err(); err != nil {
			log.Info("vectors: backfill cancelled", "brain", brainID, "embedded", embedded, "err", err)
			return
		}
		end := i + batchSize
		if end > len(toEmbed) {
			end = len(toEmbed)
		}
		batch := toEmbed[i:end]

		texts := make([]string, 0, len(batch))
		keptPaths := make([]string, 0, len(batch))
		for _, p := range batch {
			data, rerr := store.Read(ctx, brain.Path(p))
			if rerr != nil {
				continue
			}
			// Cap per-doc text so one huge raw session cannot blow up
			// the embed call. 8k chars is ~2k tokens, comfortably under
			// most embedding-model context windows.
			text := string(data)
			if len(text) > 8192 {
				text = text[:8192]
			}
			texts = append(texts, text)
			keptPaths = append(keptPaths, p)
		}
		if len(texts) == 0 {
			continue
		}

		vectors, eerr := embedder.Embed(ctx, texts)
		if eerr != nil {
			log.Warn("vectors: embed batch", "brain", brainID, "batch_start", i, "err", eerr)
			continue
		}
		if len(vectors) != len(keptPaths) {
			log.Warn("vectors: embedder returned mismatched count",
				"brain", brainID, "got", len(vectors), "want", len(keptPaths))
			continue
		}

		entries := make([]search.VectorEntry, 0, len(vectors))
		for j, vec := range vectors {
			if len(vec) == 0 {
				continue
			}
			entries = append(entries, search.VectorEntry{
				Path:   keptPaths[j],
				Dim:    len(vec),
				Model:  model,
				Vector: vec,
			})
		}
		if serr := vecIdx.StoreBatch(ctx, entries); serr != nil {
			log.Warn("vectors: store batch", "brain", brainID, "err", serr)
			continue
		}
		embedded += len(entries)
		log.Debug("vectors: batch stored", "brain", brainID, "done", embedded, "total", len(toEmbed))
	}

	log.Info("vectors: backfill done",
		"brain", brainID, "model", model, "embedded", embedded,
		"duration", time.Since(started).Truncate(time.Millisecond))
}

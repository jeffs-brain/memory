// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/search"
)

const vectorBackfillChecksumVersion = "retrieval-text-v2"

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

// backfillVectors embeds every FTS-indexed path that lacks a fresh vector
// for the configured embedding model and stores the result in vecIdx. Runs
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
	if idx == nil || vecIdx == nil || embedder == nil || model == "" {
		return
	}

	checksums, err := idx.IndexedChecksums()
	if err != nil {
		log.Warn("vectors: list indexed checksums", "brain", brainID, "err", err)
		return
	}
	paths := make([]string, 0, len(checksums))
	for path := range checksums {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	existing, err := vecIdx.LoadAll(ctx, model)
	if err != nil {
		log.Warn("vectors: load existing", "brain", brainID, "model", model, "err", err)
	}
	have := make(map[string]search.VectorEntry, len(existing))
	stalePaths := make([]string, 0)
	for _, e := range existing {
		if _, ok := checksums[e.Path]; !ok {
			stalePaths = append(stalePaths, e.Path)
			continue
		}
		have[e.Path] = e
	}

	staleDeleted := 0
	if len(stalePaths) > 0 {
		deleted, derr := vecIdx.DeleteByPaths(ctx, model, stalePaths)
		if derr != nil {
			log.Warn("vectors: delete stale", "brain", brainID, "model", model, "stale", len(stalePaths), "err", derr)
		} else {
			staleDeleted = deleted
		}
	}

	toEmbed := make([]string, 0, len(paths))
	current := 0
	changed := 0
	missing := 0
	for _, p := range paths {
		entry, ok := have[p]
		checksum := vectorBackfillChecksum(checksums[p])
		switch {
		case !ok:
			missing++
			toEmbed = append(toEmbed, p)
		case entry.Checksum != checksum:
			changed++
			toEmbed = append(toEmbed, p)
		default:
			current++
		}
	}
	if len(toEmbed) == 0 {
		log.Info("vectors: up to date",
			"brain", brainID, "model", model,
			"indexed", len(paths), "existing", len(existing), "current", current,
			"stale", len(stalePaths), "stale_deleted", staleDeleted,
		)
		return
	}

	log.Info("vectors: backfill start",
		"brain", brainID, "model", model,
		"indexed", len(paths), "existing", len(existing), "current", current,
		"missing", missing, "changed", changed, "stale", len(stalePaths),
		"stale_deleted", staleDeleted, "to_embed", len(toEmbed),
	)
	started := time.Now()

	const batchSize = 100
	embedded := 0
	readSkipped := 0
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
		keptChecksums := make([]string, 0, len(batch))
		keptRows := make([]search.IndexedRow, 0, len(batch))
		rows, rerr := idx.LookupRows(ctx, batch)
		if rerr != nil {
			log.Warn("vectors: lookup indexed rows", "brain", brainID, "batch_start", i, "err", rerr)
			readSkipped += len(batch)
			continue
		}
		byPath := make(map[string]search.IndexedRow, len(rows))
		for _, row := range rows {
			byPath[row.Path] = row
		}
		for _, p := range batch {
			row, ok := byPath[p]
			if !ok {
				readSkipped++
				log.Debug("vectors: indexed path missing during vector backfill", "brain", brainID, "path", p)
				continue
			}
			text := truncateVectorBackfillText(vectorBackfillText(row), 8192)
			texts = append(texts, text)
			keptPaths = append(keptPaths, p)
			keptChecksums = append(keptChecksums, vectorBackfillChecksum(checksums[p]))
			keptRows = append(keptRows, row)
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
				Path:     keptPaths[j],
				Checksum: keptChecksums[j],
				Dim:      len(vec),
				Model:    model,
				Vector:   vec,
				Title:    keptRows[j].Title,
				Summary:  keptRows[j].Summary,
				Topic:    keptRows[j].Scope,
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
		"attempted", len(toEmbed), "read_skipped", readSkipped,
		"current", current, "missing", missing, "changed", changed,
		"stale", len(stalePaths), "stale_deleted", staleDeleted,
		"duration", time.Since(started).Truncate(time.Millisecond))
}

func vectorBackfillChecksum(indexChecksum string) string {
	return vectorBackfillChecksumVersion + ":" + indexChecksum
}

func vectorBackfillText(row search.IndexedRow) string {
	parts := make([]string, 0, 8)
	appendPart := func(label, value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		parts = append(parts, label+": "+value)
	}
	appendPart("Title", row.Title)
	appendPart("Summary", row.Summary)
	appendPart("Scope", row.Scope)
	appendPart("Project", row.ProjectSlug)
	appendPart("Session date", row.SessionDate)
	appendPart("Tags", row.Tags)
	appendPart("Source role", row.SourceRole)
	appendPart("Content", row.Content)
	return strings.Join(parts, "\n")
}

func truncateVectorBackfillText(text string, limit int) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	const marker = "\n[...middle truncated for embedding...]\n"
	if limit <= len(marker)+2 {
		return text[:limit]
	}
	headLen := (limit - len(marker)) / 2
	tailLen := limit - len(marker) - headLen
	return text[:headLen] + marker + text[len(text)-tailLen:]
}

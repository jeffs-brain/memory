// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// rawDocumentsPrefix is the logical prefix under which [Base.Ingest]
// persists every document. Compile walks this tree to locate ingested
// documents that have not yet been chunked.
//
// Kept deliberately narrow: the package does not compete with jeff's
// rich compile pipeline; it focuses on the minimum surface required by
// the Go SDK (markdown + plain-text chunk segmentation).
var rawDocumentsPrefix = brain.RawDocumentsPrefix()

// defaultChunkMinChars is the minimum character length of a segmentation
// output. Short paragraphs below this floor are merged with the
// preceding chunk so the index does not get flooded with single-line
// stubs.
const defaultChunkMinChars = 120

// defaultChunkMaxChars bounds the biggest chunk. Longer segments are
// split at sentence boundaries when possible.
const defaultChunkMaxChars = 1800

// Compile implements [Base].
//
// Walks raw/documents (or the explicit Paths list) and feeds every
// document through the segmenter. When DryRun is set, the chunks are
// counted but not forwarded to the search index.
func (k *kbase) Compile(ctx context.Context, opts CompileOptions) (CompileResult, error) {
	start := time.Now()
	if err := k.requireStore(); err != nil {
		return CompileResult{}, err
	}

	targets, err := k.resolveCompileTargets(ctx, opts.Paths)
	if err != nil {
		return CompileResult{}, err
	}

	var res CompileResult
	for i, p := range targets {
		if opts.MaxBatch > 0 && i >= opts.MaxBatch {
			break
		}
		if ctx.Err() != nil {
			return res, ctx.Err()
		}
		data, err := k.store.Read(ctx, p)
		if err != nil {
			res.Errors++
			continue
		}
		doc := documentFromStored(p, data)
		if doc == nil {
			res.Skipped++
			continue
		}
		if opts.DryRun {
			chunks := segmentDocument(doc)
			res.Compiled++
			res.Chunks += len(chunks)
			continue
		}
		n, err := k.chunkAndIndex(ctx, doc)
		if err != nil {
			res.Errors++
			continue
		}
		res.Compiled++
		res.Chunks += n
	}
	res.Elapsed = time.Since(start)
	return res, nil
}

// chunkAndIndex segments the document, updates the search index when
// bound, and returns the number of chunks produced.
func (k *kbase) chunkAndIndex(ctx context.Context, doc *Document) (int, error) {
	chunks := segmentDocument(doc)
	idx, _, _ := k.snapshot()
	if idx == nil {
		return len(chunks), nil
	}
	// The Go SDK's search.Index reads directly from the brain store
	// via Update(). The document is already persisted by the time we
	// get here, so ask the index to reindex that single path. Ignore
	// update failures here: the next call to Update() heals any gap.
	if err := idx.Update(ctx); err != nil {
		return len(chunks), fmt.Errorf("knowledge: search index update: %w", err)
	}
	return len(chunks), nil
}

// resolveCompileTargets returns the list of logical paths the compile
// pass should walk. Uses the caller's explicit set when supplied,
// otherwise walks raw/documents in the brain store.
func (k *kbase) resolveCompileTargets(ctx context.Context, explicit []brain.Path) ([]brain.Path, error) {
	if len(explicit) > 0 {
		return explicit, nil
	}
	entries, err := k.store.List(ctx, rawDocumentsPrefix, brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		return nil, fmt.Errorf("knowledge: listing raw documents: %w", err)
	}
	out := make([]brain.Path, 0, len(entries))
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		if !strings.HasSuffix(string(e.Path), ".md") {
			continue
		}
		out = append(out, e.Path)
	}
	return out, nil
}

// documentFromStored reconstructs a Document from a persisted markdown
// file. The frontmatter drives the title/summary/tags and the body
// below the frontmatter fences is used for chunking.
func documentFromStored(p brain.Path, data []byte) *Document {
	fm, body := ParseFrontmatter(string(data))
	title := firstNonEmpty(fm.Title, fm.Name)
	if title == "" {
		title = strings.TrimSuffix(lastSegment(string(p)), ".md")
	}
	return &Document{
		ID:          hashSlug(data),
		Path:        p,
		Title:       title,
		Source:      fm.Source,
		ContentType: contentTypeMarkdown,
		Tags:        append([]string{}, fm.Tags...),
		Summary:     firstNonEmpty(fm.Summary, fm.Description),
		Body:        strings.TrimSpace(body),
		Bytes:       len(data),
	}
}

// segmentDocument splits a document body into [Chunk]s. Markdown
// headings mark chunk boundaries; within a heading, large paragraphs
// are split by sentence.
//
// This mirrors the simple path in jeff's compile.go: the wiki-linking,
// two-phase planner, and LLM summarisation passes are intentionally
// skipped. They depend on jeff's richer subsystems (llm provider wiring,
// wiki index, stats) that do not belong in this minimal port.
//
// TODO(next): port the richer chunker from
// jeff/apps/jeff/internal/knowledge/compile.go once the SDK wires up an
// llm.Provider. The current segmenter is deterministic and adequate for
// the Go-side search index; it is not a faithful reproduction of the
// upstream compile pipeline.
func segmentDocument(doc *Document) []Chunk {
	if doc == nil || strings.TrimSpace(doc.Body) == "" {
		return nil
	}

	sections := splitByHeadings(doc.Body)
	out := make([]Chunk, 0, len(sections))
	ordinal := 0
	for _, sec := range sections {
		pieces := splitLong(sec.text, defaultChunkMaxChars)
		for _, piece := range pieces {
			piece = strings.TrimSpace(piece)
			if piece == "" {
				continue
			}
			out = append(out, Chunk{
				DocumentID: doc.ID,
				Ordinal:    ordinal,
				Heading:    sec.heading,
				Text:       piece,
				Tokens:     estimateTokens(piece),
			})
			ordinal++
		}
	}
	return mergeSmallChunks(out, defaultChunkMinChars)
}

// headingSection is one logical section split off the document body.
type headingSection struct {
	heading string
	text    string
}

// splitByHeadings walks the body line-by-line and emits one section per
// `# heading` block (any level). Lines before the first heading form an
// initial section with an empty heading.
func splitByHeadings(body string) []headingSection {
	lines := strings.Split(body, "\n")
	var (
		current headingSection
		out     []headingSection
		buf     strings.Builder
	)
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		current.text = strings.TrimSpace(buf.String())
		out = append(out, current)
		buf.Reset()
	}
	for _, line := range lines {
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "#") && isHeadingLine(trim) {
			flush()
			current = headingSection{heading: strings.TrimSpace(strings.TrimLeft(trim, "# "))}
			continue
		}
		buf.WriteString(line)
		buf.WriteByte('\n')
	}
	flush()
	return out
}

// isHeadingLine reports whether line is a markdown heading. Defensive
// check: requires a run of `#` followed by a space.
func isHeadingLine(line string) bool {
	i := 0
	for i < len(line) && line[i] == '#' {
		i++
	}
	if i == 0 || i > 6 {
		return false
	}
	return i < len(line) && line[i] == ' '
}

// splitLong slices text at sentence boundaries when it exceeds maxChars.
// Falls back to hard slicing when no sentence boundary is reachable
// within the window.
func splitLong(text string, maxChars int) []string {
	text = strings.TrimSpace(text)
	if len(text) <= maxChars {
		return []string{text}
	}
	var out []string
	remaining := text
	for len(remaining) > maxChars {
		cut := findSentenceCut(remaining, maxChars)
		if cut <= 0 {
			cut = maxChars
		}
		piece := strings.TrimSpace(remaining[:cut])
		if piece != "" {
			out = append(out, piece)
		}
		remaining = strings.TrimSpace(remaining[cut:])
	}
	if strings.TrimSpace(remaining) != "" {
		out = append(out, strings.TrimSpace(remaining))
	}
	return out
}

// findSentenceCut returns an index inside text where a sentence ends
// within the window [maxChars*0.6, maxChars]. Returns 0 when no
// boundary is reachable.
func findSentenceCut(text string, maxChars int) int {
	lower := int(float64(maxChars) * 0.6)
	if lower < 120 {
		lower = 120
	}
	if lower > len(text) {
		return 0
	}
	window := text[lower:maxChars]
	// Prefer paragraph breaks, then sentence endings.
	if idx := strings.Index(window, "\n\n"); idx >= 0 {
		return lower + idx + 2
	}
	for _, sep := range []string{". ", "! ", "? "} {
		if idx := strings.LastIndex(window, sep); idx >= 0 {
			return lower + idx + len(sep)
		}
	}
	return 0
}

// mergeSmallChunks folds chunks shorter than minChars into the previous
// chunk so the index never sees single-line stubs.
func mergeSmallChunks(in []Chunk, minChars int) []Chunk {
	if len(in) == 0 {
		return in
	}
	out := make([]Chunk, 0, len(in))
	for _, c := range in {
		if len(out) > 0 && len(c.Text) < minChars {
			last := out[len(out)-1]
			last.Text = last.Text + "\n\n" + c.Text
			last.Tokens = estimateTokens(last.Text)
			out[len(out)-1] = last
			continue
		}
		out = append(out, c)
	}
	// Recompute ordinals so they stay contiguous after merging.
	for i := range out {
		out[i].Ordinal = i
	}
	return out
}

// estimateTokens is a coarse byte/4 approximation used only to set
// [Chunk.Tokens]. Mirrors the rule of thumb jeff uses when a real
// tokeniser is not wired in.
func estimateTokens(text string) int {
	return (len(text) + 3) / 4
}

// lastSegment returns the trailing path segment.
func lastSegment(p string) string {
	if idx := strings.LastIndex(p, "/"); idx >= 0 {
		return p[idx+1:]
	}
	return p
}

// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/jeffs-brain/memory/go/brain"
	storefs "github.com/jeffs-brain/memory/go/store/fs"

	"github.com/jeffs-brain/memory/go/knowledge"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

// ProgressEmitter is invoked by long-running tool handlers to emit
// interim progress to the MCP client. A nil emitter means the caller
// did not supply a progress token; handlers skip the call in that
// case.
type ProgressEmitter func(progress float64, message string)

// MemoryClient is the unified dispatch surface every tool handler
// speaks against. The local and hosted implementations share the same
// method set so handlers never branch on mode themselves.
type MemoryClient interface {
	Mode() Mode

	Remember(ctx context.Context, args RememberArgs) (map[string]any, error)
	Search(ctx context.Context, args SearchArgs) (map[string]any, error)
	Recall(ctx context.Context, args RecallArgs) (map[string]any, error)
	Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (map[string]any, error)
	IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error)
	IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (map[string]any, error)
	Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (map[string]any, error)
	Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (map[string]any, error)
	Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (map[string]any, error)
	CreateBrain(ctx context.Context, args CreateBrainArgs) (map[string]any, error)
	ListBrains(ctx context.Context) (map[string]any, error)

	Close() error
}

// RememberArgs captures the schema-validated input for memory_remember.
type RememberArgs struct {
	Content string
	Title   string
	Brain   string
	Tags    []string
	Path    string
}

// SearchArgs captures the schema-validated input for memory_search.
type SearchArgs struct {
	Query string
	Brain string
	TopK  int
	Scope string
	Sort  string
}

// RecallArgs captures the schema-validated input for memory_recall.
type RecallArgs struct {
	Query     string
	Brain     string
	Scope     string
	SessionID string
	TopK      int
}

// AskArgs captures the schema-validated input for memory_ask.
type AskArgs struct {
	Query string
	Brain string
	TopK  int
}

// IngestFileArgs captures the schema-validated input for memory_ingest_file.
type IngestFileArgs struct {
	Path  string
	Brain string
	As    string
}

// IngestURLArgs captures the schema-validated input for memory_ingest_url.
type IngestURLArgs struct {
	URL   string
	Brain string
}

// ExtractMessage is one turn supplied to memory_extract.
type ExtractMessage struct {
	Role    string
	Content string
}

// ExtractArgs captures the schema-validated input for memory_extract.
type ExtractArgs struct {
	Messages  []ExtractMessage
	Brain     string
	ActorID   string
	SessionID string
}

// ReflectArgs captures the schema-validated input for memory_reflect.
type ReflectArgs struct {
	SessionID string
	Brain     string
}

// ConsolidateArgs captures the schema-validated input for memory_consolidate.
type ConsolidateArgs struct {
	Brain string
}

// CreateBrainArgs captures the schema-validated input for memory_create_brain.
type CreateBrainArgs struct {
	Name       string
	Slug       string
	Visibility string
}

const (
	fileIngestLimit = 25 * 1024 * 1024
	urlFetchLimit   = 5 * 1024 * 1024
	defaultBrainID  = "default"
)

// NewMemoryClient dispatches to the local or hosted implementation
// based on the resolved [Config].
func NewMemoryClient(cfg Config, logger *slog.Logger) (MemoryClient, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if cfg.Mode == ModeHosted {
		return newHostedClient(cfg, logger), nil
	}
	return newLocalClient(cfg, logger)
}

// ---------------------------------------------------------------------
// Local mode
// ---------------------------------------------------------------------

// localBrain bundles the runtime resources for a single brain.
type localBrain struct {
	id        string
	root      string
	store     brain.Store
	index     *search.Index
	retriever retrieval.Retriever
	memory    *memory.Memory
	knowledge knowledge.Base
}

// close tears down the cached entries for the brain. Closing is
// best-effort: search.CloseDB is refcounted so multiple brains sharing
// the same on-disk index close cleanly.
func (b *localBrain) close() {
	if b == nil {
		return
	}
	if b.store != nil {
		_ = b.store.Close()
	}
}

type localClient struct {
	cfg      Config
	log      *slog.Logger
	provider llm.Provider
	embedder llm.Embedder

	mu     sync.Mutex
	brains map[string]*localBrain
}

// newLocalClient wires in-process resources. Provider and embedder are
// resolved from the environment via the llm package; either may be
// absent and the client degrades gracefully.
func newLocalClient(cfg Config, logger *slog.Logger) (*localClient, error) {
	if err := os.MkdirAll(cfg.BrainRoot, 0o755); err != nil {
		return nil, fmt.Errorf("memory-mcp: creating %s: %w", cfg.BrainRoot, err)
	}
	provider, perr := llm.ProviderFromEnv(llm.OSGetenv)
	if perr != nil {
		logger.Warn("memory-mcp: resolving llm provider", "err", perr)
		provider = llm.NewFake([]string{"ok"})
	}
	embedder, eerr := llm.EmbedderFromEnv(llm.OSGetenv)
	if eerr != nil {
		logger.Debug("memory-mcp: resolving embedder", "err", eerr)
		embedder = nil
	}
	return &localClient{
		cfg:      cfg,
		log:      logger,
		provider: provider,
		embedder: embedder,
		brains:   map[string]*localBrain{},
	}, nil
}

// Mode implements [MemoryClient].
func (c *localClient) Mode() Mode { return ModeLocal }

// Close releases every cached brain and closes the shared LLM
// provider + embedder. Safe to call multiple times.
func (c *localClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for id, br := range c.brains {
		br.close()
		delete(c.brains, id)
	}
	if c.provider != nil {
		_ = c.provider.Close()
		c.provider = nil
	}
	if c.embedder != nil {
		_ = c.embedder.Close()
		c.embedder = nil
	}
	return nil
}

// resolveBrainID picks the brain ID for an incoming tool call.
func (c *localClient) resolveBrainID(override string) string {
	if override = strings.TrimSpace(override); override != "" {
		return override
	}
	if c.cfg.DefaultBrain != "" {
		return c.cfg.DefaultBrain
	}
	return defaultBrainID
}

// brainRoot returns the on-disk root for a brain.
func (c *localClient) brainRoot(id string) string {
	return filepath.Join(c.cfg.BrainRoot, "brains", sanitiseBrainID(id))
}

// openBrain returns a cached [localBrain] for id, constructing one on
// first use. Creation ensures the brain directory exists so local-mode
// callers can remember/search before an explicit create_brain call.
func (c *localClient) openBrain(ctx context.Context, id string) (*localBrain, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if br, ok := c.brains[id]; ok {
		return br, nil
	}
	root := c.brainRoot(id)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("memory-mcp: mkdir %s: %w", root, err)
	}
	store, err := storefs.New(root)
	if err != nil {
		return nil, fmt.Errorf("memory-mcp: fs store %s: %w", root, err)
	}
	dbPath := filepath.Join(root, ".search.db")
	db, derr := search.OpenDB(dbPath)
	if derr != nil {
		_ = store.Close()
		return nil, fmt.Errorf("memory-mcp: open search db: %w", derr)
	}
	idx, ierr := search.NewIndex(db, store)
	if ierr != nil {
		_ = search.CloseDB(db)
		_ = store.Close()
		return nil, fmt.Errorf("memory-mcp: build search index: %w", ierr)
	}
	var (
		src retrieval.Source
		ret retrieval.Retriever
	)
	if s, serr := retrieval.NewIndexSource(idx, retrieval.IndexSourceOptions{Embedder: c.embedder}); serr == nil {
		src = s
		if r, rerr := retrieval.New(retrieval.Config{Source: s, Embedder: c.embedder}); rerr == nil {
			ret = r
		}
	}
	_ = src
	kb, kerr := knowledge.New(knowledge.Options{
		BrainID:   id,
		Store:     store,
		Index:     idx,
		Retriever: ret,
	})
	if kerr != nil {
		_ = search.CloseDB(db)
		_ = store.Close()
		return nil, fmt.Errorf("memory-mcp: knowledge base: %w", kerr)
	}
	// Subscribe so writes flow into the FTS table without an explicit
	// compile pass, then trigger an initial scan for any pre-existing
	// content on disk.
	_ = idx.Subscribe(store)
	go func() {
		if err := idx.Update(context.Background()); err != nil {
			c.log.Debug("memory-mcp: initial index update failed", "brain", id, "err", err)
		}
	}()
	br := &localBrain{
		id:        id,
		root:      root,
		store:     store,
		index:     idx,
		retriever: ret,
		memory:    memory.New(store),
		knowledge: kb,
	}
	c.brains[id] = br
	return br, nil
}

// Remember implements [MemoryClient].
func (c *localClient) Remember(ctx context.Context, args RememberArgs) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	title := deriveTitle(args.Content, args.Title)
	content := []byte(args.Content)
	req := knowledge.IngestRequest{
		BrainID:     brainID,
		ContentType: "text/markdown",
		Title:       title,
		Tags:        append([]string{}, args.Tags...),
		Content:     bytes.NewReader(content),
	}
	resp, err := br.knowledge.Ingest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("memory_remember: ingest: %w", err)
	}
	out := map[string]any{
		"id":          resp.DocumentID,
		"path":        string(resp.Path),
		"byte_size":   resp.Bytes,
		"chunk_count": resp.ChunkCount,
		"took_ms":     resp.TookMs,
		"brain_id":    brainID,
	}
	if len(args.Tags) > 0 {
		out["tags"] = args.Tags
	}
	return out, nil
}

// Search implements [MemoryClient].
func (c *localClient) Search(ctx context.Context, args SearchArgs) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	topK := args.TopK
	if topK <= 0 {
		topK = 10
	}
	started := time.Now()
	// Refresh first so writes from the same session show up.
	_ = br.index.Update(ctx)
	sort := search.ParseSortMode(args.Sort)
	results, err := br.index.Search(args.Query, search.SearchOpts{
		Scope:      mapScopeToSearch(args.Scope),
		MaxResults: topK,
		Sort:       sort,
	})
	if err != nil {
		return nil, fmt.Errorf("memory_search: %w", err)
	}
	hits := make([]map[string]any, 0, len(results))
	for _, r := range results {
		hits = append(hits, map[string]any{
			"score":    r.Score,
			"path":     r.Path,
			"title":    r.Title,
			"summary":  r.Summary,
			"content":  r.Snippet,
			"chunk_id": r.Path,
			"scope":    r.Scope,
		})
	}
	return map[string]any{
		"query":    args.Query,
		"brain_id": brainID,
		"hits":     hits,
		"took_ms":  time.Since(started).Milliseconds(),
	}, nil
}

// Recall implements [MemoryClient].
func (c *localClient) Recall(ctx context.Context, args RecallArgs) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	topK := args.TopK
	if topK <= 0 {
		topK = 5
	}
	if br.retriever != nil {
		resp, rerr := br.retriever.Retrieve(ctx, retrieval.Request{
			Query:   args.Query,
			TopK:    topK,
			Mode:    retrieval.ModeAuto,
			BrainID: brainID,
			Filters: retrieval.Filters{Scope: mapScopeToSearch(args.Scope)},
		})
		if rerr == nil {
			chunks := make([]map[string]any, 0, len(resp.Chunks))
			for _, hit := range resp.Chunks {
				chunks = append(chunks, map[string]any{
					"chunk_id":   hit.ChunkID,
					"document_id": hit.DocumentID,
					"score":      hit.Score,
					"path":       hit.Path,
					"content":    hit.Text,
					"title":      hit.Title,
					"summary":    hit.Summary,
				})
			}
			out := map[string]any{
				"query":      args.Query,
				"brain_id":   brainID,
				"session_id": args.SessionID,
				"chunks":     chunks,
			}
			return out, nil
		}
		c.log.Debug("memory_recall: retriever failed, falling back to search", "err", rerr)
	}
	// Fallback: mirror memory_search.
	return c.Search(ctx, SearchArgs{
		Query: args.Query,
		Brain: args.Brain,
		TopK:  topK,
		Scope: args.Scope,
	})
}

// Ask implements [MemoryClient]. Runs retrieval then a single blocking
// LLM completion. Progress notifications are coarse (retrieved +
// answered) until the SDK exposes a streaming `ask` surface.
func (c *localClient) Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	topK := args.TopK
	if topK <= 0 {
		topK = 8
	}
	var chunks []retrieval.RetrievedChunk
	if br.retriever != nil {
		resp, rerr := br.retriever.Retrieve(ctx, retrieval.Request{
			Query:   args.Query,
			TopK:    topK,
			Mode:    retrieval.ModeAuto,
			BrainID: brainID,
		})
		if rerr == nil {
			chunks = resp.Chunks
		}
	}
	if chunks == nil && br.index != nil {
		hits, herr := br.index.Search(args.Query, search.SearchOpts{MaxResults: topK})
		if herr == nil {
			for _, h := range hits {
				chunks = append(chunks, retrieval.RetrievedChunk{
					ChunkID:    h.Path,
					DocumentID: h.Path,
					Path:       h.Path,
					Score:      h.Score,
					Text:       h.Snippet,
					Title:      h.Title,
					Summary:    h.Summary,
				})
			}
		}
	}
	if progress != nil {
		progress(0, "retrieved")
	}
	if c.provider == nil {
		return nil, errors.New("memory_ask: no LLM provider configured")
	}
	var contextBlock strings.Builder
	for i, hit := range chunks {
		fmt.Fprintf(&contextBlock, "### [%d] %s\n%s\n\n", i+1, hit.Path, truncate(hit.Text, 2000))
	}
	prompt := fmt.Sprintf(
		"Answer the user question using the retrieved memory chunks below.\n"+
			"Cite supporting chunks inline using the numeric markers [1], [2], etc.\n"+
			"If the chunks do not answer the question, say so plainly.\n\n"+
			"## Question\n%s\n\n## Retrieved memory\n%s",
		args.Query,
		firstNonEmpty(contextBlock.String(), "_no relevant chunks_"),
	)
	resp, err := c.provider.Complete(ctx, llm.CompleteRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "You are Jeff, a personal memory assistant. Use British English."},
			{Role: llm.RoleUser, Content: prompt},
		},
		MaxTokens:   1024,
		Temperature: 0.2,
	})
	if err != nil {
		return nil, fmt.Errorf("memory_ask: llm: %w", err)
	}
	if progress != nil {
		progress(1, "answered")
	}
	citations := make([]map[string]any, 0, len(chunks))
	for i, hit := range chunks {
		if i >= 5 {
			break
		}
		citations = append(citations, map[string]any{
			"type":         "citation",
			"chunk_id":     hit.ChunkID,
			"document_id":  hit.DocumentID,
			"answer_start": 0,
			"answer_end":   0,
			"quote":        truncate(hit.Text, 200),
		})
	}
	retrievedChunks := make([]map[string]any, 0, len(chunks))
	for _, hit := range chunks {
		retrievedChunks = append(retrievedChunks, map[string]any{
			"chunk_id":    hit.ChunkID,
			"document_id": hit.DocumentID,
			"score":       hit.Score,
			"preview":     truncate(hit.Text, 512),
		})
	}
	return map[string]any{
		"answer":    resp.Text,
		"citations": citations,
		"retrieved": retrievedChunks,
	}, nil
}

// IngestFile implements [MemoryClient].
func (c *localClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	abs := args.Path
	if !filepath.IsAbs(abs) {
		resolved, rerr := filepath.Abs(abs)
		if rerr != nil {
			return nil, fmt.Errorf("memory_ingest_file: resolve path: %w", rerr)
		}
		abs = resolved
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("memory_ingest_file: stat %s: %w", abs, err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("memory_ingest_file: not a regular file: %s", abs)
	}
	if info.Size() > fileIngestLimit {
		return nil, errors.New("file_too_large: 25 MiB limit exceeded")
	}
	if progress != nil {
		progress(0, "read")
	}
	req := knowledge.IngestRequest{
		BrainID:     brainID,
		Path:        abs,
		ContentType: mimeByHint(abs, args.As),
	}
	resp, ierr := br.knowledge.Ingest(ctx, req)
	if ierr != nil {
		return nil, fmt.Errorf("memory_ingest_file: %w", ierr)
	}
	if progress != nil {
		progress(1, "indexed")
	}
	return map[string]any{
		"status":         "completed",
		"document_id":    resp.DocumentID,
		"path":           string(resp.Path),
		"hash":           hashString([]byte(resp.DocumentID)),
		"chunk_count":    resp.ChunkCount,
		"embedded_count": resp.ChunkCount,
		"duration_ms":    resp.TookMs,
		"reused":         false,
	}, nil
}

// IngestURL implements [MemoryClient].
func (c *localClient) IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "fetching")
	}
	resp, ierr := br.knowledge.IngestURL(ctx, args.URL)
	if ierr != nil {
		return nil, fmt.Errorf("memory_ingest_url: %w", ierr)
	}
	if progress != nil {
		progress(1, "indexed")
	}
	return map[string]any{
		"path": "server",
		"result": map[string]any{
			"status":         "completed",
			"document_id":    resp.DocumentID,
			"path":           string(resp.Path),
			"chunk_count":    resp.ChunkCount,
			"embedded_count": resp.ChunkCount,
			"duration_ms":    resp.TookMs,
			"reused":         false,
		},
	}, nil
}

// Extract implements [MemoryClient]. Local mode mirrors the TS wrapper's
// session and transcript shapes; we do not drive the extraction LLM
// here because the memory package already exposes [ExtractFromMessages]
// for the richer case. The Go SDK port for the bare transcript path
// records the conversation as a document so downstream consolidation
// can pick it up.
func (c *localClient) Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "extracting")
	}
	messages := make([]memory.Message, 0, len(args.Messages))
	for _, m := range args.Messages {
		messages = append(messages, memory.Message{
			Role:    llm.Role(m.Role),
			Content: m.Content,
		})
	}
	if args.SessionID != "" {
		extracted, xerr := memory.ExtractFromMessages(ctx, c.provider, "", br.memory, "", messages)
		if xerr != nil {
			return nil, fmt.Errorf("memory_extract: %w", xerr)
		}
		out := make([]map[string]any, 0, len(extracted))
		for i, e := range extracted {
			out = append(out, map[string]any{
				"id":         fmt.Sprintf("%s:%d", args.SessionID, i),
				"session_id": args.SessionID,
				"role":       "assistant",
				"content":    e.Content,
				"created_at": time.Now().UTC().Format(time.RFC3339),
			})
		}
		if progress != nil {
			progress(1, fmt.Sprintf("%d memories", len(out)))
		}
		return map[string]any{"mode": "session", "messages": out}, nil
	}
	// Transcript mode: persist a markdown transcript so downstream
	// consolidation can pick it up.
	var body strings.Builder
	for _, m := range args.Messages {
		fmt.Fprintf(&body, "[%s]: %s\n\n", m.Role, m.Content)
	}
	req := knowledge.IngestRequest{
		BrainID:     brainID,
		ContentType: "text/markdown",
		Title:       fmt.Sprintf("Transcript %s", time.Now().UTC().Format(time.RFC3339)),
		Content:     bytes.NewReader([]byte(body.String())),
	}
	resp, xerr := br.knowledge.Ingest(ctx, req)
	if xerr != nil {
		return nil, fmt.Errorf("memory_extract: transcript: %w", xerr)
	}
	if progress != nil {
		progress(1, "recorded")
	}
	return map[string]any{
		"mode": "transcript",
		"document": map[string]any{
			"id":              resp.DocumentID,
			"brain_id":        brainID,
			"title":           req.Title,
			"path":            string(resp.Path),
			"source":          "extract",
			"content_type":    "text/markdown",
			"byte_size":       resp.Bytes,
			"checksum_sha256": hashString([]byte(resp.DocumentID)),
			"metadata":        map[string]any{"message_count": len(args.Messages)},
			"created_at":      time.Now().UTC().Format(time.RFC3339),
			"updated_at":      time.Now().UTC().Format(time.RFC3339),
			"deleted_at":      nil,
		},
	}, nil
}

// Reflect implements [MemoryClient]. Local mode runs
// [memory.Reflector.ForceReflect] so callers can exercise the surface
// end-to-end without a dedicated session backend.
func (c *localClient) Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "reflecting")
	}
	reflector := memory.NewReflector(br.memory)
	result := reflector.ForceReflect(ctx, c.provider, "", "", nil)
	if progress != nil {
		progress(1, "done")
	}
	if result == nil {
		return map[string]any{
			"reflection_status":    "no_result",
			"reflection_attempted": true,
			"ended_at":             time.Now().UTC().Format(time.RFC3339),
		}, nil
	}
	return map[string]any{
		"reflection_status": "completed",
		"reflection": map[string]any{
			"outcome":               result.Outcome,
			"should_record_episode": result.ShouldRecordEpisode,
			"path":                  "",
			"retry_feedback":        result.RetryFeedback,
		},
		"reflection_attempted": true,
		"ended_at":             time.Now().UTC().Format(time.RFC3339),
	}, nil
}

// Consolidate implements [MemoryClient]. Local mode falls back to
// [knowledge.Base.Compile] which re-chunks and re-indexes every
// persisted document in the brain.
func (c *localClient) Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "consolidating")
	}
	result, cerr := br.knowledge.Compile(ctx, knowledge.CompileOptions{})
	if cerr != nil {
		return nil, fmt.Errorf("memory_consolidate: %w", cerr)
	}
	if progress != nil {
		progress(1, "done")
	}
	return map[string]any{
		"result": map[string]any{
			"compiled": result.Compiled,
			"chunks":   result.Chunks,
			"skipped":  result.Skipped,
			"errors":   result.Errors,
			"elapsed_ms": result.Elapsed.Milliseconds(),
		},
	}, nil
}

// CreateBrain implements [MemoryClient]. Local mode writes a
// config.json alongside the brain root so [ListBrains] can surface
// structured metadata.
func (c *localClient) CreateBrain(ctx context.Context, args CreateBrainArgs) (map[string]any, error) {
	slug := strings.TrimSpace(args.Slug)
	if slug == "" {
		slug = slugify(args.Name)
	}
	if slug == "" {
		return nil, errors.New("memory_create_brain: slug empty")
	}
	root := c.brainRoot(slug)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("memory_create_brain: mkdir %s: %w", root, err)
	}
	visibility := strings.TrimSpace(args.Visibility)
	if visibility == "" {
		visibility = "private"
	}
	cfg := map[string]any{
		"version":    1,
		"name":       args.Name,
		"slug":       slug,
		"visibility": visibility,
		"createdAt":  time.Now().UTC().Format(time.RFC3339),
	}
	body, _ := json.MarshalIndent(cfg, "", "  ")
	if err := os.WriteFile(filepath.Join(root, "config.json"), append(body, '\n'), 0o644); err != nil {
		return nil, fmt.Errorf("memory_create_brain: write config: %w", err)
	}
	return map[string]any{
		"id":         slug,
		"slug":       slug,
		"name":       args.Name,
		"visibility": visibility,
		"created_at": cfg["createdAt"],
	}, nil
}

// ListBrains implements [MemoryClient].
func (c *localClient) ListBrains(ctx context.Context) (map[string]any, error) {
	dir := filepath.Join(c.cfg.BrainRoot, "brains")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]any{"items": []any{}}, nil
		}
		return nil, fmt.Errorf("memory_list_brains: %w", err)
	}
	items := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		cfgPath := filepath.Join(dir, e.Name(), "config.json")
		parsed := map[string]any{}
		if raw, rerr := os.ReadFile(cfgPath); rerr == nil {
			_ = json.Unmarshal(raw, &parsed)
		}
		slug := stringOrDefault(parsed, "slug", e.Name())
		items = append(items, map[string]any{
			"id":         slug,
			"slug":       slug,
			"name":       stringOrDefault(parsed, "name", e.Name()),
			"visibility": stringOrDefault(parsed, "visibility", "private"),
			"created_at": parsed["createdAt"],
		})
	}
	return map[string]any{"items": items}, nil
}

// ---------------------------------------------------------------------
// Hosted mode
// ---------------------------------------------------------------------

type hostedClient struct {
	cfg        Config
	log        *slog.Logger
	httpClient *http.Client
}

func newHostedClient(cfg Config, logger *slog.Logger) *hostedClient {
	return &hostedClient{
		cfg:        cfg,
		log:        logger,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// Mode implements [MemoryClient].
func (c *hostedClient) Mode() Mode { return ModeHosted }

// Close implements [MemoryClient]. Hosted mode holds no persistent
// resources so Close is a no-op.
func (c *hostedClient) Close() error { return nil }

// resolveBrainID returns the brain ID or raises a wire-visible error
// so callers see the same shape the spec mandates.
func (c *hostedClient) resolveBrainID(override string) (string, error) {
	if override = strings.TrimSpace(override); override != "" {
		return override, nil
	}
	if c.cfg.DefaultBrain != "" {
		return c.cfg.DefaultBrain, nil
	}
	return "", errors.New("memory-mcp: brain id required in hosted mode; set JB_BRAIN or pass `brain`")
}

// doJSON performs a JSON request and returns the decoded body as a map.
func (c *hostedClient) doJSON(ctx context.Context, method, path string, body any) (map[string]any, error) {
	var buf io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("memory-mcp: encoding %s %s: %w", method, path, err)
		}
		buf = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.cfg.Endpoint+path, buf)
	if err != nil {
		return nil, fmt.Errorf("memory-mcp: building request: %w", err)
	}
	req.Header.Set("authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("accept", "application/json")
	if body != nil {
		req.Header.Set("content-type", "application/json")
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("memory-mcp: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024*1024))
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("memory-mcp: %s %s: HTTP %d: %s", method, path, resp.StatusCode, truncate(string(raw), 256))
	}
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("memory-mcp: decoding %s %s: %w", method, path, err)
	}
	return out, nil
}

// Remember implements [MemoryClient].
func (c *hostedClient) Remember(ctx context.Context, args RememberArgs) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"title":   args.Title,
		"content": args.Content,
	}
	if args.Path != "" {
		body["path"] = args.Path
	}
	if len(args.Tags) > 0 {
		body["metadata"] = map[string]any{"tags": strings.Join(args.Tags, ",")}
	}
	return c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/documents"), body)
}

// Search implements [MemoryClient].
func (c *hostedClient) Search(ctx context.Context, args SearchArgs) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	qs := url.Values{}
	qs.Set("q", args.Query)
	if args.TopK > 0 {
		qs.Set("top_k", fmt.Sprintf("%d", args.TopK))
	}
	if args.Scope != "" {
		qs.Set("scope", args.Scope)
	}
	if args.Sort != "" {
		qs.Set("sort", args.Sort)
	}
	return c.doJSON(ctx, http.MethodGet, brainPath(brainID, "/search?"+qs.Encode()), nil)
}

// Recall implements [MemoryClient].
func (c *hostedClient) Recall(ctx context.Context, args RecallArgs) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	body := map[string]any{"query": args.Query}
	if args.Scope != "" {
		body["scope"] = args.Scope
	}
	if args.SessionID != "" {
		body["session_id"] = args.SessionID
	}
	if args.TopK > 0 {
		body["top_k"] = args.TopK
	}
	return c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/recall"), body)
}

// Ask implements [MemoryClient]. Hosted mode returns the final answer
// without streaming until the SDK supports incremental responses.
func (c *hostedClient) Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	body := map[string]any{"query": args.Query}
	if args.TopK > 0 {
		body["top_k"] = args.TopK
	}
	if progress != nil {
		progress(0, "requesting")
	}
	resp, err := c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/ask"), body)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(1, "answered")
	}
	return resp, nil
}

// IngestFile implements [MemoryClient].
func (c *hostedClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	abs := args.Path
	if !filepath.IsAbs(abs) {
		resolved, rerr := filepath.Abs(abs)
		if rerr != nil {
			return nil, fmt.Errorf("memory_ingest_file: resolve path: %w", rerr)
		}
		abs = resolved
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("memory_ingest_file: stat %s: %w", abs, err)
	}
	if info.Size() > fileIngestLimit {
		return nil, errors.New("file_too_large: 25 MiB limit exceeded")
	}
	raw, err := os.ReadFile(abs)
	if err != nil {
		return nil, fmt.Errorf("memory_ingest_file: read: %w", err)
	}
	if progress != nil {
		progress(0, "uploading")
	}
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	header := textproto.MIMEHeader{}
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, filepath.Base(abs)))
	header.Set("Content-Type", mimeByHint(abs, args.As))
	part, perr := writer.CreatePart(header)
	if perr != nil {
		return nil, fmt.Errorf("memory_ingest_file: multipart: %w", perr)
	}
	if _, werr := part.Write(raw); werr != nil {
		return nil, fmt.Errorf("memory_ingest_file: write part: %w", werr)
	}
	if cerr := writer.Close(); cerr != nil {
		return nil, fmt.Errorf("memory_ingest_file: close writer: %w", cerr)
	}
	req, rerr := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.Endpoint+brainPath(brainID, "/documents/ingest/file"), body)
	if rerr != nil {
		return nil, fmt.Errorf("memory_ingest_file: build: %w", rerr)
	}
	req.Header.Set("authorization", "Bearer "+c.cfg.Token)
	req.Header.Set("content-type", writer.FormDataContentType())
	req.Header.Set("accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("memory_ingest_file: post: %w", err)
	}
	defer resp.Body.Close()
	rawResp, _ := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("memory_ingest_file: HTTP %d: %s", resp.StatusCode, truncate(string(rawResp), 256))
	}
	var out map[string]any
	if err := json.Unmarshal(rawResp, &out); err != nil {
		return nil, fmt.Errorf("memory_ingest_file: decode: %w", err)
	}
	if progress != nil {
		progress(1, "ingested")
	}
	return out, nil
}

// IngestURL implements [MemoryClient].
func (c *hostedClient) IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "fetching")
	}
	resp, err := c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/documents/ingest/url"), map[string]any{"url": args.URL})
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(1, "ingested")
	}
	return resp, nil
}

// Extract implements [MemoryClient].
func (c *hostedClient) Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	if args.SessionID != "" {
		out := make([]any, 0, len(args.Messages))
		for i, msg := range args.Messages {
			isLast := i == len(args.Messages)-1
			meta := map[string]any{"skip_extract": !isLast}
			if args.ActorID != "" {
				meta["actor_id"] = args.ActorID
			}
			payload := map[string]any{
				"role":     msg.Role,
				"content":  msg.Content,
				"metadata": meta,
			}
			path := brainPath(brainID, "/sessions/"+url.PathEscape(args.SessionID)+"/messages")
			result, ierr := c.doJSON(ctx, http.MethodPost, path, payload)
			if ierr != nil {
				return nil, ierr
			}
			out = append(out, result)
			if progress != nil {
				progress(float64(i+1), fmt.Sprintf("%d/%d", i+1, len(args.Messages)))
			}
		}
		return map[string]any{"mode": "session", "messages": out}, nil
	}
	var transcript strings.Builder
	for _, m := range args.Messages {
		fmt.Fprintf(&transcript, "[%s] %s\n\n", m.Role, m.Content)
	}
	doc, err := c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/documents"), map[string]any{
		"title":   fmt.Sprintf("Transcript %s", time.Now().UTC().Format(time.RFC3339)),
		"content": transcript.String(),
		"source":  "extract",
	})
	if err != nil {
		return nil, err
	}
	return map[string]any{"mode": "transcript", "document": doc}, nil
}

// Reflect implements [MemoryClient].
func (c *hostedClient) Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	path := brainPath(brainID, "/sessions/"+url.PathEscape(args.SessionID)+"/close")
	return c.doJSON(ctx, http.MethodPost, path, map[string]any{})
}

// Consolidate implements [MemoryClient].
func (c *hostedClient) Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (map[string]any, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	return c.doJSON(ctx, http.MethodPost, brainPath(brainID, "/consolidate"), map[string]any{})
}

// CreateBrain implements [MemoryClient].
func (c *hostedClient) CreateBrain(ctx context.Context, args CreateBrainArgs) (map[string]any, error) {
	body := map[string]any{"name": args.Name}
	if args.Slug != "" {
		body["slug"] = args.Slug
	}
	if args.Visibility != "" {
		body["visibility"] = args.Visibility
	} else {
		body["visibility"] = "private"
	}
	return c.doJSON(ctx, http.MethodPost, "/v1/brains", body)
}

// ListBrains implements [MemoryClient].
func (c *hostedClient) ListBrains(ctx context.Context) (map[string]any, error) {
	return c.doJSON(ctx, http.MethodGet, "/v1/brains", nil)
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

// brainPath builds a /v1/brains/{id}/... URL suffix.
func brainPath(id, suffix string) string {
	return "/v1/brains/" + url.PathEscape(id) + suffix
}

// truncate clips s to n runes, appending nothing.
func truncate(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// firstNonEmpty returns the first non-empty input.
func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// stringOrDefault fetches key from m as a string, returning def on miss.
func stringOrDefault(m map[string]any, key, def string) string {
	if raw, ok := m[key]; ok {
		if s, ok := raw.(string); ok && s != "" {
			return s
		}
	}
	return def
}

// deriveTitle extracts a title heuristically from the markdown body.
func deriveTitle(body, fallback string) string {
	if fallback = strings.TrimSpace(fallback); fallback != "" {
		return fallback
	}
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
	}
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			if len(line) > 120 {
				line = line[:120]
			}
			return line
		}
	}
	return "Untitled memory"
}

// slugify mirrors the knowledge helper without importing it (unexported).
func slugify(title string) string {
	s := strings.ToLower(strings.TrimSpace(title))
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune('-')
		}
	}
	out := collapseHyphens(b.String())
	out = strings.Trim(out, "-")
	if len(out) > 60 {
		out = out[:60]
		out = strings.TrimRight(out, "-")
	}
	return out
}

// collapseHyphens reduces runs of hyphens to a single hyphen.
func collapseHyphens(s string) string {
	var b strings.Builder
	prev := false
	for _, r := range s {
		if r == '-' {
			if prev {
				continue
			}
			prev = true
			b.WriteRune(r)
			continue
		}
		prev = false
		b.WriteRune(r)
	}
	return b.String()
}

// sanitiseBrainID strips filesystem-hostile characters without pulling
// the unexported helper from package knowledge.
func sanitiseBrainID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return "_invalid"
	}
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '.' || r == '_' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := b.String()
	if strings.HasPrefix(out, ".") {
		out = "_" + strings.TrimPrefix(out, ".")
	}
	return out
}

// mapScopeToSearch converts the MCP-level scope to the search index
// column value. Returns the empty string for `all` so the scope filter
// matches every row.
func mapScopeToSearch(scope string) string {
	switch strings.ToLower(strings.TrimSpace(scope)) {
	case "global":
		return "global_memory"
	case "project":
		return "project_memory"
	case "agent":
		return "global_memory"
	}
	return ""
}

// mimeByHint picks a MIME type based on the caller-supplied hint or the
// file extension.
func mimeByHint(path, hint string) string {
	switch strings.ToLower(strings.TrimSpace(hint)) {
	case "markdown":
		return "text/markdown"
	case "text":
		return "text/plain"
	case "pdf":
		return "application/pdf"
	case "json":
		return "application/json"
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".markdown":
		return "text/markdown"
	case ".txt":
		return "text/plain"
	case ".pdf":
		return "application/pdf"
	case ".json":
		return "application/json"
	case ".html", ".htm":
		return "text/html"
	}
	return "application/octet-stream"
}

// hashString returns the hex SHA-256 of data.
func hashString(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

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
	"github.com/jeffs-brain/memory/go/cmd/memory-mcp/tools"
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

	Remember(ctx context.Context, args RememberArgs) (*tools.RememberResult, error)
	Search(ctx context.Context, args SearchArgs) (*tools.SearchResult, error)
	Recall(ctx context.Context, args RecallArgs) (*tools.RecallResult, error)
	Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (*tools.AskResult, error)
	IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (*tools.IngestResult, error)
	IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (*tools.IngestURLResult, error)
	Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (*tools.ExtractResult, error)
	ExtractAfterIngest(ctx context.Context, args ExtractAfterIngestArgs) (*tools.ExtractAfterIngestResult, error)
	Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (*tools.ReflectResult, error)
	Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (*tools.ConsolidateResult, error)
	CreateBrain(ctx context.Context, args CreateBrainArgs) (*tools.CreateBrainResult, error)
	ListBrains(ctx context.Context) (*tools.ListBrainsResult, error)

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

// ExtractAfterIngestArgs captures input for post-ingest extraction.
// Content should be provided directly to avoid re-fetching.
// When Content is empty, the DocumentSource label is used for logging.
type ExtractAfterIngestArgs struct {
	Content        string
	DocumentSource string
	Brain          string
	ActorID        string
	SessionID      string
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

// brainRoot returns the on-disk root for a brain. When the ID passes
// [brain.ValidateBrainID] it is used directly; otherwise the legacy
// sanitiser normalises it to a safe filesystem component.
func (c *localClient) brainRoot(id string) string {
	safe := id
	if brain.ValidateBrainID(id) != nil {
		safe = sanitiseBrainID(id)
	}
	return filepath.Join(c.cfg.BrainRoot, "brains", safe)
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
func (c *localClient) Remember(ctx context.Context, args RememberArgs) (*tools.RememberResult, error) {
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
	var tags []string
	if len(args.Tags) > 0 {
		tags = args.Tags
	}
	return &tools.RememberResult{
		ID:         resp.DocumentID,
		Path:       string(resp.Path),
		ByteSize:   resp.Bytes,
		ChunkCount: resp.ChunkCount,
		TookMs:     resp.TookMs,
		BrainID:    brainID,
		Tags:       tags,
	}, nil
}

// Search implements [MemoryClient].
func (c *localClient) Search(ctx context.Context, args SearchArgs) (*tools.SearchResult, error) {
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
	hits := make([]tools.SearchHit, 0, len(results))
	for _, r := range results {
		hits = append(hits, tools.SearchHit{
			Score:   r.Score,
			Path:    r.Path,
			Title:   r.Title,
			Summary: r.Summary,
			Content: r.Snippet,
			ChunkID: r.Path,
			Scope:   r.Scope,
		})
	}
	return &tools.SearchResult{
		Query:   args.Query,
		BrainID: brainID,
		Hits:    hits,
		TookMs:  time.Since(started).Milliseconds(),
	}, nil
}

// Recall implements [MemoryClient].
func (c *localClient) Recall(ctx context.Context, args RecallArgs) (*tools.RecallResult, error) {
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
			chunks := make([]tools.RecallChunk, 0, len(resp.Chunks))
			for _, hit := range resp.Chunks {
				chunks = append(chunks, tools.RecallChunk{
					ChunkID:    hit.ChunkID,
					DocumentID: hit.DocumentID,
					Score:      hit.Score,
					Path:       hit.Path,
					Content:    hit.Text,
					Title:      hit.Title,
					Summary:    hit.Summary,
				})
			}
			return &tools.RecallResult{
				Query:     args.Query,
				BrainID:   brainID,
				SessionID: args.SessionID,
				Chunks:    chunks,
			}, nil
		}
		c.log.Debug("memory_recall: retriever failed, falling back to search", "err", rerr)
	}
	// Fallback: mirror memory_search and convert to RecallResult.
	searchResult, serr := c.Search(ctx, SearchArgs{
		Query: args.Query,
		Brain: args.Brain,
		TopK:  topK,
		Scope: args.Scope,
	})
	if serr != nil {
		return nil, serr
	}
	chunks := make([]tools.RecallChunk, 0, len(searchResult.Hits))
	for _, hit := range searchResult.Hits {
		chunks = append(chunks, tools.RecallChunk{
			ChunkID: hit.ChunkID,
			Path:    hit.Path,
			Score:   hit.Score,
			Content: hit.Content,
			Title:   hit.Title,
			Summary: hit.Summary,
		})
	}
	return &tools.RecallResult{
		Query:     args.Query,
		BrainID:   searchResult.BrainID,
		SessionID: args.SessionID,
		Chunks:    chunks,
	}, nil
}

// Ask implements [MemoryClient]. Runs retrieval then a single blocking
// LLM completion. Progress notifications are coarse (retrieved +
// answered) until the SDK exposes a streaming `ask` surface.
func (c *localClient) Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (*tools.AskResult, error) {
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
	citations := make([]tools.AskCitation, 0, len(chunks))
	for i, hit := range chunks {
		if i >= 5 {
			break
		}
		citations = append(citations, tools.AskCitation{
			Type:        "citation",
			ChunkID:     hit.ChunkID,
			DocumentID:  hit.DocumentID,
			AnswerStart: 0,
			AnswerEnd:   0,
			Quote:       truncate(hit.Text, 200),
		})
	}
	retrieved := make([]tools.AskRetrievedChunk, 0, len(chunks))
	for _, hit := range chunks {
		retrieved = append(retrieved, tools.AskRetrievedChunk{
			ChunkID:    hit.ChunkID,
			DocumentID: hit.DocumentID,
			Score:      hit.Score,
			Preview:    truncate(hit.Text, 512),
		})
	}
	return &tools.AskResult{
		Answer:    resp.Text,
		Citations: citations,
		Retrieved: retrieved,
	}, nil
}

// IngestFile implements [MemoryClient].
func (c *localClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (*tools.IngestResult, error) {
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
	return &tools.IngestResult{
		Status:        "completed",
		DocumentID:    resp.DocumentID,
		Path:          string(resp.Path),
		Hash:          hashString([]byte(resp.DocumentID)),
		ChunkCount:    resp.ChunkCount,
		EmbeddedCount: resp.ChunkCount,
		DurationMs:    resp.TookMs,
		Reused:        false,
	}, nil
}

// IngestURL implements [MemoryClient].
func (c *localClient) IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (*tools.IngestURLResult, error) {
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

	// Read the stored content from the brain store so downstream
	// extraction can use it without re-fetching the URL.
	var storedContent string
	raw, readErr := br.store.Read(ctx, resp.Path)
	if readErr == nil {
		storedContent = string(raw)
	}

	return &tools.IngestURLResult{
		Path: "server",
		Result: tools.IngestResult{
			Status:        "completed",
			DocumentID:    resp.DocumentID,
			Path:          string(resp.Path),
			ChunkCount:    resp.ChunkCount,
			EmbeddedCount: resp.ChunkCount,
			DurationMs:    resp.TookMs,
			Reused:        false,
		},
		DocumentContent: storedContent,
	}, nil
}

// Extract implements [MemoryClient]. Local mode mirrors the TS wrapper's
// session and transcript shapes; we do not drive the extraction LLM
// here because the memory package already exposes [ExtractFromMessages]
// for the richer case. The Go SDK port for the bare transcript path
// records the conversation as a document so downstream consolidation
// can pick it up.
func (c *localClient) Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (*tools.ExtractResult, error) {
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
		out := make([]tools.ExtractedMessage, 0, len(extracted))
		now := time.Now().UTC().Format(time.RFC3339)
		for i, e := range extracted {
			out = append(out, tools.ExtractedMessage{
				ID:        fmt.Sprintf("%s:%d", args.SessionID, i),
				SessionID: args.SessionID,
				Role:      "assistant",
				Content:   e.Content,
				CreatedAt: now,
			})
		}
		if progress != nil {
			progress(1, fmt.Sprintf("%d memories", len(out)))
		}
		return &tools.ExtractResult{Mode: "session", Messages: out}, nil
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
	now := time.Now().UTC().Format(time.RFC3339)
	return &tools.ExtractResult{
		Mode: "transcript",
		Document: &tools.ExtractDocument{
			ID:             resp.DocumentID,
			BrainID:        brainID,
			Title:          req.Title,
			Path:           string(resp.Path),
			Source:         "extract",
			ContentType:    "text/markdown",
			ByteSize:       resp.Bytes,
			ChecksumSHA256: hashString([]byte(resp.DocumentID)),
			Metadata:       map[string]int{"message_count": len(args.Messages)},
			CreatedAt:      now,
			UpdatedAt:      now,
			DeletedAt:      nil,
		},
	}, nil
}

// ExtractAfterIngest implements [MemoryClient]. Uses the provided document
// content and runs the memory extractor to derive structured facts.
// Content must be supplied directly (read from brain store or file);
// no re-fetching of URLs occurs here.
// Extraction failure is non-fatal: returns empty result.
func (c *localClient) ExtractAfterIngest(ctx context.Context, args ExtractAfterIngestArgs) (*tools.ExtractAfterIngestResult, error) {
	empty := &tools.ExtractAfterIngestResult{Memories: []tools.ExtractedMemory{}}

	if c.provider == nil {
		return empty, nil
	}

	content := args.Content
	if strings.TrimSpace(content) == "" {
		return empty, nil
	}

	brainID := c.resolveBrainID(args.Brain)
	br, err := c.openBrain(ctx, brainID)
	if err != nil {
		c.log.Warn("extract-after-ingest: failed to open brain", "brain", brainID, "error", err)
		return empty, nil
	}

	// Truncate by rune count to avoid splitting multi-byte characters.
	runes := []rune(content)
	if len(runes) > 128_000 {
		runes = runes[:128_000]
		content = string(runes)
	}

	source := args.DocumentSource
	if source == "" {
		source = "unknown"
	}
	messages := []memory.Message{
		{
			Role: "user",
			Content: fmt.Sprintf(
				"The following document was ingested from %q. Extract any important facts, knowledge, or structured information from it:\n\n<ingested-document>\n%s\n</ingested-document>",
				source, content,
			),
		},
	}

	extracted, xerr := memory.ExtractFromMessages(ctx, c.provider, "", br.memory, "", messages)
	if xerr != nil {
		c.log.Warn("extract-after-ingest: extraction failed", "document", source, "error", xerr)
		return empty, nil
	}

	memories := make([]tools.ExtractedMemory, 0, len(extracted))
	for _, e := range extracted {
		memories = append(memories, tools.ExtractedMemory{
			Filename: e.Filename,
			Content:  e.Content,
		})
	}
	return &tools.ExtractAfterIngestResult{
		FactsExtracted: len(extracted),
		Memories:       memories,
	}, nil
}

// Reflect implements [MemoryClient]. Local mode runs
// [memory.Reflector.ForceReflect] so callers can exercise the surface
// end-to-end without a dedicated session backend.
func (c *localClient) Reflect(ctx context.Context, args ReflectArgs, progress ProgressEmitter) (*tools.ReflectResult, error) {
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
	now := time.Now().UTC().Format(time.RFC3339)
	if result == nil {
		return &tools.ReflectResult{
			ReflectionStatus:    "no_result",
			ReflectionAttempted: true,
			EndedAt:             now,
		}, nil
	}
	return &tools.ReflectResult{
		ReflectionStatus: "completed",
		Reflection: &tools.ReflectionDetail{
			Outcome:             result.Outcome,
			ShouldRecordEpisode: result.ShouldRecordEpisode,
			Path:                "",
			RetryFeedback:       result.RetryFeedback,
		},
		ReflectionAttempted: true,
		EndedAt:             now,
	}, nil
}

// Consolidate implements [MemoryClient]. Local mode falls back to
// [knowledge.Base.Compile] which re-chunks and re-indexes every
// persisted document in the brain.
func (c *localClient) Consolidate(ctx context.Context, args ConsolidateArgs, progress ProgressEmitter) (*tools.ConsolidateResult, error) {
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
	return &tools.ConsolidateResult{
		Result: tools.CompileDetail{
			Compiled:  result.Compiled,
			Chunks:    result.Chunks,
			Skipped:   result.Skipped,
			Errors:    result.Errors,
			ElapsedMs: result.Elapsed.Milliseconds(),
		},
	}, nil
}

// CreateBrain implements [MemoryClient]. Local mode writes a
// config.json alongside the brain root so [ListBrains] can surface
// structured metadata.
func (c *localClient) CreateBrain(ctx context.Context, args CreateBrainArgs) (*tools.CreateBrainResult, error) {
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
	createdAt := time.Now().UTC().Format(time.RFC3339)

	// brainConfig is the on-disk config shape.
	type brainConfig struct {
		Version    int    `json:"version"`
		Name       string `json:"name"`
		Slug       string `json:"slug"`
		Visibility string `json:"visibility"`
		CreatedAt  string `json:"createdAt"`
	}
	cfg := brainConfig{
		Version:    1,
		Name:       args.Name,
		Slug:       slug,
		Visibility: visibility,
		CreatedAt:  createdAt,
	}
	body, _ := json.MarshalIndent(cfg, "", "  ")
	if err := os.WriteFile(filepath.Join(root, "config.json"), append(body, '\n'), 0o644); err != nil {
		return nil, fmt.Errorf("memory_create_brain: write config: %w", err)
	}
	return &tools.CreateBrainResult{
		ID:         slug,
		Slug:       slug,
		Name:       args.Name,
		Visibility: visibility,
		CreatedAt:  createdAt,
	}, nil
}

// brainDiskConfig is the JSON shape persisted in each brain's config.json.
type brainDiskConfig struct {
	Slug       string `json:"slug"`
	Name       string `json:"name"`
	Visibility string `json:"visibility"`
	CreatedAt  string `json:"createdAt"`
}

// ListBrains implements [MemoryClient].
func (c *localClient) ListBrains(ctx context.Context) (*tools.ListBrainsResult, error) {
	dir := filepath.Join(c.cfg.BrainRoot, "brains")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &tools.ListBrainsResult{Items: []tools.BrainInfo{}}, nil
		}
		return nil, fmt.Errorf("memory_list_brains: %w", err)
	}
	items := make([]tools.BrainInfo, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		cfgPath := filepath.Join(dir, e.Name(), "config.json")
		var parsed brainDiskConfig
		if raw, rerr := os.ReadFile(cfgPath); rerr == nil {
			_ = json.Unmarshal(raw, &parsed)
		}
		slug := parsed.Slug
		if slug == "" {
			slug = e.Name()
		}
		name := parsed.Name
		if name == "" {
			name = e.Name()
		}
		visibility := parsed.Visibility
		if visibility == "" {
			visibility = "private"
		}
		items = append(items, tools.BrainInfo{
			ID:         slug,
			Slug:       slug,
			Name:       name,
			Visibility: visibility,
			CreatedAt:  parsed.CreatedAt,
		})
	}
	return &tools.ListBrainsResult{Items: items}, nil
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

// doJSONRaw performs a JSON request and returns the raw response bytes.
func (c *hostedClient) doJSONRaw(ctx context.Context, method, path string, body any) ([]byte, error) {
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
	return raw, nil
}

// doJSONInto performs a JSON request and decodes the response into dst.
func (c *hostedClient) doJSONInto(ctx context.Context, method, path string, body, dst any) error {
	raw, err := c.doJSONRaw(ctx, method, path, body)
	if err != nil {
		return err
	}
	if len(raw) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("memory-mcp: decoding %s %s: %w", method, path, err)
	}
	return nil
}

// Remember implements [MemoryClient].
func (c *hostedClient) Remember(ctx context.Context, args RememberArgs) (*tools.RememberResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	type tagMeta struct {
		Tags string `json:"tags"`
	}
	type rememberBody struct {
		Title    string   `json:"title"`
		Content  string   `json:"content"`
		Path     string   `json:"path,omitempty"`
		Metadata *tagMeta `json:"metadata,omitempty"`
	}
	req := rememberBody{Title: args.Title, Content: args.Content, Path: args.Path}
	if len(args.Tags) > 0 {
		req.Metadata = &tagMeta{Tags: strings.Join(args.Tags, ",")}
	}
	var result tools.RememberResult
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/documents"), req, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Search implements [MemoryClient].
func (c *hostedClient) Search(ctx context.Context, args SearchArgs) (*tools.SearchResult, error) {
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
	var result tools.SearchResult
	if err := c.doJSONInto(ctx, http.MethodGet, brainPath(brainID, "/search?"+qs.Encode()), nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Recall implements [MemoryClient].
func (c *hostedClient) Recall(ctx context.Context, args RecallArgs) (*tools.RecallResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	type recallBody struct {
		Query     string `json:"query"`
		Scope     string `json:"scope,omitempty"`
		SessionID string `json:"session_id,omitempty"`
		TopK      int    `json:"top_k,omitempty"`
	}
	body := recallBody{
		Query:     args.Query,
		Scope:     args.Scope,
		SessionID: args.SessionID,
		TopK:      args.TopK,
	}
	var result tools.RecallResult
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/recall"), body, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Ask implements [MemoryClient]. Hosted mode returns the final answer
// without streaming until the SDK supports incremental responses.
func (c *hostedClient) Ask(ctx context.Context, args AskArgs, progress ProgressEmitter) (*tools.AskResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	type askBody struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k,omitempty"`
	}
	body := askBody{Query: args.Query, TopK: args.TopK}
	if progress != nil {
		progress(0, "requesting")
	}
	var result tools.AskResult
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/ask"), body, &result); err != nil {
		return nil, err
	}
	if progress != nil {
		progress(1, "answered")
	}
	return &result, nil
}

// IngestFile implements [MemoryClient].
func (c *hostedClient) IngestFile(ctx context.Context, args IngestFileArgs, progress ProgressEmitter) (*tools.IngestResult, error) {
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
	var result tools.IngestResult
	if err := json.Unmarshal(rawResp, &result); err != nil {
		return nil, fmt.Errorf("memory_ingest_file: decode: %w", err)
	}
	if progress != nil {
		progress(1, "ingested")
	}
	return &result, nil
}

// IngestURL implements [MemoryClient].
func (c *hostedClient) IngestURL(ctx context.Context, args IngestURLArgs, progress ProgressEmitter) (*tools.IngestURLResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	if progress != nil {
		progress(0, "fetching")
	}
	type ingestURLBody struct {
		URL string `json:"url"`
	}
	var result tools.IngestURLResult
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/documents/ingest/url"), ingestURLBody{URL: args.URL}, &result); err != nil {
		return nil, err
	}
	if progress != nil {
		progress(1, "ingested")
	}
	return &result, nil
}

// Extract implements [MemoryClient].
func (c *hostedClient) Extract(ctx context.Context, args ExtractArgs, progress ProgressEmitter) (*tools.ExtractResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	if args.SessionID != "" {
		out := make([]tools.ExtractedMessage, 0, len(args.Messages))
		for i, msg := range args.Messages {
			isLast := i == len(args.Messages)-1
			type msgMeta struct {
				SkipExtract bool   `json:"skip_extract"`
				ActorID     string `json:"actor_id,omitempty"`
			}
			type msgPayload struct {
				Role     string  `json:"role"`
				Content  string  `json:"content"`
				Metadata msgMeta `json:"metadata"`
			}
			payload := msgPayload{
				Role:    msg.Role,
				Content: msg.Content,
				Metadata: msgMeta{
					SkipExtract: !isLast,
					ActorID:     args.ActorID,
				},
			}
			path := brainPath(brainID, "/sessions/"+url.PathEscape(args.SessionID)+"/messages")
			var extracted tools.ExtractedMessage
			if ierr := c.doJSONInto(ctx, http.MethodPost, path, payload, &extracted); ierr != nil {
				return nil, ierr
			}
			out = append(out, extracted)
			if progress != nil {
				progress(float64(i+1), fmt.Sprintf("%d/%d", i+1, len(args.Messages)))
			}
		}
		return &tools.ExtractResult{Mode: "session", Messages: out}, nil
	}
	var transcript strings.Builder
	for _, m := range args.Messages {
		fmt.Fprintf(&transcript, "[%s] %s\n\n", m.Role, m.Content)
	}
	type transcriptBody struct {
		Title   string `json:"title"`
		Content string `json:"content"`
		Source  string `json:"source"`
	}
	body := transcriptBody{
		Title:   fmt.Sprintf("Transcript %s", time.Now().UTC().Format(time.RFC3339)),
		Content: transcript.String(),
		Source:  "extract",
	}
	var doc tools.ExtractDocument
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/documents"), body, &doc); err != nil {
		return nil, err
	}
	return &tools.ExtractResult{Mode: "transcript", Document: &doc}, nil
}

// ExtractAfterIngest implements [MemoryClient] for hosted mode.
func (c *hostedClient) ExtractAfterIngest(_ context.Context, _ ExtractAfterIngestArgs) (*tools.ExtractAfterIngestResult, error) {
	// Hosted mode does not support extract-after-ingest yet.
	// Return empty result -- non-fatal.
	return &tools.ExtractAfterIngestResult{Memories: []tools.ExtractedMemory{}}, nil
}

// Reflect implements [MemoryClient].
func (c *hostedClient) Reflect(ctx context.Context, args ReflectArgs, _ ProgressEmitter) (*tools.ReflectResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	path := brainPath(brainID, "/sessions/"+url.PathEscape(args.SessionID)+"/close")
	var result tools.ReflectResult
	if err := c.doJSONInto(ctx, http.MethodPost, path, struct{}{}, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Consolidate implements [MemoryClient].
func (c *hostedClient) Consolidate(ctx context.Context, args ConsolidateArgs, _ ProgressEmitter) (*tools.ConsolidateResult, error) {
	brainID, err := c.resolveBrainID(args.Brain)
	if err != nil {
		return nil, err
	}
	var result tools.ConsolidateResult
	if err := c.doJSONInto(ctx, http.MethodPost, brainPath(brainID, "/consolidate"), struct{}{}, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// CreateBrain implements [MemoryClient].
func (c *hostedClient) CreateBrain(ctx context.Context, args CreateBrainArgs) (*tools.CreateBrainResult, error) {
	type createBody struct {
		Name       string `json:"name"`
		Slug       string `json:"slug,omitempty"`
		Visibility string `json:"visibility"`
	}
	visibility := args.Visibility
	if visibility == "" {
		visibility = "private"
	}
	body := createBody{Name: args.Name, Slug: args.Slug, Visibility: visibility}
	var result tools.CreateBrainResult
	if err := c.doJSONInto(ctx, http.MethodPost, "/v1/brains", body, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ListBrains implements [MemoryClient].
func (c *hostedClient) ListBrains(ctx context.Context) (*tools.ListBrainsResult, error) {
	var result tools.ListBrainsResult
	if err := c.doJSONInto(ctx, http.MethodGet, "/v1/brains", nil, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

// brainPath builds a /v1/brains/{id}/... URL suffix.
func brainPath(id, suffix string) string {
	return "/v1/brains/" + url.PathEscape(id) + suffix
}

// truncate clips s to n runes, appending nothing.
// Uses []rune to avoid splitting multi-byte characters.
func truncate(s string, n int) string {
	if n <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n])
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

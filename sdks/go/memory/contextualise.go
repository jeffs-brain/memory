// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// contextualPrefixMaxTokens caps the LLM output for a contextual
// prefix.
const contextualPrefixMaxTokens = 120

// contextualPrefixTemperature keeps the output deterministic.
const contextualPrefixTemperature = 0.0

// contextualPrefixMarker is the literal delimiter prepended before the
// extracted fact body.
const contextualPrefixMarker = "Context: "

// contextualPrefixSystemPrompt is the system instruction for the
// situating-prefix LLM call.
const contextualPrefixSystemPrompt = `You situate extracted memory facts inside their parent session so downstream retrieval carries the surrounding context.

Output ONE short paragraph, 50 to 100 tokens, British English. No em dashes. No lists, no headings, no preamble. Do not repeat the fact verbatim. Do not speculate. State only what the session header and the fact body already support.

Cover in order:
1. when the session happened (date / weekday) if known,
2. the broader topic or theme of the session,
3. how this specific fact sits within that session.

Start directly with the sentence. Do not prefix with "Context:" or any label.`

// ContextualiserConfig configures a Contextualiser instance.
type ContextualiserConfig struct {
	// Provider is the LLM provider used for situating-prefix calls.
	Provider llm.Provider

	// Model overrides the model used for the call. Empty string means
	// the provider's default.
	Model string

	// CacheDir, when non-empty, is a directory where per-fact cache
	// entries are written as JSON files.
	CacheDir string

	// Concurrency caps the in-flight LLM calls. Zero selects 12.
	Concurrency int

	// MaxTokens overrides the output cap on the situating-prefix call.
	MaxTokens int
}

// Contextualiser builds short situating prefixes for extracted memory
// facts.
type Contextualiser struct {
	cfg ContextualiserConfig
	sem chan struct{}

	mu    sync.Mutex
	cache map[string]string
}

// NewContextualiser builds a Contextualiser. Returns nil when the
// config has no provider so callers can treat a disabled contextualiser
// as a plain no-op.
func NewContextualiser(cfg ContextualiserConfig) *Contextualiser {
	if cfg.Provider == nil {
		return nil
	}
	conc := cfg.Concurrency
	if conc <= 0 {
		conc = 12
	}
	if conc > 32 {
		conc = 32
	}
	return &Contextualiser{
		cfg:   cfg,
		sem:   make(chan struct{}, conc),
		cache: make(map[string]string),
	}
}

// Enabled reports whether the contextualiser will issue real LLM calls.
func (c *Contextualiser) Enabled() bool {
	return c != nil && c.cfg.Provider != nil
}

// ModelName returns the model the contextualiser uses, or the empty
// string when disabled.
func (c *Contextualiser) ModelName() string {
	if !c.Enabled() {
		return ""
	}
	return c.cfg.Model
}

// BuildPrefix returns a situating prefix for fact, keyed by (session
// id, fact content, model).
func (c *Contextualiser) BuildPrefix(ctx context.Context, sessionID, sessionSummary, factBody string) string {
	if !c.Enabled() || factBody == "" {
		return ""
	}

	key := c.cacheKey(sessionID, factBody)

	c.mu.Lock()
	if v, ok := c.cache[key]; ok {
		c.mu.Unlock()
		return v
	}
	c.mu.Unlock()

	if cached, ok := c.readCache(key); ok {
		c.mu.Lock()
		c.cache[key] = cached
		c.mu.Unlock()
		return cached
	}

	select {
	case c.sem <- struct{}{}:
	case <-ctx.Done():
		return ""
	}
	defer func() { <-c.sem }()

	prefix, err := c.callProvider(ctx, sessionSummary, factBody)
	if err != nil {
		slog.Warn("memory: contextual prefix build failed", "err", err, "session", sessionID)
		return ""
	}

	c.mu.Lock()
	c.cache[key] = prefix
	c.mu.Unlock()
	if err := c.writeCache(key, prefix); err != nil {
		slog.Debug("memory: contextual prefix cache write failed", "err", err)
	}
	return prefix
}

// ApplyContextualPrefix prepends the contextual prefix to body with the
// canonical delimiter, or returns body unchanged when prefix is empty.
func ApplyContextualPrefix(prefix, body string) string {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return body
	}
	return contextualPrefixMarker + prefix + "\n\n" + body
}

// callProvider fires the actual LLM call.
func (c *Contextualiser) callProvider(ctx context.Context, sessionSummary, factBody string) (string, error) {
	model := c.ModelName()
	body := factBody
	if len(body) > 4096 {
		body = body[:4096]
	}
	summary := strings.TrimSpace(sessionSummary)
	if summary == "" {
		summary = "(no session header supplied)"
	}

	var b strings.Builder
	b.WriteString("Session header:\n")
	b.WriteString(summary)
	b.WriteString("\n\nFact body:\n")
	b.WriteString(body)
	b.WriteString("\n")

	maxTokens := contextualPrefixMaxTokens
	if c.cfg.MaxTokens > 0 {
		maxTokens = c.cfg.MaxTokens
	}
	resp, err := c.cfg.Provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: contextualPrefixSystemPrompt},
			{Role: RoleUser, Content: b.String()},
		},
		MaxTokens:   maxTokens,
		Temperature: contextualPrefixTemperature,
	})
	if err != nil {
		return "", err
	}

	prefix := strings.TrimSpace(resp.Text)
	prefix = strings.ReplaceAll(prefix, "\r\n", "\n")
	if low := strings.ToLower(prefix); strings.HasPrefix(low, "context:") {
		prefix = strings.TrimSpace(prefix[len("context:"):])
	}
	prefix = strings.Join(strings.Fields(prefix), " ")
	return prefix, nil
}

// cacheKey computes a deterministic fingerprint.
func (c *Contextualiser) cacheKey(sessionID, factBody string) string {
	h := sha256.New()
	fmt.Fprintf(h, "v1\n")
	fmt.Fprintf(h, "model=%s\n", c.ModelName())
	fmt.Fprintf(h, "session=%s\n", sessionID)
	fmt.Fprintf(h, "body=%s", factBody)
	return hex.EncodeToString(h.Sum(nil))
}

type cachedPrefix struct {
	Prefix  string    `json:"prefix"`
	Model   string    `json:"model"`
	Written time.Time `json:"written"`
}

func (c *Contextualiser) readCache(key string) (string, bool) {
	if c.cfg.CacheDir == "" {
		return "", false
	}
	raw, err := os.ReadFile(c.cacheFilename(key))
	if err != nil {
		return "", false
	}
	var rec cachedPrefix
	if err := json.Unmarshal(raw, &rec); err != nil {
		return "", false
	}
	return rec.Prefix, true
}

func (c *Contextualiser) writeCache(key, prefix string) error {
	if c.cfg.CacheDir == "" {
		return nil
	}
	path := c.cacheFilename(key)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	rec := cachedPrefix{
		Prefix:  prefix,
		Model:   c.ModelName(),
		Written: time.Now().UTC(),
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func (c *Contextualiser) cacheFilename(key string) string {
	return filepath.Join(c.cfg.CacheDir, key[:2], key+".json")
}

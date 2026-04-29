// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/query"
	"github.com/jeffs-brain/memory/go/retrieval"
)

var errBodyTooLarge = errors.New("request body too large")

// readLimitedBody reads up to limit bytes from r.Body, returning
// [errBodyTooLarge] when exceeded. The body is always drained.
func readLimitedBody(r *http.Request, limit int64) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	defer func() { _ = r.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, errBodyTooLarge
	}
	return body, nil
}

// decodeJSONBody decodes the request body into target with a size cap.
func decodeJSONBody(r *http.Request, target any, limit int64) error {
	body, err := readLimitedBody(r, limit)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			return fmt.Errorf("body exceeds %d bytes", limit)
		}
		return err
	}
	if len(body) == 0 {
		return fmt.Errorf("empty body")
	}
	if err := json.Unmarshal(body, target); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

// resolveBrain pulls brainId from the path and looks it up via
// BrainManager. Writes the appropriate Problem+JSON and returns nil
// when the brain is missing.
func (d *Daemon) resolveBrain(w http.ResponseWriter, r *http.Request) *BrainResources {
	id := r.PathValue("brainId")
	if id == "" {
		httpd.ValidationError(w, "missing brainId")
		return nil
	}
	br, err := d.Brains.Get(r.Context(), id)
	if err != nil {
		if errors.Is(err, ErrBrainNotFound) {
			httpd.NotFound(w, "brain not found: "+id)
			return nil
		}
		httpd.InternalError(w, err.Error())
		return nil
	}
	return br
}

func augmentRetrievalQuery(question, questionDate string) string {
	expansion := query.ExpandTemporal(question, questionDate)
	if !expansion.Resolved || len(expansion.DateHints) == 0 {
		return strings.TrimSpace(question)
	}
	tokens := make([]string, 0, len(expansion.DateHints)*2)
	for _, hint := range expansion.DateHints {
		trimmed := strings.TrimSpace(hint)
		if trimmed == "" {
			continue
		}
		tokens = append(tokens, `"`+trimmed+`"`, `"`+strings.ReplaceAll(trimmed, "/", "-")+`"`)
	}
	seen := make(map[string]bool, len(tokens))
	unique := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if token == "" || seen[token] {
			continue
		}
		seen[token] = true
		unique = append(unique, token)
	}
	if len(unique) == 0 {
		return strings.TrimSpace(question)
	}
	return strings.TrimSpace(question + " " + strings.Join(unique, " "))
}

func hydrateFallbackChunk(
	ctx context.Context,
	store brain.Store,
	chunk retrieval.RetrievedChunk,
	sessionDate string,
) retrieval.RetrievedChunk {
	text := chunk.Text
	metadata := cloneChunkMetadata(chunk.Metadata)
	if sessionDate != "" {
		if metadata == nil {
			metadata = make(map[string]any)
		}
		metadata["session_date"] = sessionDate
	}
	if strings.TrimSpace(text) != "" && fallbackMetadataString(metadata, "expansion") == "episodic_recall" {
		chunk.Metadata = metadata
		return chunk
	}
	if store != nil && chunk.Path != "" {
		if data, err := store.Read(ctx, brain.Path(chunk.Path)); err == nil {
			raw := strings.TrimSpace(string(data))
			if raw != "" {
				if _, body := memory.ParseFrontmatter(raw); strings.TrimSpace(body) != "" {
					text = body
				} else {
					text = raw
				}
				metadata = enrichFallbackMetadata(metadata, raw)
			}
		}
	}
	chunk.Text = text
	if len(metadata) == 0 {
		chunk.Metadata = nil
	} else {
		chunk.Metadata = metadata
	}
	return chunk
}

func fallbackMetadataString(metadata map[string]any, key string) string {
	if metadata == nil {
		return ""
	}
	value, ok := metadata[key]
	if !ok {
		return ""
	}
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func cloneChunkMetadata(src map[string]any) map[string]any {
	if len(src) == 0 {
		return nil
	}
	out := make(map[string]any, len(src))
	for key, value := range src {
		out[key] = value
	}
	return out
}

func enrichFallbackMetadata(metadata map[string]any, raw string) map[string]any {
	set := func(key, value string) {
		if value == "" {
			return
		}
		if metadata == nil {
			metadata = make(map[string]any)
		}
		metadata[key] = value
	}
	set("session_id", frontmatterValue(raw, "session_id", "sessionid"))
	set("session_date", frontmatterValue(raw, "session_date", "sessiondate"))
	set("observed_on", frontmatterValue(raw, "observed_on", "observedon"))
	set("modified", frontmatterValue(raw, "modified"))
	return metadata
}

func frontmatterValue(raw string, keys ...string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	lines := strings.Split(raw, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return ""
	}
	wanted := make(map[string]bool, len(keys))
	for _, key := range keys {
		normalised := strings.ToLower(strings.TrimSpace(key))
		if normalised != "" {
			wanted[normalised] = true
		}
	}
	for _, line := range lines[1:] {
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			break
		}
		parts := strings.SplitN(trimmed, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(parts[0]))
		if !wanted[key] {
			continue
		}
		return strings.TrimSpace(parts[1])
	}
	return ""
}

// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
)

// rememberRequest captures the wire shape for POST .../remember.
type rememberRequest struct {
	Note   string   `json:"note"`
	Tags   []string `json:"tags,omitempty"`
	Source string   `json:"source,omitempty"`
	Slug   string   `json:"slug,omitempty"`
	Scope  string   `json:"scope,omitempty"` // "global" (default) or "project:<slug>"
}

// recallRequest captures the wire shape for POST .../recall.
type recallRequest struct {
	Query   string `json:"query"`
	TopK    int    `json:"topK,omitempty"`
	Project string `json:"project,omitempty"`
}

// extractRequest captures the wire shape for POST .../extract.
//
// Messages may be supplied directly; if missing the handler returns
// validation_error.
type extractRequest struct {
	Project  string             `json:"project,omitempty"`
	Model    string             `json:"model,omitempty"`
	Messages []extractMsgWire   `json:"messages"`
}

type extractMsgWire struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// reflectRequest mirrors extractRequest.
type reflectRequest struct {
	Project  string             `json:"project,omitempty"`
	Model    string             `json:"model,omitempty"`
	Messages []extractMsgWire   `json:"messages"`
}

// consolidateRequest is the body for POST .../consolidate.
type consolidateRequest struct {
	Mode  string `json:"mode,omitempty"` // "full" (default) or "quick"
	Model string `json:"model,omitempty"`
}

// handleRemember writes a free-form note into the brain's memory
// tree. Without an explicit slug a deterministic timestamp is used so
// the route is idempotent within a second.
func (d *Daemon) handleRemember(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req rememberRequest
	if err := decodeJSONBody(r, &req, 256*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if strings.TrimSpace(req.Note) == "" {
		httpd.ValidationError(w, "note required")
		return
	}
	slug := req.Slug
	if slug == "" {
		slug = "note-" + time.Now().UTC().Format("20060102t150405z")
	}
	var p brain.Path
	scope := strings.TrimSpace(req.Scope)
	switch {
	case scope == "" || scope == "global":
		p = brain.MemoryGlobalTopic(slug)
	case strings.HasPrefix(scope, "project:"):
		project := strings.TrimPrefix(scope, "project:")
		if project == "" {
			httpd.ValidationError(w, "project slug required for project scope")
			return
		}
		p = brain.MemoryProjectTopic(project, slug)
	default:
		httpd.ValidationError(w, "unknown scope: "+scope)
		return
	}
	body := buildRememberBody(req)
	if err := br.Store.Write(r.Context(), p, []byte(body)); err != nil {
		if !httpd.WriteStoreError(w, err) {
			httpd.InternalError(w, err.Error())
		}
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{
		"path": string(p),
		"slug": slug,
	})
}

// handleRecall delegates to memory.Memory.Recall via the daemon's
// shared LLM provider. Returns a flat list of {path, content,
// linkedFrom?} entries for the client.
func (d *Daemon) handleRecall(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req recallRequest
	if err := decodeJSONBody(r, &req, 64*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if req.Query == "" {
		httpd.ValidationError(w, "query required")
		return
	}
	memories, err := br.Memory.Recall(r.Context(), d.LLM, "", req.Project, req.Query, nil, memory.RecallWeights{Global: 1, Project: 1})
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	out := make([]map[string]any, 0, len(memories))
	for _, m := range memories {
		out = append(out, map[string]any{
			"path":       string(m.Path),
			"content":    m.Content,
			"linkedFrom": m.LinkedFrom,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"memories": out})
}

// handleExtract runs a synchronous extraction pass over the supplied
// messages and returns the structured result without applying it to
// the store. The client decides whether to persist.
func (d *Daemon) handleExtract(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req extractRequest
	if err := decodeJSONBody(r, &req, 1<<20); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		httpd.ValidationError(w, "messages required")
		return
	}
	memMsgs := wireMessagesToMemory(req.Messages)
	results, err := memory.ExtractFromMessages(r.Context(), d.LLM, req.Model, br.Memory, req.Project, memMsgs)
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"memories": results})
}

// handleReflect runs the reflection pipeline against the supplied
// messages, returning the structured result. Heuristics are applied
// in-place via the reflector to keep parity with the TS surface.
func (d *Daemon) handleReflect(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req reflectRequest
	if err := decodeJSONBody(r, &req, 1<<20); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		httpd.ValidationError(w, "messages required")
		return
	}
	reflector := memory.NewReflector(br.Memory)
	memMsgs := wireMessagesToMemory(req.Messages)
	result := reflector.ForceReflect(r.Context(), d.LLM, req.Model, req.Project, memMsgs)
	if result == nil {
		writeJSON(w, http.StatusOK, map[string]any{"result": nil})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"result": result})
}

// handleConsolidate runs the consolidation pipeline.
func (d *Daemon) handleConsolidate(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req consolidateRequest
	if err := decodeJSONBody(r, &req, 64*1024); err != nil {
		// allow empty body
		if !strings.Contains(err.Error(), "empty") {
			httpd.ValidationError(w, err.Error())
			return
		}
	}
	cons := memory.NewConsolidator(d.LLM, req.Model, br.Memory)
	var report *memory.ConsolidationReport
	var err error
	if strings.EqualFold(req.Mode, "quick") {
		report, err = cons.RunQuick(r.Context())
	} else {
		report, err = cons.RunFull(r.Context())
	}
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, report)
}

// wireMessagesToMemory converts the wire shape into memory.Message.
func wireMessagesToMemory(msgs []extractMsgWire) []memory.Message {
	out := make([]memory.Message, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, memory.Message{
			Role:    llm.Role(strings.ToLower(m.Role)),
			Content: m.Content,
		})
	}
	return out
}

// buildRememberBody renders the persisted markdown for a remember
// note, including frontmatter that distinguishes it from extraction
// output.
func buildRememberBody(req rememberRequest) string {
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString("name: \"remembered\"\n")
	if req.Source != "" {
		b.WriteString("source: ")
		b.WriteString(quoteYAML(req.Source))
		b.WriteString("\n")
	}
	now := time.Now().UTC().Format(time.RFC3339)
	b.WriteString("created: ")
	b.WriteString(now)
	b.WriteString("\n")
	b.WriteString("modified: ")
	b.WriteString(now)
	b.WriteString("\n")
	if len(req.Tags) > 0 {
		b.WriteString("tags:\n")
		for _, t := range req.Tags {
			fmt.Fprintf(&b, "  - %s\n", t)
		}
	}
	b.WriteString("---\n\n")
	b.WriteString(strings.TrimSpace(req.Note))
	b.WriteString("\n")
	return b.String()
}

// quoteYAML wraps value in double quotes and escapes embedded quotes.
func quoteYAML(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	return `"` + value + `"`
}

// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/jeffs-brain/memory/go/eval/lme"
	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/knowledge"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/query"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

// askReaderModeBasic is the default; preserves the original /ask prompt
// shape and generation params.
// askReaderModeAugmented opts in to the LME-style CoT reader prompt so
// /ask matches the eval harness on recency, enumeration, and temporal
// guidance.
const (
	askReaderModeBasic     = "basic"
	askReaderModeAugmented = "augmented"
)

type askRequest struct {
	Question     string `json:"question"`
	TopK         int    `json:"topK,omitempty"`
	Mode         string `json:"mode,omitempty"`
	Model        string `json:"model,omitempty"`
	ReaderMode   string `json:"readerMode,omitempty"`
	QuestionDate string `json:"questionDate,omitempty"`
}

// handleAsk runs retrieval, then streams an LLM completion as SSE
// events: retrieve -> answer_delta* -> citation* -> done. Errors
// after the headers are sent emit an error event and close the
// stream cleanly.
func (d *Daemon) handleAsk(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	var req askRequest
	if err := decodeJSONBody(r, &req, 256*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if req.Question == "" {
		httpd.ValidationError(w, "question required")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	// Run retrieval before opening SSE so failures still produce a
	// Problem+JSON response. Once the stream is live every error
	// must travel as an `error` event.
	chunks := d.retrieveForAsk(r, br, req)

	stream := httpd.SSEStart(w)
	if stream == nil {
		return
	}

	// Retrieve frame first so the client sees evidence before any
	// answer tokens arrive.
	_ = stream.SendJSON("retrieve", map[string]any{
		"chunks": chunks,
		"topK":   req.TopK,
		"mode":   req.Mode,
	})

	complete := buildAskCompleteRequest(req, chunks)

	chOut, err := d.LLM.CompleteStream(r.Context(), complete)
	if err != nil {
		_ = stream.SendJSON("error", map[string]string{"message": err.Error()})
		_ = stream.SendJSON("done", map[string]any{"ok": false})
		return
	}

	for chunk := range chOut {
		select {
		case <-r.Context().Done():
			return
		default:
		}
		if chunk.DeltaText != "" {
			if err := stream.SendJSON("answer_delta", map[string]string{"text": chunk.DeltaText}); err != nil {
				return
			}
		}
		if chunk.Stop != "" {
			break
		}
	}

	for _, c := range chunks {
		_ = stream.SendJSON("citation", map[string]any{
			"chunkId": c.ChunkID,
			"path":    c.Path,
			"title":   c.Title,
			"score":   c.Score,
		})
	}

	_ = stream.SendJSON("done", map[string]any{"ok": true})
}

// retrieveForAsk returns retrieved chunks directly so the SSE loop can
// stream citations without re-querying. Freshness: awaits the one-shot
// initial scan, then relies on the Subscribe mechanism for runtime
// writes. No per-ask full re-scan, which would serialise every
// concurrent /ask on the refresh write lock under load.
func (d *Daemon) retrieveForAsk(r *http.Request, br *BrainResources, req askRequest) []retrieval.RetrievedChunk {
	br.WaitReady(r.Context())
	var chunks []retrieval.RetrievedChunk
	if br.Retriever != nil {
		mode := retrieval.Mode(req.Mode)
		if mode == "" {
			mode = retrieval.ModeAuto
		}
		resp, err := br.Retriever.Retrieve(r.Context(), retrieval.Request{
			Query:   req.Question,
			TopK:    req.TopK,
			Mode:    mode,
			BrainID: br.ID,
		})
		if err == nil {
			chunks = resp.Chunks
		}
	}
	if len(chunks) == 0 && br.Search != nil {
		results, err := br.Search.Search(req.Question, search.SearchOpts{MaxResults: req.TopK})
		if err == nil {
			for _, h := range results {
				chunks = append(chunks, retrieval.RetrievedChunk{
					ChunkID:    h.Path,
					DocumentID: h.Path,
					Path:       h.Path,
					Score:      1.0 / float64(h.Score+1),
					Text:       h.Snippet,
					Title:      h.Title,
					Summary:    h.Summary,
				})
			}
		}
	}
	if len(chunks) == 0 && br.Knowledge != nil {
		resp, err := br.Knowledge.Search(r.Context(), knowledge.SearchRequest{Query: req.Question, MaxResults: req.TopK})
		if err == nil {
			for _, h := range resp.Hits {
				chunks = append(chunks, retrieval.RetrievedChunk{
					ChunkID:    string(h.Path),
					DocumentID: string(h.Path),
					Path:       string(h.Path),
					Score:      h.Score,
					Text:       h.Snippet,
					Title:      h.Title,
					Summary:    h.Summary,
				})
			}
		}
	}
	if len(chunks) > req.TopK {
		chunks = chunks[:req.TopK]
	}
	return chunks
}

// askSystemPrompt is kept short so the full prompt budget belongs to
// retrieved context.
const askSystemPrompt = "You are Jeffs Brain, a helpful assistant. When evidence is supplied, ground your answer in it and cite the path of any source you rely on. When no evidence is supplied, answer concisely from general knowledge."

func buildAskPrompt(question string, chunks []retrieval.RetrievedChunk) string {
	var b strings.Builder
	if len(chunks) > 0 {
		b.WriteString("## Evidence\n\n")
		for _, c := range chunks {
			b.WriteString("### ")
			if c.Title != "" {
				b.WriteString(c.Title)
			} else {
				b.WriteString(c.Path)
			}
			b.WriteString(" (")
			b.WriteString(c.Path)
			b.WriteString(")\n")
			body := c.Text
			if body == "" {
				body = c.Summary
			}
			b.WriteString(body)
			b.WriteString("\n\n")
		}
	}
	b.WriteString("## Question\n\n")
	b.WriteString(question)
	return b.String()
}

// buildAskCompleteRequest dispatches between the basic /ask prompt and
// the augmented LME CoT reader prompt. Basic mode is byte-identical to
// the original behaviour; augmented mode wraps the same retrieval output
// in the reader template with recency, enumeration, and temporal
// guidance, applies temporal expansion when a questionDate is supplied,
// and pins generation params to the LME reader defaults.
func buildAskCompleteRequest(req askRequest, chunks []retrieval.RetrievedChunk) llm.CompleteRequest {
	mode := strings.ToLower(strings.TrimSpace(req.ReaderMode))
	if mode == askReaderModeAugmented {
		content := buildAugmentedAskContent(req.Question, chunks)
		// Match the in-process runner: prefix the resolved date hints
		// onto the content (not the question) so the reader sees them
		// alongside the evidence it must reason over.
		if req.QuestionDate != "" && content != "" {
			expansion := query.ExpandTemporal(req.Question, req.QuestionDate)
			if expansion.Resolved && len(expansion.DateHints) > 0 {
				content = fmt.Sprintf("[Resolved temporal references: %s]\n\n%s",
					strings.Join(expansion.DateHints, ", "), content)
			}
		}
		prompt := lme.BuildReaderPrompt(req.Question, req.QuestionDate, content)
		return llm.CompleteRequest{
			Model: req.Model,
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: prompt},
			},
			Temperature: lme.ReaderTemperature,
			MaxTokens:   lme.ReaderMaxTokens,
		}
	}

	prompt := buildAskPrompt(req.Question, chunks)
	return llm.CompleteRequest{
		Model: req.Model,
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: askSystemPrompt},
			{Role: llm.RoleUser, Content: prompt},
		},
		Temperature: 0.2,
		MaxTokens:   1024,
	}
}

// buildAugmentedAskContent renders retrieved chunks in the shape the LME
// reader consumes. When chunks carry session_id frontmatter we pipe them
// through the lme session-block preprocessor so the reader sees
// chronologically sorted, assistant-filtered turns. Otherwise we fall
// back to the "### title (path)\n{body}" framing.
func buildAugmentedAskContent(question string, chunks []retrieval.RetrievedChunk) string {
	if len(chunks) == 0 {
		return ""
	}
	var raw strings.Builder
	sessionLike := false
	for i, c := range chunks {
		if i > 0 {
			raw.WriteString("\n\n")
		}
		body := c.Text
		if body == "" {
			body = c.Summary
		}
		if strings.Contains(body, "session_id:") {
			sessionLike = true
		}
		raw.WriteString(body)
	}
	if sessionLike {
		return lme.ProcessSessionContextForQuestion(raw.String(), question)
	}
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("### ")
		if c.Title != "" {
			b.WriteString(c.Title)
		} else {
			b.WriteString(c.Path)
		}
		b.WriteString(" (")
		b.WriteString(c.Path)
		b.WriteString(")\n")
		body := c.Text
		if body == "" {
			body = c.Summary
		}
		b.WriteString(body)
		b.WriteString("\n\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

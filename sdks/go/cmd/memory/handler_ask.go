// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"strings"

	"github.com/jeffs-brain/memory/go/eval/lme"
	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/knowledge"
	"github.com/jeffs-brain/memory/go/llm"
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
	CandidateK   int    `json:"candidateK,omitempty"`
	RerankTopN   int    `json:"rerankTopN,omitempty"`
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
	readerMode, ok := normaliseAskReaderMode(req.ReaderMode)
	if !ok {
		httpd.ValidationError(w, "readerMode must be 'basic' or 'augmented'")
		return
	}
	req.ReaderMode = readerMode
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
		mode := normaliseRetrievalMode(req.Mode)
		resp, err := br.Retriever.Retrieve(r.Context(), retrieval.Request{
			Query:        req.Question,
			QuestionDate: req.QuestionDate,
			TopK:         req.TopK,
			CandidateK:   req.CandidateK,
			RerankTopN:   req.RerankTopN,
			Mode:         mode,
			BrainID:      br.ID,
		})
		if err == nil {
			chunks = resp.Chunks
		}
	}
	if len(chunks) == 0 && br.Search != nil {
		results, err := br.Search.Search(augmentRetrievalQuery(req.Question, req.QuestionDate), search.SearchOpts{MaxResults: req.TopK})
		if err == nil {
			for _, h := range results {
				chunk := retrieval.RetrievedChunk{
					ChunkID:    h.Path,
					DocumentID: h.Path,
					Path:       h.Path,
					Score:      1.0 / float64(h.Score+1),
					Text:       h.Snippet,
					Title:      h.Title,
					Summary:    h.Summary,
				}
				chunks = append(chunks, hydrateFallbackChunk(r.Context(), br.Store, chunk, h.SessionDate))
			}
		}
	}
	if len(chunks) == 0 && br.Knowledge != nil {
		resp, err := br.Knowledge.Search(r.Context(), knowledge.SearchRequest{Query: req.Question, MaxResults: req.TopK})
		if err == nil {
			for _, h := range resp.Hits {
				chunk := retrieval.RetrievedChunk{
					ChunkID:    string(h.Path),
					DocumentID: string(h.Path),
					Path:       string(h.Path),
					Score:      h.Score,
					Text:       h.Snippet,
					Title:      h.Title,
					Summary:    h.Summary,
				}
				chunks = append(chunks, hydrateFallbackChunk(r.Context(), br.Store, chunk, ""))
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
	mode, _ := normaliseAskReaderMode(req.ReaderMode)
	if mode == askReaderModeAugmented {
		content := buildAugmentedAskContent(req.Question, req.QuestionDate, chunks)
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

func normaliseAskReaderMode(raw string) (string, bool) {
	switch mode := strings.ToLower(strings.TrimSpace(raw)); mode {
	case "", askReaderModeBasic:
		return askReaderModeBasic, true
	case askReaderModeAugmented:
		return askReaderModeAugmented, true
	default:
		return "", false
	}
}

// buildAugmentedAskContent renders retrieved chunks in the same
// numbered/date-tagged shape used by the retrieve-only benchmark path.
func buildAugmentedAskContent(question, questionDate string, chunks []retrieval.RetrievedChunk) string {
	if len(chunks) == 0 {
		return ""
	}
	passages := make([]lme.RetrievedPassage, 0, len(chunks))
	for _, chunk := range chunks {
		body := chunk.Text
		if body == "" {
			body = chunk.Summary
		}
		passages = append(passages, lme.RetrievedPassage{
			Path:      chunk.Path,
			Score:     chunk.Score,
			Body:      body,
			Date:      metadataStringValue(chunk.Metadata, "session_date", "sessionDate", "observed_on", "observedOn", "modified"),
			SessionID: metadataStringValue(chunk.Metadata, "session_id", "sessionId"),
		})
	}
	return lme.RenderRetrievedPassages(passages, question, questionDate)
}

func metadataStringValue(meta map[string]any, keys ...string) string {
	for _, key := range keys {
		value, ok := meta[key]
		if !ok {
			continue
		}
		text, ok := value.(string)
		if !ok {
			continue
		}
		text = strings.TrimSpace(text)
		if text != "" {
			return text
		}
	}
	return ""
}

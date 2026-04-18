// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/retrieval"
)

// AgenticOpts bounds a single agentic-mode question run. MaxIterations
// caps the tool-call ladder; QuestionTimeout caps wall-clock time per
// question so a runaway loop cannot stall the benchmark.
type AgenticOpts struct {
	MaxIterations   int
	QuestionTimeout time.Duration
}

const (
	defaultAgenticMaxIterations = 8
	defaultAgenticQuestionTO    = 90 * time.Second
)

// AgentResources is the minimal per-question surface the agentic loop
// needs. When Retriever is non-nil it powers the kb_search tool;
// otherwise kb_search falls back to a naive store.List scan that still
// returns something usable for tests without a full retrieval stack.
type AgentResources struct {
	Store     brain.Store
	Memory    *memory.Memory
	Retriever retrieval.Retriever
	// ProjectPath anchors memory recall / retrieval filters to the LME
	// project slug so the agent only searches the eval brain's project
	// memory, not arbitrary global state.
	ProjectPath string
}

// AgentFactory constructs a fresh [AgentResources] per question so
// per-question state (retrieval cache, memory cursor) does not bleed
// across questions in a concurrent run.
type AgentFactory func(context.Context, brain.Store) (*AgentResources, error)

// defaultAgentFactory wires the minimal resource set: just the store
// and memory facade, no retriever. The agent falls back to the naive
// keyword scan which is enough to drive tests and small corpora. CLI
// callers that wire a full daemon pass their own factory.
func defaultAgentFactory(_ context.Context, store brain.Store) (*AgentResources, error) {
	return &AgentResources{
		Store:       store,
		Memory:      memory.New(store),
		ProjectPath: "/eval/lme",
	}, nil
}

// RunQuestionsAgentic drives each question through a tool-enabled
// agent loop. The loop offers the model two tools — kb_search (hybrid
// retrieval against the eval brain) and memory_recall (surface
// extracted memory topics) — and lets the model choose when to call
// them before producing a final answer. Respects the parent context
// and opts.QuestionTimeout. Costs accumulate via [EstimateUSD] under
// the agent bucket; readerModel is used for pricing.
func RunQuestionsAgentic(
	parentCtx context.Context,
	factory AgentFactory,
	store brain.Store,
	provider llm.Provider,
	actorModel string,
	questions []Question,
	opts AgenticOpts,
	costs *CostAccumulator,
	workers int,
) []QuestionOutcome {
	maxIter := opts.MaxIterations
	if maxIter <= 0 {
		maxIter = defaultAgenticMaxIterations
	}
	timeout := opts.QuestionTimeout
	if timeout <= 0 {
		timeout = defaultAgenticQuestionTO
	}

	total := len(questions)
	var completed atomic.Int64
	startTime := time.Now()
	fmt.Fprintf(os.Stderr, "[agentic] starting %d questions, %d workers, %s timeout each, model=%s\n",
		total, workers, timeout, actorModel)

	outcomes := runQuestionsConcurrent(parentCtx, questions, workers, func(ctx context.Context, _ int, q Question) QuestionOutcome {
		outcome := processQuestionAgentic(ctx, factory, store, provider, actorModel, q, maxIter, timeout, costs)
		done := completed.Add(1)
		status := "ok"
		if outcome.Error != "" {
			status = "err"
		}
		toolsUsed := strings.Join(outcome.ToolCalls, ",")
		if toolsUsed == "" {
			toolsUsed = "-"
		}
		elapsed := time.Since(startTime).Seconds()
		rate := float64(done) / elapsed
		var eta time.Duration
		if rate > 0 {
			eta = time.Duration(float64(total-int(done))/rate) * time.Second
		}
		fmt.Fprintf(os.Stderr,
			"[agentic] %d/%d %s (%s, %dms, tools=%s, ans=%dchars) eta=%s\n",
			done, total, q.ID, status, outcome.LatencyMs, toolsUsed, len(outcome.AgentAnswer),
			eta.Truncate(time.Second))
		return outcome
	})

	fmt.Fprintf(os.Stderr, "[agentic] done: %d outcomes in %s\n",
		len(outcomes), time.Since(startTime).Truncate(time.Second))
	return outcomes
}

// processQuestionAgentic runs a single question through the agent
// loop. A fresh [AgentResources] is built per question so per-question
// state never leaks between workers.
func processQuestionAgentic(
	parentCtx context.Context,
	factory AgentFactory,
	store brain.Store,
	provider llm.Provider,
	actorModel string,
	q Question,
	maxIter int,
	timeout time.Duration,
	costs *CostAccumulator,
) QuestionOutcome {
	qStart := time.Now()

	outcome := QuestionOutcome{
		ID:           q.ID,
		Category:     q.Category,
		Question:     q.Question,
		QuestionDate: q.QuestionDate,
		GroundTruth:  q.Answer,
	}

	ctx, cancel := context.WithTimeout(parentCtx, timeout)
	defer cancel()

	res, err := factory(ctx, store)
	if err != nil {
		outcome.Error = fmt.Sprintf("agent factory: %v", err)
		outcome.LatencyMs = int(time.Since(qStart).Milliseconds())
		return outcome
	}
	if res.Store == nil {
		res.Store = store
	}
	if res.Memory == nil {
		res.Memory = memory.New(res.Store)
	}
	if res.ProjectPath == "" {
		res.ProjectPath = "/eval/lme"
	}

	answer, toolCalls, usage, runErr := runAgentLoop(ctx, res, provider, actorModel, q, maxIter)
	if costs != nil {
		costs.AddAgent(EstimateUSD(actorModel, usage))
	}
	outcome.InputTokens = usage.InputTokens
	outcome.OutputTokens = usage.OutputTokens
	outcome.ToolCalls = toolCalls
	outcome.AgentAnswer = strings.TrimSpace(answer)
	outcome.LatencyMs = int(time.Since(qStart).Milliseconds())
	if runErr != nil {
		outcome.Error = runErr.Error()
	}
	return outcome
}

// agentSystemPrompt anchors the reader in its role: it is a memory-
// aware assistant with tools that expose the eval brain's extracted
// memories. The prompt steers the model to call tools before answering
// so we see the full retrieval path exercised.
const agentSystemPrompt = `You are an assistant that answers user questions using an external memory tool set.

Tools available:
- kb_search: Hybrid BM25 + vector search across the eval brain. Returns relevant extracted memory chunks.
- memory_recall: Surface memory topic files matching a keyword. Good for exact phrase recall.

Process:
1. Read the question carefully.
2. Call kb_search with a focused query derived from the question. Prefer calling at least once before answering.
3. Optionally call memory_recall to surface adjacent memory files when kb_search hits look thin.
4. Read the returned chunks and produce a concise, factual answer grounded strictly in the retrieved content.
5. If no memory matches, answer "I don't know" rather than guessing.

Temporal reasoning:
- Each retrieved chunk may carry a session_date. When the same topic appears with different values on different dates, prefer the most recent.
- Treat supersession phrases ("now", "currently", "actually", "no longer", "I updated") as hard overrides.

Output:
- Final answer only. Do not repeat the question.
- Do not describe the tools you used. Just give the answer.`

// agentMaxTokens caps output per turn; tool-calling turns rarely need
// more than a few hundred tokens.
const agentMaxTokens = 800
const agentTemperature = 0.0

// runAgentLoop drives the tool-calling loop for a single question.
// Returns the final answer, the ordered list of tool names invoked,
// the aggregated usage, and any terminal error.
func runAgentLoop(
	ctx context.Context,
	res *AgentResources,
	provider llm.Provider,
	model string,
	q Question,
	maxIter int,
) (string, []string, Usage, error) {
	system := agentSystemPrompt
	if q.QuestionDate != "" {
		system += fmt.Sprintf("\n\nCurrent date: %s.", q.QuestionDate)
	}

	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: system},
		{Role: llm.RoleUser, Content: q.Question},
	}

	tools := []llm.ToolDef{
		{
			Name:        "kb_search",
			Description: "Hybrid search across the eval brain's extracted memories. Returns up to top_k ranked chunks.",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Focused search query derived from the user question.",
					},
					"top_k": map[string]any{
						"type":        "integer",
						"description": "Maximum number of chunks to return. Defaults to 8.",
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Name:        "memory_recall",
			Description: "Surface memory topic files whose body matches a keyword.",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"keyword": map[string]any{
						"type":        "string",
						"description": "Keyword or short phrase to match against memory topic bodies.",
					},
				},
				"required": []string{"keyword"},
			},
		},
	}

	var (
		toolCalls []string
		usage     Usage
	)

	for turn := 0; turn < maxIter; turn++ {
		if err := ctx.Err(); err != nil {
			if err == context.DeadlineExceeded {
				return lastAssistantText(messages), toolCalls, usage, fmt.Errorf("question timeout")
			}
			return lastAssistantText(messages), toolCalls, usage, err
		}

		resp, err := provider.Complete(ctx, llm.CompleteRequest{
			Model:       model,
			Messages:    messages,
			Tools:       tools,
			MaxTokens:   agentMaxTokens,
			Temperature: agentTemperature,
		})
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded {
				return lastAssistantText(messages), toolCalls, usage, fmt.Errorf("question timeout")
			}
			return lastAssistantText(messages), toolCalls, usage, fmt.Errorf("complete turn %d: %w", turn, err)
		}
		usage.InputTokens += resp.TokensIn
		usage.OutputTokens += resp.TokensOut

		// No tool calls: treat the text as the final answer.
		if len(resp.ToolCalls) == 0 {
			return resp.Text, toolCalls, usage, nil
		}

		// Record the assistant turn then append each tool call result
		// back onto the message history so the next Complete call sees
		// the loop state. The SDK's llm.Message does not carry tool
		// blocks explicitly; we stringify the tool_use and tool_result
		// as text so every provider backend can consume them. The
		// Anthropic adapter flattens RoleTool → user which is fine
		// because the system prompt anchors the conversation shape.
		if resp.Text != "" {
			messages = append(messages, llm.Message{Role: llm.RoleAssistant, Content: resp.Text})
		}
		for _, tc := range resp.ToolCalls {
			toolCalls = append(toolCalls, tc.Name)
			messages = append(messages, llm.Message{
				Role:    llm.RoleAssistant,
				Content: fmt.Sprintf("[tool_use name=%s id=%s input=%s]", tc.Name, tc.ID, string(tc.Arguments)),
			})
			result := executeAgentTool(ctx, res, tc)
			messages = append(messages, llm.Message{
				Role:    llm.RoleTool,
				Content: fmt.Sprintf("[tool_result id=%s]\n%s", tc.ID, result),
			})
		}
	}

	// Exhausted the iteration budget: synthesise a final answer from
	// the last assistant text so we still have something to score.
	return lastAssistantText(messages), toolCalls, usage, fmt.Errorf("max iterations (%d) reached", maxIter)
}

// lastAssistantText walks messages in reverse for the last assistant
// turn's content. Used when the loop terminates without a clean text
// response.
func lastAssistantText(messages []llm.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == llm.RoleAssistant {
			content := messages[i].Content
			if strings.HasPrefix(content, "[tool_use") {
				continue
			}
			return content
		}
	}
	return ""
}

// executeAgentTool dispatches a tool call against the agent resources.
// Every unknown tool returns a short error string so the model can
// self-correct; every tool output is capped in size to keep context
// cost bounded.
func executeAgentTool(ctx context.Context, res *AgentResources, tc llm.ToolCall) string {
	switch tc.Name {
	case "kb_search":
		return toolKBSearch(ctx, res, tc.Arguments)
	case "memory_recall":
		return toolMemoryRecall(ctx, res, tc.Arguments)
	default:
		return fmt.Sprintf("error: unknown tool %q", tc.Name)
	}
}

// kbSearchArgs is the parsed input for the kb_search tool.
type kbSearchArgs struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

// toolKBSearch runs a hybrid retrieval call when a retriever is wired
// and falls back to a naive store walk when it is not. The fallback
// keeps the agent functional in tests and minimal setups; production
// runs will always carry a retriever.
func toolKBSearch(ctx context.Context, res *AgentResources, raw json.RawMessage) string {
	var args kbSearchArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return fmt.Sprintf("error: invalid arguments: %v", err)
	}
	if args.Query == "" {
		return "error: query is required"
	}
	topK := args.TopK
	if topK <= 0 {
		topK = 8
	}

	if res.Retriever != nil {
		resp, err := res.Retriever.Retrieve(ctx, retrieval.Request{
			Query:   args.Query,
			TopK:    topK,
			Mode:    retrieval.ModeAuto,
			Filters: retrieval.Filters{PathPrefix: string(brain.MemoryProjectPrefix(memory.ProjectSlug(res.ProjectPath)))},
		})
		if err != nil {
			return fmt.Sprintf("error: retrieve: %v", err)
		}
		return formatChunks(resp.Chunks)
	}

	return naiveStoreSearch(ctx, res, args.Query, topK)
}

// memoryRecallArgs is the parsed input for the memory_recall tool.
type memoryRecallArgs struct {
	Keyword string `json:"keyword"`
}

// toolMemoryRecall performs a shallow keyword scan over the project
// memory topics. Simpler than kb_search but useful for exact-phrase
// recall the hybrid pipeline can miss on tiny corpora.
func toolMemoryRecall(ctx context.Context, res *AgentResources, raw json.RawMessage) string {
	var args memoryRecallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return fmt.Sprintf("error: invalid arguments: %v", err)
	}
	if args.Keyword == "" {
		return "error: keyword is required"
	}

	return naiveStoreSearch(ctx, res, args.Keyword, 8)
}

// naiveStoreSearch walks the project memory prefix looking for files
// whose body contains the query (case-insensitive). Used as the
// fallback for kb_search and as the full implementation of
// memory_recall.
func naiveStoreSearch(ctx context.Context, res *AgentResources, query string, topK int) string {
	slug := memory.ProjectSlug(res.ProjectPath)
	prefix := brain.MemoryProjectPrefix(slug)
	files, err := res.Store.List(ctx, prefix, brain.ListOpts{Recursive: true, IncludeGenerated: true})
	if err != nil {
		return fmt.Sprintf("error: list store: %v", err)
	}

	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })

	needle := strings.ToLower(query)
	type hit struct {
		path    brain.Path
		excerpt string
	}
	var hits []hit
	for _, f := range files {
		if f.IsDir {
			continue
		}
		if strings.HasSuffix(string(f.Path), "MEMORY.md") {
			continue
		}
		data, readErr := res.Store.Read(ctx, f.Path)
		if readErr != nil {
			continue
		}
		body := string(data)
		if !strings.Contains(strings.ToLower(body), needle) {
			continue
		}
		hits = append(hits, hit{path: f.Path, excerpt: excerpt(body, query, 280)})
		if len(hits) >= topK {
			break
		}
	}

	if len(hits) == 0 {
		// Last-resort fallback: scan the bulk-ingest raw/lme prefix
		// so the agent still has something when the replay path was
		// skipped. This keeps pure-retrieval runs usable.
		rawFiles, rawErr := res.Store.List(ctx, brain.Path("raw/lme"), brain.ListOpts{Recursive: true, IncludeGenerated: true})
		if rawErr == nil {
			sort.Slice(rawFiles, func(i, j int) bool { return rawFiles[i].Path < rawFiles[j].Path })
			for _, f := range rawFiles {
				if f.IsDir {
					continue
				}
				data, readErr := res.Store.Read(ctx, f.Path)
				if readErr != nil {
					continue
				}
				body := string(data)
				if !strings.Contains(strings.ToLower(body), needle) {
					continue
				}
				hits = append(hits, hit{path: f.Path, excerpt: excerpt(body, query, 280)})
				if len(hits) >= topK {
					break
				}
			}
		}
	}

	if len(hits) == 0 {
		return "no matches found"
	}

	var b strings.Builder
	for i, h := range hits {
		fmt.Fprintf(&b, "#%d %s\n%s\n\n", i+1, h.path, h.excerpt)
	}
	return strings.TrimSpace(b.String())
}

// excerpt returns a centred snippet around the first match of needle
// in body. Falls back to the first n runes when the needle is absent.
func excerpt(body, needle string, n int) string {
	lowerBody := strings.ToLower(body)
	lowerNeedle := strings.ToLower(needle)
	idx := strings.Index(lowerBody, lowerNeedle)
	if idx < 0 {
		if len(body) > n {
			return body[:n]
		}
		return body
	}
	start := idx - n/2
	if start < 0 {
		start = 0
	}
	end := start + n
	if end > len(body) {
		end = len(body)
	}
	snippet := body[start:end]
	if start > 0 {
		snippet = "..." + snippet
	}
	if end < len(body) {
		snippet = snippet + "..."
	}
	return snippet
}

// formatChunks renders retrieval hits into the compact text the
// agent's Complete call consumes on the next turn.
func formatChunks(chunks []retrieval.RetrievedChunk) string {
	if len(chunks) == 0 {
		return "no matches found"
	}
	var b strings.Builder
	for i, c := range chunks {
		header := c.Path
		if header == "" {
			header = c.ChunkID
		}
		fmt.Fprintf(&b, "#%d %s (score=%.3f)\n%s\n\n", i+1, header, c.Score, strings.TrimSpace(c.Text))
	}
	return strings.TrimSpace(b.String())
}

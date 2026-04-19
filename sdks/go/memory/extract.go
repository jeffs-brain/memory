// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/query"
)

// Extraction thresholds.
const (
	extractMaxTokens             = 4096
	extractTemperature           = 0.2
	extractMinMessages           = 6
	extractMaxRecent             = 40
	heuristicUserFactLimit       = 2
	heuristicMilestoneFactLimit  = 2
	heuristicPreferenceFactLimit = 2
	heuristicPendingFactLimit    = 3
)

var (
	heuristicDateTagRe                  = regexp.MustCompile(`\b\d{4}[-/]\d{2}[-/]\d{2}\b`)
	heuristicMonthNameDateRe            = regexp.MustCompile(`(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b`)
	heuristicQuantityRe                 = regexp.MustCompile(`\b\d{1,6}(?:\.\d+)?\b`)
	heuristicUnitQuantityRe             = regexp.MustCompile(`(?i)\b(\d{1,6}(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?|km|kilometres?|miles?|metres?|meters?|kg|kilograms?|pounds?|lbs?|grams?|percent|%)\b`)
	heuristicWordUnitQuantityRe         = regexp.MustCompile(`(?i)\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?)\b`)
	heuristicDurationFactRe             = regexp.MustCompile(`(?i)\b(?:\d{1,4}-day|[a-z]+-day|[a-z]+-week|[a-z]+-month|[a-z]+-year|week-long|month-long|year-long)\b`)
	heuristicOrdinalRe                  = regexp.MustCompile(`(?i)\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b`)
	heuristicSessionDateRe              = regexp.MustCompile(`\b\d{4}[/-]\d{2}[/-]\d{2}(?:\s+\([A-Za-z]{3}\))?(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b`)
	firstPersonFactRe                   = regexp.MustCompile(`(?i)\b(i|i'm|i’ve|i've|my|we|we're|we’ve|we've|our)\b`)
	heuristicWordRe                     = regexp.MustCompile(`[A-Za-z][A-Za-z-]{2,}`)
	heuristicMilestoneEventRe           = regexp.MustCompile(`(?i)\b(?:(?:just|recently)\s+)?(?:completed|submitted|graduated|finished|started|joined|accepted|presented)\b`)
	heuristicMilestoneTimeRe            = regexp.MustCompile(`(?i)\b(?:today|yesterday|recently|just|last\s+(?:week|month|year|summer|spring|fall|autumn|winter))\b`)
	heuristicMilestoneTopicRe           = regexp.MustCompile(`(?i)\b(?:degree|thesis|dissertation|paper|research|conference|course|class|project|internship|job|role|group|club|community|network|forum|association|society|linkedin)\b`)
	heuristicPreferenceBesidesLikeRe    = regexp.MustCompile(`(?i)\bbesides\s+([^.!?]+?),\s*i\s+(?:also\s+)?like\s+([^.!?]+?)(?:[.!?]|$)`)
	heuristicPreferenceLikeRe           = regexp.MustCompile(`(?i)\bi\s+(?:also\s+)?(?:like|love|prefer|enjoy)\s+([^.!?]+?)(?:[.!?]|$)`)
	heuristicPreferenceCompatibleRe     = regexp.MustCompile(`(?i)\bcompatible with (?:my|the)\s+([^.!?,\n]+)`)
	heuristicPreferenceDesignedForRe    = regexp.MustCompile(`(?i)\bspecifically designed for\s+([^.!?,\n]+)`)
	heuristicPreferenceAsUserRe         = regexp.MustCompile(`(?i)\bas a[n]?\s+([^.!?,\n]+?)\s+user\b`)
	heuristicPreferenceFieldRe          = regexp.MustCompile(`(?i)\bfield of\s+([^.!?,\n]+)`)
	heuristicPreferenceAdvancedRe       = regexp.MustCompile(`(?i)\badvanced topics in\s+([^.!?,\n]+)`)
	heuristicPreferenceSkipBasicsRe     = regexp.MustCompile(`(?i)\bskip the basics\b`)
	heuristicPreferenceWorkingInFieldRe = regexp.MustCompile(`(?i)\b(?:i am|i'm)\s+working in the field\b`)
	heuristicPendingActionLeadRe        = regexp.MustCompile(`(?i)^\s*(?:i(?:'ve)?(?:\s+still)?\s+(?:need|have)\s+to|i(?:'ve)?\s+got\s+to|i\s+must|i\s+should|i\s+need\s+to\s+remember\s+to|remember\s+to|don't\s+let\s+me\s+forget\s+to)\s+([^.!?]+)`)
	heuristicPendingActionStartRe       = regexp.MustCompile(`(?i)^(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|call|email|pay|renew|cancel|buy|send|post|fix|follow\s+up)\b`)
	heuristicRecommendationRequestRe    = regexp.MustCompile(`(?i)\b(?:recommend|suggest|looking for|look for|what should i|which should i|where should i stay|what to watch|what to read|what to serve)\b`)
	heuristicRecommendationUnderRe      = regexp.MustCompile(`(?i)\bunder\s+(\d{1,4}(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|pages?|£|€|\$))\b`)
	heuristicRecommendationNotTooRe     = regexp.MustCompile(`(?i)\b(?:nothing|not)\s+too\s+([^,.!?;\n]+)`)
	heuristicRecommendationWithoutRe    = regexp.MustCompile(`(?i)\bwithout\s+([^,.!?;\n]+)`)
	heuristicRecommendationFamilyRe     = regexp.MustCompile(`(?i)\b(?:family-friendly|kid-friendly)\b`)
	heuristicRecommendationLightRe      = regexp.MustCompile(`(?i)\b(?:light-hearted|feel-good|cosy|cozy)\b`)
	heuristicAppointmentRe              = regexp.MustCompile(`(?i)\b(?:appointment|check-?up|consultation|follow-?up|therapy session|scan|surgery|dentist|doctor|gp|dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|psychiatrist|psychologist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b`)
	heuristicMedicalEntityRe            = regexp.MustCompile(`(?i)\b(?:gp|doctor|dentist|dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|psychiatrist|psychologist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b`)
	heuristicEventRe                    = regexp.MustCompile(`(?i)\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar|service|mass|worship|prayer)\b`)
	heuristicEventAttendanceRe          = regexp.MustCompile(`(?i)\b(?:attend(?:ed|ing)?|went to|go(?:ing)? to|joined|join(?:ing)?|participat(?:ed|ing)|volunteer(?:ed|ing)|present(?:ed|ing)|watch(?:ed|ing)|listen(?:ed|ing)\s+to|got back from|completed)\b`)
	heuristicEventTitleRe               = regexp.MustCompile(`\b([A-Z][A-Za-z0-9&'/-]*(?:\s+[A-Z][A-Za-z0-9&'/-]*){0,6}\s+(?:Workshop|Conference|Concert|Festival|Meetup|Show|Screening|Class|Course|Webinar|Lecture|Seminar))\b`)
	heuristicWithPersonRe               = regexp.MustCompile(`\bwith\s+(Dr\.?\s+[A-Z][a-zA-Z'-]+)\b`)
	heuristicRelativeDateRe             = regexp.MustCompile(`(?i)\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b`)
	heuristicClockTimeRe                = regexp.MustCompile(`(?i)\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b`)
	heuristicReligiousServiceRe         = regexp.MustCompile(`(?i)\battend(?:ed|ing)?\s+([^,.!?]+?\s+service(?:\s+at\s+[^,.!?]+)?)\b`)
)

var heuristicStopwords = map[string]bool{
	"been":       true,
	"city":       true,
	"definitely": true,
	"feels":      true,
	"following":  true,
	"getting":    true,
	"have":       true,
	"just":       true,
	"last":       true,
	"lately":     true,
	"miles":      true,
	"months":     true,
	"really":     true,
	"routine":    true,
	"sticking":   true,
	"their":      true,
	"weeks":      true,
}

// Extractor manages background memory extraction. It runs after each
// turn to distil durable knowledge from the conversation into memory
// files.
type Extractor struct {
	mem *Memory

	mu         sync.Mutex
	lastCursor int
	inProgress bool

	ctx *Contextualiser
}

// NewExtractor creates a new Extractor bound to the supplied Memory.
func NewExtractor(mem *Memory) *Extractor {
	return &Extractor{mem: mem}
}

// SetContextualiser wires an optional [Contextualiser] into the
// extractor.
func (e *Extractor) SetContextualiser(c *Contextualiser) {
	if e == nil {
		return
	}
	e.ctx = c
}

// extractionPrompt is the system prompt for the extraction agent. Ported
// verbatim from jeff; prompt content is tuned.
const extractionPrompt = `You are a memory extraction agent. Analyse the recent conversation messages below and determine what durable knowledge should be saved to the persistent memory system.

You MUST respond with ONLY a JSON object. Do NOT call tools, do NOT write prose. Just output the JSON.

Both speakers contribute durable knowledge. Treat user turns and assistant turns as equally valid sources of facts. Capture everything the user stated AND everything the assistant provided: recommendations (restaurants, hotels, shops, books), specific named suggestions, recipes, itineraries, enumerated lists or rankings the assistant gave, answers the assistant produced, corrections the assistant issued, plans the assistant proposed, colours or attributes the assistant described, and any quantities or dates the assistant cited. If the assistant enumerated items (a list of jobs, options, steps, or candidates), save the full enumeration verbatim including positions where relevant. When in doubt, extract both sides.

Memory types:
- user: User's role, preferences, knowledge level, working style
- feedback: Corrections or confirmations about approach (what to avoid or keep doing)
- project: Non-obvious context about ongoing work, goals, decisions, deadlines (includes assistant recommendations and enumerations worth recalling later)
- reference: Pointers to external systems, URLs, project names, named entities the assistant surfaced (restaurants, hotels, businesses, books, product names)

Memory scopes:
- global (~/.config/jeff/memory/): Cross-project knowledge. Types: user, feedback
- project (project memory directory): Project-specific knowledge. Types: project, reference

When deciding scope:
- user preferences, working style, general corrections → global
- project architecture, project-specific decisions, external system pointers, assistant recommendations and enumerations → project
- default to "project" if unsure

Examples of assistant-turn facts that MUST be captured:
- "I recommend Roscioli for romantic Italian in Rome." → create a reference memory naming the restaurant, cuisine, city.
- "Here are seven work-from-home jobs for seniors: 1. Virtual Assistant, 2. ..., 7. Transcriptionist." → save the full numbered list so later recall can reconstruct any position.
- "The Plesiosaur in the children's book had a blue scaly body." → save the attribute with its subject.

Updates and quantitative facts that MUST be captured:
- When the user gives a new count, total, amount, ratio, progress update, milestone, or outcome, save it even if an older memory on the same topic already exists.
- Prefer an update with supersedes when the new statement revises prior state.
- Stable personal facts like favourite ratios, purchase amounts, fundraising outcomes, reading progress, completed counts, and milestone dates are durable memory.
- Do not discard a later update just because it seems small. A new number often replaces an older one.

Examples of user-turn updates that MUST be captured:
- "I just finished my fifth issue of National Geographic." → update the reading-progress memory and supersede the older "finished three issues" state when applicable.
- "I initially aimed to raise $200 and ended up raising $250." → save both the goal and the achieved amount so later questions can compute the difference.
- "I settled on a 3:1 gin-to-vermouth ratio for a classic martini." → save this as a durable user preference.
- "I spent $200 on the designer handbag and $500 on skincare." → save the concrete amounts, not just the product categories.

Do NOT save:
- Code patterns, architecture, or file paths derivable from the codebase
- Git history or recent changes (use git log for those)
- Debugging solutions (the fix is in the code)
- Ephemeral task details or in-progress work
- Anything already in the existing memories listed below

For each memory worth saving, output:
- action: "create" (new file) or "update" (modify existing)
- filename: e.g. "feedback_testing.md" (kebab-case, descriptive)
- name: human-readable name
- description: one-line description (used for future recall)
- type: user | feedback | project | reference
- scope: "global" or "project" (default to "project" if unsure)
- content: the memory content (structured with Why: and How to apply: lines for feedback/project types)
- index_entry: one-line entry for MEMORY.md (under 150 chars)
- supersedes (optional): when the user has corrected, updated, or contradicted an earlier stated fact for the same topic, set this to the filename of the earlier memory so it is retired. Only fill when you are confident the new fact replaces a specific older one; prefer leaving empty when unsure.

If nothing is worth saving, return: {"memories": []}

Respond with ONLY valid JSON: {"memories": [...]}`

// extractUserPrompt builds the user message for the extraction agent.
func extractUserPrompt(messages []Message, existingManifest, memDirDisplay string) string {
	var b strings.Builder

	if existingManifest != "" {
		b.WriteString("## Existing memory files\n\n")
		b.WriteString(existingManifest)
		b.WriteString("\n\nCheck this list before writing — update an existing file rather than creating a duplicate.\n\n")
	}

	b.WriteString("## Recent conversation\n\n")
	for _, m := range messages {
		role := string(m.Role)
		content := m.Content
		if len(content) > 2000 {
			content = content[:2000] + "\n[...truncated]"
		}
		if m.Role == RoleTool {
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			b.WriteString(fmt.Sprintf("[%s (%s)]: %s\n\n", role, m.Name, content))
			continue
		}
		b.WriteString(fmt.Sprintf("[%s]: %s\n\n", role, content))
	}

	b.WriteString(fmt.Sprintf("\nMemory directory: %s\n", memDirDisplay))

	return b.String()
}

// extractionResult represents a parsed extraction response.
type extractionResult struct {
	Memories []ExtractedMemory `json:"memories"`
}

// ExtractedMemory represents a single memory extracted from a
// conversation by the extraction LLM.
type ExtractedMemory struct {
	Action      string   `json:"action"`
	Filename    string   `json:"filename"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Type        string   `json:"type"`
	Content     string   `json:"content"`
	IndexEntry  string   `json:"index_entry"`
	Scope       string   `json:"scope"`
	Supersedes  string   `json:"supersedes,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	// SessionID, when set, is written into the fact's frontmatter so
	// multi-session queries can filter or aggregate by origin session.
	SessionID string `json:"-"`
	// ObservedOn mirrors the session date for the fact. Populated by
	// replay ingests.
	ObservedOn string `json:"-"`
	// SessionDate is the short ISO YYYY-MM-DD form of the parent
	// session's date, written into frontmatter as session_date.
	SessionDate string `json:"-"`
	// ContextPrefix, when non-empty, is a short LLM-generated situating
	// prefix prepended to the fact body before writing.
	ContextPrefix string `json:"-"`
	// ModifiedOverride, when non-empty, replaces the default "now"
	// timestamp written into the memory's frontmatter modified field.
	ModifiedOverride string `json:"-"`
}

// MaybeExtract checks if extraction should run and, if so, distils
// durable knowledge from recent conversation messages into memory
// files.
//
// This is designed to be called from a background goroutine after the
// main agentic loop completes. It is safe for concurrent use.
func (e *Extractor) MaybeExtract(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	messages []Message,
) {
	e.mu.Lock()
	if e.inProgress {
		e.mu.Unlock()
		return
	}
	e.inProgress = true
	cursor := e.lastCursor
	e.mu.Unlock()

	defer func() {
		e.mu.Lock()
		e.inProgress = false
		e.mu.Unlock()
	}()

	if len(messages)-cursor < extractMinMessages {
		return
	}

	slug := ProjectSlug(projectPath)

	var physicalHints []string
	if p, ok := e.mem.store.LocalPath(brain.MemoryGlobalPrefix()); ok {
		physicalHints = append(physicalHints, p)
	}
	if p, ok := e.mem.store.LocalPath(brain.MemoryProjectPrefix(slug)); ok {
		physicalHints = append(physicalHints, p)
	}
	if hasMemoryWrites(messages[cursor:], physicalHints...) {
		e.mu.Lock()
		e.lastCursor = len(messages)
		e.mu.Unlock()
		return
	}

	recent := messages[cursor:]
	if len(recent) > extractMaxRecent {
		recent = recent[len(recent)-extractMaxRecent:]
	}

	projectTopics, _ := e.mem.ListProjectTopics(ctx, projectPath)
	globalTopics, _ := e.mem.ListGlobalTopics(ctx)

	manifest := buildManifests(projectTopics, globalTopics)

	memDirDisplay := string(brain.MemoryProjectPrefix(slug))
	if len(physicalHints) > 0 {
		memDirDisplay = physicalHints[len(physicalHints)-1]
	}
	userPrompt := extractUserPrompt(recent, manifest, memDirDisplay)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: extractionPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   extractMaxTokens,
		Temperature: extractTemperature,
	})
	if err != nil {
		return
	}

	result := parseExtractionResult(resp.Text)
	result.Memories = appendHeuristicExtractions(recent, result.Memories)
	if len(result.Memories) == 0 {
		e.mu.Lock()
		e.lastCursor = len(messages)
		e.mu.Unlock()
		return
	}

	if e.ctx.Enabled() {
		summary := extractSessionSummary(recent)
		for i := range result.Memories {
			prefix := e.ctx.BuildPrefix(ctx, "", summary, result.Memories[i].Content)
			if prefix != "" {
				result.Memories[i].ContextPrefix = prefix
			}
		}
	}

	if err := e.mem.ApplyExtractions(ctx, slug, result.Memories); err != nil {
		slog.Warn("memory: apply extractions failed", "err", err)
	}

	e.mu.Lock()
	e.lastCursor = len(messages)
	e.mu.Unlock()
}

// ResetCursor resets the extraction cursor so the next MaybeExtract
// call processes all messages from scratch.
func (e *Extractor) ResetCursor() {
	e.mu.Lock()
	e.lastCursor = 0
	e.mu.Unlock()
}

// ExtractFromMessages runs the extraction LLM call and returns
// structured results without applying them. Useful for replay-style
// ingests that want to post-process before writing.
func ExtractFromMessages(
	ctx context.Context,
	provider llm.Provider,
	model string,
	mem *Memory,
	projectPath string,
	messages []Message,
) ([]ExtractedMemory, error) {
	if len(messages) < 2 {
		return nil, nil
	}

	slug := ProjectSlug(projectPath)
	recent := messages
	if len(recent) > extractMaxRecent {
		recent = recent[len(recent)-extractMaxRecent:]
	}

	projectTopics, _ := mem.ListProjectTopics(ctx, projectPath)
	globalTopics, _ := mem.ListGlobalTopics(ctx)

	manifest := buildManifests(projectTopics, globalTopics)

	memDirDisplay := string(brain.MemoryProjectPrefix(slug))
	userPrompt := extractUserPrompt(recent, manifest, memDirDisplay)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: extractionPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   extractMaxTokens,
		Temperature: extractTemperature,
	})
	if err != nil {
		return nil, fmt.Errorf("extraction LLM call: %w", err)
	}

	result := parseExtractionResult(resp.Text)
	return appendHeuristicExtractions(recent, result.Memories), nil
}

// buildManifests glues the project and global manifests into a single
// labelled block for the extraction prompt.
func buildManifests(projectTopics, globalTopics []TopicFile) string {
	var b strings.Builder
	if pm := buildManifest(projectTopics); pm != "" {
		b.WriteString("## Project memory files\n\n")
		b.WriteString(pm)
	}
	if gm := buildManifest(globalTopics); gm != "" {
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString("## Global memory files\n\n")
		b.WriteString(gm)
	}
	return b.String()
}

// extractSessionSummary derives a short one-line session header from a
// conversation slice.
func extractSessionSummary(messages []Message) string {
	for _, m := range messages {
		if m.Role == RoleSystem && strings.TrimSpace(m.Content) != "" {
			return truncateOneLine(m.Content, 240)
		}
	}
	for _, m := range messages {
		if m.Role == RoleUser && strings.TrimSpace(m.Content) != "" {
			return truncateOneLine(m.Content, 240)
		}
	}
	return ""
}

// truncateOneLine collapses newlines to spaces and caps at n runes.
func truncateOneLine(s string, n int) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if n > 0 && len(s) > n {
		s = s[:n] + "..."
	}
	return s
}

func appendHeuristicExtractions(messages []Message, extracted []ExtractedMemory) []ExtractedMemory {
	out := make([]ExtractedMemory, 0, len(extracted)+6)
	out = append(out, extracted...)
	out = append(out, deriveHeuristicUserFacts(messages, out)...)
	out = append(out, deriveHeuristicMilestoneFacts(messages, out)...)
	out = append(out, deriveHeuristicPendingFacts(messages, out)...)
	out = append(out, deriveHeuristicEventFacts(messages, out)...)
	out = append(out, deriveHeuristicPreferenceFacts(messages, out)...)
	return out
}

func deriveHeuristicUserFacts(messages []Message, existing []ExtractedMemory) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages)
	out := make([]ExtractedMemory, 0, heuristicUserFactLimit)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, sentence := range splitIntoFactSentences(message.Content) {
			canonical := strings.ToLower(strings.TrimSpace(sentence))
			if canonical == "" || !firstPersonFactRe.MatchString(sentence) || !hasQuantifiedFact(sentence) || seen[canonical] {
				continue
			}
			slug := heuristicFactSlug(sentence)
			if slug == "" {
				continue
			}
			observedOn := ""
			if hasAnchor {
				observedOn = resolveHeuristicObservedOn(sentence, anchor)
				if observedOn == "" {
					observedOn = anchor.UTC().Format(time.RFC3339)
				}
			}
			out = append(out, ExtractedMemory{
				Action:      "create",
				Filename:    buildHeuristicFilename("user-fact", iso, slug),
				Name:        "User Fact: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(sentence, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(sentence, observedOn),
				IndexEntry:  truncateOneLine(sentence, 140),
				ObservedOn:  observedOn,
			})
			seen[canonical] = true
			if len(out) >= heuristicUserFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicMilestoneFacts(messages []Message, existing []ExtractedMemory) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages)
	out := make([]ExtractedMemory, 0, heuristicMilestoneFactLimit)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, sentence := range splitIntoFactSentences(message.Content) {
			canonical := strings.ToLower(strings.TrimSpace(sentence))
			if canonical == "" || !hasMilestoneFact(sentence) || seen[canonical] {
				continue
			}
			slug := heuristicFactSlug(sentence)
			if slug == "" {
				continue
			}
			observedOn := ""
			if hasAnchor {
				observedOn = resolveHeuristicObservedOn(sentence, anchor)
				if observedOn == "" {
					observedOn = anchor.UTC().Format(time.RFC3339)
				}
			}
			out = append(out, ExtractedMemory{
				Action:      "create",
				Filename:    buildHeuristicFilename("user-fact", iso, "milestone-"+slug),
				Name:        "User Fact: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(sentence, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(sentence, observedOn),
				IndexEntry:  truncateOneLine(sentence, 140),
				ObservedOn:  observedOn,
			})
			seen[canonical] = true
			if len(out) >= heuristicMilestoneFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicEventFacts(messages []Message, existing []ExtractedMemory) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages)
	out := make([]ExtractedMemory, 0, 2)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, sentence := range splitIntoFactSentences(message.Content) {
			summary := inferAppointmentSummary(sentence)
			if summary == "" {
				summary = inferEventSummary(sentence)
			}
			if summary == "" {
				continue
			}
			canonical := normaliseMemoryText(summary)
			if seen[canonical] {
				continue
			}
			slug := heuristicFactSlug(summary)
			if slug == "" {
				continue
			}
			observedOn := ""
			if hasAnchor {
				observedOn = resolveHeuristicObservedOn(sentence, anchor)
				if observedOn == "" {
					observedOn = anchor.UTC().Format(time.RFC3339)
				}
			}
			out = append(out, ExtractedMemory{
				Action:      "create",
				Filename:    buildHeuristicFilename("user-fact", iso, "event-"+slug),
				Name:        "User Event: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(summary, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(summary+"\n\nEvidence: "+sentence, observedOn),
				IndexEntry:  truncateOneLine(summary, 140),
				ObservedOn:  observedOn,
			})
			seen[canonical] = true
			if len(out) >= 2 {
				return out
			}
		}
	}

	return out
}

type heuristicPreferenceCandidate struct {
	summary  string
	evidence string
}

func deriveHeuristicPreferenceFacts(messages []Message, existing []ExtractedMemory) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	for _, memory := range existing {
		if !isHeuristicPreferenceFact(memory) {
			continue
		}
		summary := strings.ToLower(heuristicPreferenceSummary(memory.Content))
		if summary != "" {
			seen[summary] = true
		}
	}
	iso := deriveHeuristicISODate(messages)
	out := make([]ExtractedMemory, 0, heuristicPreferenceFactLimit)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, candidate := range buildHeuristicPreferenceCandidates(message.Content) {
			canonical := strings.ToLower(candidate.summary)
			if canonical == "" || seen[canonical] {
				continue
			}
			slug := heuristicFactSlug(candidate.summary)
			if slug == "" {
				continue
			}
			out = append(out, ExtractedMemory{
				Action:      "create",
				Filename:    buildHeuristicFilename("user-preference", iso, slug),
				Name:        "User Preference: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(candidate.summary, 140),
				Type:        "user",
				Scope:       "global",
				Content:     buildHeuristicPreferenceContent(candidate),
				IndexEntry:  truncateOneLine(candidate.summary, 140),
			})
			seen[canonical] = true
			if len(out) >= heuristicPreferenceFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicPendingFacts(messages []Message, existing []ExtractedMemory) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages)
	out := make([]ExtractedMemory, 0, heuristicPendingFactLimit)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, sentence := range splitIntoFactSentences(message.Content) {
			for _, action := range extractPendingActions(sentence) {
				summary := buildPendingTaskSummary(action)
				canonical := normaliseMemoryText(summary)
				if canonical == "" || seen[canonical] {
					continue
				}
				slug := heuristicFactSlug(summary)
				if slug == "" {
					continue
				}
				out = append(out, ExtractedMemory{
					Action:      "create",
					Filename:    buildHeuristicFilename("user-fact", iso, "task-"+slug),
					Name:        "User Task: " + toTitleCase(action),
					Description: truncateOneLine(summary, 140),
					Type:        "user",
					Scope:       "global",
					Content:     summary + "\n\nEvidence: " + sentence,
					IndexEntry:  truncateOneLine(summary, 140),
				})
				seen[canonical] = true
				if len(out) >= heuristicPendingFactLimit {
					return out
				}
			}
		}
	}

	return out
}

func splitIntoFactSentences(content string) []string {
	parts := strings.FieldsFunc(content, func(r rune) bool {
		switch r {
		case '\n', '\r', '.', '!', '?':
			return true
		default:
			return false
		}
	})
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func hasQuantifiedFact(sentence string) bool {
	return heuristicUnitQuantityRe.MatchString(sentence) ||
		heuristicWordUnitQuantityRe.MatchString(sentence) ||
		heuristicDateTagRe.MatchString(sentence) ||
		heuristicMonthNameDateRe.MatchString(sentence) ||
		heuristicDurationFactRe.MatchString(sentence) ||
		heuristicQuantityRe.MatchString(sentence) ||
		heuristicOrdinalRe.MatchString(sentence)
}

func hasMilestoneFact(sentence string) bool {
	return firstPersonFactRe.MatchString(sentence) &&
		!hasQuantifiedFact(sentence) &&
		heuristicMilestoneTopicRe.MatchString(sentence) &&
		(heuristicMilestoneEventRe.MatchString(sentence) || heuristicMilestoneTimeRe.MatchString(sentence))
}

func heuristicFactSlug(sentence string) string {
	words := heuristicWordRe.FindAllString(strings.ToLower(sentence), -1)
	kept := make([]string, 0, 5)
	for _, word := range words {
		if heuristicStopwords[word] {
			continue
		}
		kept = append(kept, word)
		if len(kept) >= 5 {
			break
		}
	}
	return strings.Join(kept, "-")
}

func buildHeuristicFilename(prefix, iso, slug string) string {
	parts := []string{prefix}
	if iso != "" {
		parts = append(parts, iso)
	}
	parts = append(parts, slug)
	return strings.Join(parts, "-") + ".md"
}

func deriveHeuristicISODate(messages []Message) string {
	if anchor, ok := deriveHeuristicSessionAnchor(messages); ok {
		return anchor.UTC().Format("2006-01-02")
	}
	return ""
}

func deriveHeuristicSessionAnchor(messages []Message) (time.Time, bool) {
	for _, message := range messages {
		if message.Role != RoleSystem {
			continue
		}
		match := heuristicSessionDateRe.FindString(message.Content)
		if match == "" {
			continue
		}
		for _, layout := range []string{
			"2006/01/02 (Mon) 15:04",
			"2006/01/02 15:04",
			"2006/01/02",
			"2006-01-02 15:04",
			"2006-01-02",
			time.RFC3339,
		} {
			if parsed, err := time.Parse(layout, match); err == nil {
				return parsed.UTC(), true
			}
		}
	}
	return time.Time{}, false
}

func buildExistingMemoryTextSet(existing []ExtractedMemory) map[string]bool {
	seen := make(map[string]bool, len(existing)*4)
	for _, memory := range existing {
		for _, candidate := range []string{memory.Content, memory.Description, memory.IndexEntry, inferSearchableSummary(memory.Content)} {
			normalised := normaliseMemoryText(candidate)
			if normalised != "" {
				seen[normalised] = true
			}
		}
	}
	return seen
}

func normaliseMemoryText(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}

func extractPendingActions(sentence string) []string {
	if strings.HasSuffix(strings.TrimSpace(sentence), "?") {
		return nil
	}
	matched := heuristicPendingActionLeadRe.FindStringSubmatch(sentence)
	if len(matched) < 2 {
		return nil
	}
	fragment := strings.TrimSpace(matched[1])
	if fragment == "" {
		return nil
	}

	parts := regexp.MustCompile(`(?i)\s*(?:,|;|\bthen\b|\band\b)\s*`).Split(fragment, -1)
	out := make([]string, 0, len(parts))
	current := ""
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if heuristicPendingActionStartRe.MatchString(part) {
			if current != "" {
				out = append(out, current)
			}
			current = part
			continue
		}
		if current != "" {
			current = strings.TrimSpace(current + " " + part)
		}
	}

	if current != "" {
		out = append(out, current)
	}
	if len(out) == 0 {
		cleaned := cleanPendingActionClause(fragment)
		if cleaned == "" {
			return nil
		}
		return []string{cleaned}
	}

	cleaned := make([]string, 0, len(out))
	for _, action := range out {
		if action = cleanPendingActionClause(action); action != "" {
			cleaned = append(cleaned, action)
		}
	}
	return cleaned
}

func cleanPendingActionClause(value string) string {
	trimmed := strings.TrimSpace(value)
	trimmed = strings.TrimLeft(trimmed, ",:; ")
	trimmed = strings.TrimRight(trimmed, ",:; ")
	return strings.Join(strings.Fields(trimmed), " ")
}

func buildPendingTaskSummary(action string) string {
	return ensureTrailingFullStop("The user still needs to " + stripTrailingFullStop(action))
}

func inferPendingTaskSummary(text string) string {
	for _, sentence := range splitIntoFactSentences(text) {
		if actions := extractPendingActions(sentence); len(actions) > 0 {
			return buildPendingTaskSummary(actions[0])
		}
	}
	return ""
}

func inferSearchableSummary(content string) string {
	text := stripSearchPrefixes(content)
	if text == "" {
		return ""
	}
	if summary := inferPendingTaskSummary(text); summary != "" {
		return summary
	}
	if summary := inferAppointmentSummary(text); summary != "" {
		return summary
	}
	if summary := inferEventSummary(text); summary != "" {
		return summary
	}
	if preference := inferHeuristicPreference(text); preference != nil {
		return preference.summary
	}
	return ""
}

func stripSearchPrefixes(content string) string {
	trimmed := strings.TrimSpace(content)
	for _, prefix := range []string{"[Date:", "[Observed on "} {
		if strings.HasPrefix(trimmed, prefix) {
			if idx := strings.Index(trimmed, "]\n\n"); idx >= 0 {
				trimmed = strings.TrimSpace(trimmed[idx+3:])
			}
		}
	}
	return strings.TrimSpace(trimmed)
}

func buildHeuristicPreferenceCandidates(content string) []heuristicPreferenceCandidate {
	candidates := []string{normalisePreferenceText(content)}
	candidates = append(candidates, splitIntoFactSentences(content)...)
	seen := make(map[string]bool)
	out := make([]heuristicPreferenceCandidate, 0, 2)
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		inferred := inferHeuristicPreference(candidate)
		if inferred == nil {
			continue
		}
		canonical := strings.ToLower(inferred.summary)
		if seen[canonical] {
			continue
		}
		seen[canonical] = true
		out = append(out, *inferred)
	}
	return out
}

func inferHeuristicPreference(content string) *heuristicPreferenceCandidate {
	text := normalisePreferenceText(content)
	if text == "" {
		return nil
	}
	if explicit := inferExplicitPreference(text); explicit != nil {
		return explicit
	}
	if compatibility := inferCompatibilityPreference(text); compatibility != nil {
		return compatibility
	}
	if constrained := inferConstraintPreference(text); constrained != nil {
		return constrained
	}
	if advanced := inferAdvancedPreference(text); advanced != nil {
		return advanced
	}
	return nil
}

func inferExplicitPreference(text string) *heuristicPreferenceCandidate {
	if matched := heuristicPreferenceBesidesLikeRe.FindStringSubmatch(text); len(matched) >= 3 {
		first := cleanPreferenceFragment(matched[1])
		second := cleanPreferenceFragment(matched[2])
		if first != "" && second != "" {
			if hotelMatch := regexp.MustCompile(`(?i)^hotels?\s+with\s+(.+)$`).FindStringSubmatch(second); len(hotelMatch) >= 2 {
				return &heuristicPreferenceCandidate{
					summary:  fmt.Sprintf("The user prefers hotels with %s and %s.", first, cleanPreferenceFragment(hotelMatch[1])),
					evidence: text,
				}
			}
			return &heuristicPreferenceCandidate{
				summary:  fmt.Sprintf("The user prefers %s and %s.", first, second),
				evidence: text,
			}
		}
	}
	if matched := heuristicPreferenceLikeRe.FindStringSubmatch(text); len(matched) >= 2 {
		fragment := cleanPreferenceFragment(matched[1])
		if fragment != "" {
			return &heuristicPreferenceCandidate{
				summary:  fmt.Sprintf("The user prefers %s.", fragment),
				evidence: text,
			}
		}
	}
	return nil
}

func inferCompatibilityPreference(text string) *heuristicPreferenceCandidate {
	rawSubject := captureGroup(heuristicPreferenceCompatibleRe, text)
	if rawSubject == "" {
		rawSubject = captureGroup(heuristicPreferenceDesignedForRe, text)
	}
	if rawSubject == "" {
		rawSubject = captureGroup(heuristicPreferenceAsUserRe, text)
	}
	if rawSubject == "" {
		return nil
	}
	subject := cleanPreferenceFragment(rawSubject)
	if subject == "" {
		return nil
	}
	return &heuristicPreferenceCandidate{
		summary:  fmt.Sprintf("The user prefers %s compatible with their %s.", inferPreferenceCategory(text), subject),
		evidence: text,
	}
}

func inferConstraintPreference(text string) *heuristicPreferenceCandidate {
	category := inferRecommendationCategory(text)
	if category == "" || !heuristicRecommendationRequestRe.MatchString(text) {
		return nil
	}
	constraints := collectRecommendationConstraints(text)
	if len(constraints) == 0 {
		return nil
	}
	return &heuristicPreferenceCandidate{
		summary:  fmt.Sprintf("The user prefers %s with these constraints: %s.", category, strings.Join(constraints, "; ")),
		evidence: text,
	}
}

func inferAdvancedPreference(text string) *heuristicPreferenceCandidate {
	topic := captureGroup(heuristicPreferenceFieldRe, text)
	if topic == "" {
		topic = captureGroup(heuristicPreferenceAdvancedRe, text)
	}
	if topic == "" {
		return nil
	}
	if !heuristicPreferenceSkipBasicsRe.MatchString(text) &&
		!heuristicPreferenceWorkingInFieldRe.MatchString(text) &&
		!strings.Contains(strings.ToLower(text), "advanced") {
		return nil
	}
	cleaned := cleanPreferenceFragment(topic)
	if cleaned == "" {
		return nil
	}
	return &heuristicPreferenceCandidate{
		summary:  fmt.Sprintf("The user prefers advanced publications, papers, and conferences on %s rather than introductory material.", cleaned),
		evidence: text,
	}
}

func captureGroup(pattern *regexp.Regexp, text string) string {
	matched := pattern.FindStringSubmatch(text)
	if len(matched) < 2 {
		return ""
	}
	return strings.TrimSpace(matched[1])
}

func inferPreferenceCategory(text string) string {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "camera"), strings.Contains(lower, "photography"), strings.Contains(lower, "lens"),
		strings.Contains(lower, "flash"), strings.Contains(lower, "tripod"), strings.Contains(lower, "camera bag"),
		strings.Contains(lower, "gear"):
		return "photography accessories and gear"
	case strings.Contains(lower, "phone"), strings.Contains(lower, "iphone"),
		strings.Contains(lower, "screen protector"), strings.Contains(lower, "power bank"):
		return "phone accessories"
	case inferRecommendationCategory(text) != "":
		return inferRecommendationCategory(text)
	default:
		return "accessories and options"
	}
}

func inferRecommendationCategory(text string) string {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "film"), strings.Contains(lower, "movie"), strings.Contains(lower, "cinema"):
		return "films"
	case strings.Contains(lower, "show"), strings.Contains(lower, "series"), strings.Contains(lower, "tv"):
		return "shows"
	case strings.Contains(lower, "book"), strings.Contains(lower, "novel"), strings.Contains(lower, "read"), strings.Contains(lower, "reading"):
		return "books"
	case strings.Contains(lower, "hotel"), strings.Contains(lower, "accommodation"), strings.Contains(lower, "stay"):
		return "hotels"
	case strings.Contains(lower, "restaurant"), strings.Contains(lower, "dinner"), strings.Contains(lower, "lunch"):
		return "restaurants"
	case strings.Contains(lower, "game"):
		return "games"
	case strings.Contains(lower, "podcast"):
		return "podcasts"
	default:
		return ""
	}
}

func collectRecommendationConstraints(text string) []string {
	seen := make(map[string]bool)
	out := make([]string, 0, 4)
	add := func(value string) {
		cleaned := cleanPreferenceFragment(value)
		canonical := strings.ToLower(cleaned)
		if cleaned == "" || seen[canonical] {
			return
		}
		seen[canonical] = true
		out = append(out, cleaned)
	}

	if heuristicRecommendationFamilyRe.MatchString(text) {
		add("family-friendly")
	}
	if light := heuristicRecommendationLightRe.FindString(text); light != "" {
		add(light)
	}
	if notToo := captureGroup(heuristicRecommendationNotTooRe, text); notToo != "" {
		add("not too " + notToo)
	}
	if under := captureGroup(heuristicRecommendationUnderRe, text); under != "" {
		add("under " + under)
	}
	if without := captureGroup(heuristicRecommendationWithoutRe, text); without != "" {
		add("without " + without)
	}

	return out
}

func normalisePreferenceText(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func cleanPreferenceFragment(value string) string {
	return strings.Trim(strings.Join(strings.Fields(strings.Trim(value, ",:; ")), " "), ",:; ")
}

func buildHeuristicPreferenceContent(candidate heuristicPreferenceCandidate) string {
	return candidate.summary + "\n\nEvidence: " + candidate.evidence
}

func heuristicPreferenceSummary(content string) string {
	marker := "\n\nEvidence:"
	if idx := strings.Index(content, marker); idx >= 0 {
		return strings.TrimSpace(content[:idx])
	}
	return strings.TrimSpace(content)
}

func isHeuristicPreferenceFact(memory ExtractedMemory) bool {
	return memory.Scope == "global" &&
		memory.Type == "user" &&
		strings.HasPrefix(memory.Filename, "user-preference-")
}

func captureWholeMatch(pattern *regexp.Regexp, text string) string {
	return strings.TrimSpace(pattern.FindString(text))
}

func resolveHeuristicObservedOn(sentence string, anchor time.Time) string {
	if matched := captureWholeMatch(heuristicDateTagRe, sentence); matched != "" {
		for _, layout := range []string{"2006-01-02", "2006/01/02"} {
			if parsed, err := time.Parse(layout, matched); err == nil {
				return parsed.UTC().Format(time.RFC3339)
			}
		}
	}

	if parsed, ok := parseMonthNameDate(sentence, anchor); ok {
		return parsed.UTC().Format(time.RFC3339)
	}

	expansion := query.ExpandTemporal(sentence, anchor.Format("2006/01/02 (Mon) 15:04"))
	if len(expansion.DateHints) == 0 {
		return ""
	}
	if parsed, err := time.Parse("2006/01/02", expansion.DateHints[0]); err == nil {
		return parsed.UTC().Format(time.RFC3339)
	}
	return ""
}

func parseMonthNameDate(text string, anchor time.Time) (time.Time, bool) {
	matched := heuristicMonthNameDateRe.FindStringSubmatch(text)
	if len(matched) == 0 {
		return time.Time{}, false
	}
	full := matched[0]
	parts := strings.Fields(strings.ReplaceAll(strings.ReplaceAll(full, ",", ""), ".", ""))
	if len(parts) < 2 {
		return time.Time{}, false
	}

	month := strings.ToLower(parts[0])
	dayRaw := strings.TrimRight(parts[1], "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	day, err := strconv.Atoi(dayRaw)
	if err != nil {
		return time.Time{}, false
	}

	year := anchor.Year()
	if len(parts) >= 3 {
		if parsedYear, err := strconv.Atoi(parts[2]); err == nil {
			year = parsedYear
		}
	}

	monthMap := map[string]time.Month{
		"january": time.January, "february": time.February, "march": time.March,
		"april": time.April, "may": time.May, "june": time.June,
		"july": time.July, "august": time.August, "september": time.September,
		"october": time.October, "november": time.November, "december": time.December,
	}
	monthValue, ok := monthMap[month]
	if !ok {
		return time.Time{}, false
	}

	resolved := time.Date(year, monthValue, day, 0, 0, 0, 0, time.UTC)
	if len(parts) < 3 && resolved.After(anchor.Add(24*time.Hour)) {
		resolved = resolved.AddDate(-1, 0, 0)
	}
	return resolved, true
}

func withObservedDatePrefix(content, observedOn string) string {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" || observedOn == "" {
		return trimmed
	}
	prefix := buildInlineDateTokens(observedOn)
	if prefix == "" {
		return trimmed
	}
	return prefix + trimmed
}

func buildInlineDateTokens(rfc3339 string) string {
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(rfc3339))
	if err != nil {
		return ""
	}
	return fmt.Sprintf("[Date: %s %s %s %d]\n\n",
		parsed.Format("2006-01-02"),
		parsed.Weekday().String(),
		parsed.Month().String(),
		parsed.Year(),
	)
}

func inferAppointmentSummary(text string) string {
	for _, sentence := range splitIntoFactSentences(text) {
		if !heuristicAppointmentRe.MatchString(sentence) || sentence == "" ||
			heuristicPendingActionLeadRe.MatchString(sentence) ||
			strings.HasSuffix(strings.TrimSpace(sentence), "?") {
			continue
		}
		medicalEntity := captureWholeMatch(heuristicMedicalEntityRe, sentence)
		withPerson := captureGroup(heuristicWithPersonRe, sentence)
		temporal := extractTemporalAnchor(sentence)
		subject := "a medical appointment"
		if medicalEntity != "" {
			subject = withIndefiniteArticle(medicalEntity) + " " + medicalEntity + " appointment"
		}
		parts := []string{"The user has " + subject}
		if withPerson != "" {
			parts = append(parts, "with "+withPerson)
		}
		if temporal != "" {
			parts = append(parts, temporal)
		}
		return ensureTrailingFullStop(strings.Join(parts, " "))
	}
	return ""
}

func inferEventSummary(text string) string {
	for _, sentence := range splitIntoFactSentences(text) {
		if summary := inferReligiousServiceSummary(sentence); summary != "" {
			return summary
		}
		if !firstPersonFactRe.MatchString(sentence) ||
			!heuristicEventRe.MatchString(sentence) ||
			heuristicPendingActionLeadRe.MatchString(sentence) ||
			!heuristicEventAttendanceRe.MatchString(sentence) ||
			strings.HasSuffix(strings.TrimSpace(sentence), "?") {
			continue
		}
		title := captureGroup(heuristicEventTitleRe, sentence)
		if title == "" {
			title = extractLooseEventPhrase(sentence)
		}
		if title == "" {
			continue
		}
		parts := []string{"The user attended " + prefixEventPhrase(title)}
		if temporal := extractTemporalAnchor(sentence); temporal != "" {
			parts = append(parts, temporal)
		}
		return ensureTrailingFullStop(strings.Join(parts, " "))
	}
	return ""
}

func inferReligiousServiceSummary(sentence string) string {
	if !firstPersonFactRe.MatchString(sentence) ||
		heuristicPendingActionLeadRe.MatchString(sentence) ||
		strings.HasSuffix(strings.TrimSpace(sentence), "?") {
		return ""
	}
	service := captureGroup(heuristicReligiousServiceRe, sentence)
	if service == "" {
		return ""
	}
	return ensureTrailingFullStop("The user attended " + prefixEventPhrase(service))
}

func extractLooseEventPhrase(sentence string) string {
	re := regexp.MustCompile(`(?i)\b(?:a|an|the)\s+([^,.!?]+?\s+(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar|service|mass|worship|prayer))\b`)
	matched := strings.TrimSpace(re.FindString(sentence))
	return matched
}

func prefixEventPhrase(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	if regexp.MustCompile(`(?i)^(?:a|an|the)\b`).MatchString(trimmed) {
		return trimmed
	}
	return "the " + trimmed
}

func extractTemporalAnchor(text string) string {
	dateAnchor := captureWholeMatch(heuristicDateTagRe, text)
	if dateAnchor == "" {
		dateAnchor = captureWholeMatch(heuristicRelativeDateRe, text)
	}
	timeAnchor := captureGroup(heuristicClockTimeRe, text)
	parts := make([]string, 0, 2)
	if dateAnchor != "" {
		parts = append(parts, "on "+dateAnchor)
	}
	if timeAnchor != "" {
		parts = append(parts, "at "+timeAnchor)
	}
	return strings.Join(parts, " ")
}

func ensureTrailingFullStop(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	if strings.HasSuffix(trimmed, ".") || strings.HasSuffix(trimmed, "!") || strings.HasSuffix(trimmed, "?") {
		return trimmed
	}
	return trimmed + "."
}

func stripTrailingFullStop(value string) string {
	return strings.TrimRight(strings.TrimSpace(value), ".!?")
}

func withIndefiniteArticle(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return "a"
	}
	switch strings.ToLower(trimmed[:1]) {
	case "a", "e", "i", "o", "u":
		return "an"
	default:
		return "a"
	}
}

func toTitleCase(value string) string {
	parts := strings.Fields(value)
	for i, part := range parts {
		if part == "" {
			continue
		}
		parts[i] = strings.ToUpper(part[:1]) + part[1:]
	}
	return strings.Join(parts, " ")
}

// hasMemoryWrites checks if any assistant message in the slice contains
// tool calls that wrote to either the project or global memory
// directory.
func hasMemoryWrites(messages []Message, memDirs ...string) bool {
	for _, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Name == "write" || tc.Name == "edit" {
				args := string(tc.Arguments)
				for _, dir := range memDirs {
					if dir != "" && strings.Contains(args, dir) {
						return true
					}
				}
				if strings.Contains(args, "memory/") {
					return true
				}
			}
		}
	}
	return false
}

// parseExtractionResult extracts memories from the model's JSON
// response.
func parseExtractionResult(content string) extractionResult {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result extractionResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return extractionResult{}
	}

	return result
}

// ApplyExtractions writes extracted memories through the brain store
// and updates MEMORY.md indices. All writes happen in a single batch.
func (m *Memory) ApplyExtractions(ctx context.Context, projectSlug string, memories []ExtractedMemory) error {
	var projectEntries, globalEntries []string
	type pendingTopic struct {
		path    brain.Path
		content []byte
	}
	var pending []pendingTopic

	for _, em := range memories {
		if em.Filename == "" || em.Content == "" {
			continue
		}

		filename := sanitiseFilename(em.Filename)
		if !strings.HasSuffix(filename, ".md") {
			filename += ".md"
		}
		slug := strings.TrimSuffix(filename, ".md")

		var p brain.Path
		if em.Scope == "global" {
			p = brain.MemoryGlobalTopic(slug)
		} else {
			p = brain.MemoryProjectTopic(projectSlug, slug)
		}

		content := buildTopicFileContent(em)
		pending = append(pending, pendingTopic{path: p, content: content})

		if em.IndexEntry != "" {
			if em.Scope == "global" {
				globalEntries = append(globalEntries, em.IndexEntry)
			} else {
				projectEntries = append(projectEntries, em.IndexEntry)
			}
		}
	}

	if len(pending) == 0 {
		return nil
	}

	return m.store.Batch(ctx, brain.BatchOptions{Reason: "extract"}, func(b brain.Batch) error {
		for _, p := range pending {
			if err := b.Write(ctx, p.path, p.content); err != nil {
				return err
			}
		}
		for _, em := range memories {
			if em.Supersedes == "" {
				continue
			}
			oldFile := sanitiseFilename(em.Supersedes)
			if !strings.HasSuffix(oldFile, ".md") {
				oldFile += ".md"
			}
			oldSlug := strings.TrimSuffix(oldFile, ".md")
			newFile := sanitiseFilename(em.Filename)
			if !strings.HasSuffix(newFile, ".md") {
				newFile += ".md"
			}
			var oldPath brain.Path
			if em.Scope == "global" {
				oldPath = brain.MemoryGlobalTopic(oldSlug)
			} else {
				oldPath = brain.MemoryProjectTopic(projectSlug, oldSlug)
			}
			if err := stampSupersededBy(ctx, b, oldPath, newFile); err != nil {
				return err
			}
		}
		if len(projectEntries) > 0 {
			if err := m.appendIndexEntries(ctx, b, brain.MemoryProjectIndex(projectSlug), projectEntries); err != nil {
				return err
			}
		}
		if len(globalEntries) > 0 {
			if err := m.appendIndexEntries(ctx, b, brain.MemoryGlobalIndex(), globalEntries); err != nil {
				return err
			}
		}
		return nil
	})
}

// stampSupersededBy rewrites an existing memory file's frontmatter with
// a superseded_by pointer to the new file.
func stampSupersededBy(ctx context.Context, b brain.Batch, oldPath brain.Path, newFile string) error {
	raw, err := b.Read(ctx, oldPath)
	if err != nil {
		return nil
	}
	content := string(raw)
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return nil
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return nil
	}

	replaced := false
	for i := 1; i < closeIdx; i++ {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "superseded_by:") {
			lines[i] = fmt.Sprintf("superseded_by: %s", newFile)
			replaced = true
			break
		}
	}
	if !replaced {
		inserted := make([]string, 0, len(lines)+1)
		inserted = append(inserted, lines[:closeIdx]...)
		inserted = append(inserted, fmt.Sprintf("superseded_by: %s", newFile))
		inserted = append(inserted, lines[closeIdx:]...)
		lines = inserted
	}
	return b.Write(ctx, oldPath, []byte(strings.Join(lines, "\n")))
}

// sanitiseFilename strips path traversal from an LLM-supplied filename.
func sanitiseFilename(name string) string {
	if idx := strings.LastIndexAny(name, "/\\"); idx >= 0 {
		name = name[idx+1:]
	}
	return name
}

// buildTopicFileContent builds the full markdown file contents for an
// extracted memory, including YAML frontmatter.
func buildTopicFileContent(em ExtractedMemory) []byte {
	now := time.Now().UTC().Format(time.RFC3339)
	modified := now
	created := now
	if em.ModifiedOverride != "" {
		modified = em.ModifiedOverride
		created = em.ModifiedOverride
	}
	var b strings.Builder
	b.WriteString("---\n")
	if em.Name != "" {
		b.WriteString(fmt.Sprintf("name: %s\n", em.Name))
	}
	if em.Description != "" {
		b.WriteString(fmt.Sprintf("description: %s\n", em.Description))
	}
	if em.Type != "" {
		b.WriteString(fmt.Sprintf("type: %s\n", em.Type))
	}
	if em.Action == "create" {
		b.WriteString(fmt.Sprintf("created: %s\n", created))
	}
	b.WriteString(fmt.Sprintf("modified: %s\n", modified))
	b.WriteString("source: session\n")
	if em.Supersedes != "" {
		b.WriteString(fmt.Sprintf("supersedes: %s\n", em.Supersedes))
	}
	if em.SessionID != "" {
		b.WriteString(fmt.Sprintf("session_id: %s\n", em.SessionID))
	}
	if em.ObservedOn != "" {
		b.WriteString(fmt.Sprintf("observed_on: %s\n", em.ObservedOn))
	}
	if em.SessionDate != "" {
		b.WriteString(fmt.Sprintf("session_date: %s\n", em.SessionDate))
	}
	if len(em.Tags) > 0 {
		b.WriteString(fmt.Sprintf("tags: [%s]\n", strings.Join(em.Tags, ", ")))
	}
	b.WriteString("---\n\n")
	b.WriteString(ApplyContextualPrefix(em.ContextPrefix, em.Content))
	b.WriteString("\n")
	return []byte(b.String())
}

// appendIndexEntries reads the current index, appends any new entries
// that are not already present, and writes it back via the batch.
func (m *Memory) appendIndexEntries(ctx context.Context, b brain.Batch, indexPath brain.Path, entries []string) error {
	var content string
	existing, err := b.Read(ctx, indexPath)
	if err == nil {
		content = strings.TrimSpace(string(existing))
	}
	for _, entry := range entries {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		if strings.Contains(content, entry) {
			continue
		}
		if content != "" {
			content += "\n"
		}
		content += entry
	}
	return b.Write(ctx, indexPath, []byte(content+"\n"))
}

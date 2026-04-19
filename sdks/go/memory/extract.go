// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"sort"
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
	extractTemperature           = 0
	extractMinMessages           = 2
	extractMaxRecent             = 80
	existingMemoryLimit          = 24
	existingMemoryPreviewLimit   = 400
	heuristicUserFactLimit       = 2
	heuristicMilestoneFactLimit  = 2
	heuristicPreferenceFactLimit = 2
	heuristicPendingFactLimit    = 3
	heuristicAssistantFactLimit  = 8
)

var (
	heuristicDateTagRe                  = regexp.MustCompile(`\b\d{4}[-/]\d{2}[-/]\d{2}\b`)
	heuristicMonthNameDateRe            = regexp.MustCompile(`(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b`)
	heuristicQuantityRe                 = regexp.MustCompile(`\b\d{1,6}(?:\.\d+)?\b`)
	heuristicUnitQuantityRe             = regexp.MustCompile(`(?i)\b(\d{1,6}(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?|km|kilometres?|miles?|metres?|meters?|kg|kilograms?|pounds?|lbs?|grams?|percent|%|kbps|mbps|gbps|tbps|kb/s|mb/s|gb/s|tb/s)\b`)
	heuristicWordUnitQuantityRe         = regexp.MustCompile(`(?i)\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?)\b`)
	heuristicDurationFactRe             = regexp.MustCompile(`(?i)\b(?:\d{1,4}-day|[a-z]+-day|[a-z]+-week|[a-z]+-month|[a-z]+-year|week-long|month-long|year-long)\b`)
	heuristicOrdinalRe                  = regexp.MustCompile(`(?i)\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b`)
	heuristicCadenceFactRe              = regexp.MustCompile(`(?i)\b(?:(?:every|each)\s+(?:other\s+)?(?:day|morning|afternoon|evening|night|weekday|weekend|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:biweekly|fortnightly|usually|normally|annually|yearly)|(?:once|twice)\s+(?:a|per)\s+(?:day|week|month|year)|\d+\s+times?\s+(?:a|per)\s+(?:day|week|month|year))\b`)
	heuristicAssistantShiftHeaderRe     = regexp.MustCompile(`(?i)\b(?:shift|morning|evening|night|day|on-?call|coverage|rotation|roster)\b`)
	heuristicAssistantShiftValueRe      = regexp.MustCompile(`(?i)\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b`)
	heuristicStorageLocationFactRe      = regexp.MustCompile(`(?i)\b(?:i|i'm|i’ve|i've|i have)\s+(?:been\s+)?(?:keep(?:ing)?|kept|stor(?:e|ing|ed)|stash(?:ed|ing)?|leave|left|put|placed)\b[^.!?\n]*\b(?:under|inside|in|on|at|behind|beside|next to)\b`)
	heuristicSessionDateRe              = regexp.MustCompile(`\b\d{4}[/-]\d{2}[/-]\d{2}(?:\s+\([A-Za-z]{3}\))?(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b`)
	heuristicSessionIDRe                = regexp.MustCompile(`(?im)\bsession[_ ]id\s*[:=]\s*([A-Za-z0-9._-]+)\b`)
	heuristicMarkdownTableLineRe        = regexp.MustCompile(`^\s*\|.*\|\s*$`)
	heuristicMarkdownSeparatorCellRe    = regexp.MustCompile(`^\s*:?-{3,}:?\s*$`)
	heuristicWeekdayCellRe              = regexp.MustCompile(`(?i)^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$`)
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
	heuristicFilenameDatedRe            = regexp.MustCompile(`^(user-(?:fact|preference))-(\d{4}-\d{2}-\d{2})-(.+)\.md$`)
	heuristicFilenameRe                 = regexp.MustCompile(`^(user-(?:fact|preference))-(.+)\.md$`)
	heuristicFileSegmentUnsafeRe        = regexp.MustCompile(`[^A-Za-z0-9._-]+`)
	heuristicFileSegmentDashRe          = regexp.MustCompile(`-+`)
	summaryTokenRe                      = regexp.MustCompile(`[a-z0-9]+`)
	autoTagWeekdayRe                    = regexp.MustCompile(`(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`)
	autoTagQuantityRe                   = regexp.MustCompile(`\b\d{1,6}(?:\.\d+)?\b`)
	autoTagProperNounRe                 = regexp.MustCompile(`\b[A-Z][a-zA-Z]+\b`)
	autoTagMoneyRe                      = regexp.MustCompile(`[\$£€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?`)
	autoTagRelativeTemporalRe           = regexp.MustCompile(`(?i)\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b`)
	autoTagClockTimeRe                  = regexp.MustCompile(`(?i)\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b`)
	autoTagPendingActionRe              = regexp.MustCompile(`(?i)\b(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|renew|cancel|follow\s+up)\b`)
	autoTagMedicalRe                    = regexp.MustCompile(`(?i)\b(?:appointment|check-?up|consultation|follow-?up|doctor|gp|dentist|dermatologist|orthodontist|hygienist|therapist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian|clinic|hospital|prescription)\b`)
	autoTagEventRe                      = regexp.MustCompile(`(?i)\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar)\b`)
	autoTagEntertainmentRe              = regexp.MustCompile(`(?i)\b(?:film|movie|show|series|book|novel|game|podcast|cinema)\b`)
	autoTagExtraMedicalEntityRe         = regexp.MustCompile(`(?i)\b(?:clinic|hospital|prescription)\b`)
	extractionTrailingCommaRe           = regexp.MustCompile(`,(\s*[}\]])`)
)

type sentenceSplitProtection struct {
	pattern     *regexp.Regexp
	placeholder string
	original    string
}

var heuristicSentenceSplitProtections = []sentenceSplitProtection{
	{pattern: regexp.MustCompile(`\bDr\.`), placeholder: `Dr__DOT__`, original: `Dr.`},
	{pattern: regexp.MustCompile(`\bMr\.`), placeholder: `Mr__DOT__`, original: `Mr.`},
	{pattern: regexp.MustCompile(`\bMrs\.`), placeholder: `Mrs__DOT__`, original: `Mrs.`},
	{pattern: regexp.MustCompile(`\bMs\.`), placeholder: `Ms__DOT__`, original: `Ms.`},
	{pattern: regexp.MustCompile(`\bProf\.`), placeholder: `Prof__DOT__`, original: `Prof.`},
}

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

var autoTagStopNouns = map[string]bool{
	"the":       true,
	"this":      true,
	"that":      true,
	"these":     true,
	"those":     true,
	"when":      true,
	"where":     true,
	"what":      true,
	"who":       true,
	"why":       true,
	"how":       true,
	"observed":  true,
	"date":      true,
	"mon":       true,
	"tue":       true,
	"wed":       true,
	"thu":       true,
	"fri":       true,
	"sat":       true,
	"sun":       true,
	"user":      true,
	"assistant": true,
}

var summaryStopwords = map[string]bool{
	"a":           true,
	"an":          true,
	"and":         true,
	"appointment": true,
	"event":       true,
	"has":         true,
	"is":          true,
	"task":        true,
	"the":         true,
	"this":        true,
	"to":          true,
	"user":        true,
	"with":        true,
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

Preserve structured assistant outputs when they contain durable facts. If the assistant gives a roster, timetable, schedule, table, comparison, shortlist, or direct factual answer, keep the exact names, positions, shifts, prices, speeds, sizes, counts, and other concrete attributes rather than flattening them into a vague summary.

Preserve concrete historical facts exactly when they matter. Keep explicit user experiences, measurements, comparisons, relatives, places, and time references in the memory content instead of flattening them into a vague preference or goal. Examples:
- "My car was getting 30 miles per gallon in the city a few months ago." should preserve the 30 miles per gallon fact and timeframe.
- "I went on a two-week trip to Europe with my parents and younger brother last month." should preserve the trip, relatives, destination, and timeframe.
- "I've been sticking to my daily tidying routine for 4 weeks." should preserve the duration as a concrete user fact.
- If the conversation also reveals a broader preference, keep the concrete event as well rather than replacing it.

When a user states a concrete personal measurement, duration, past event, or status update, create a separate user memory for that fact even if the rest of the session is mostly recommendations, troubleshooting, or planning.

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
- "Sunday roster: Admon, 8 am - 4 pm (Day Shift)." → save the person's name, shift, and exact hours.
- "You upgraded your internet plan to 500 Mbps." → save the exact plan value, not a vague note about faster internet.

Updates and quantitative facts that MUST be captured:
- When the user gives a new count, total, amount, ratio, progress update, milestone, or outcome, save it even if an older memory on the same topic already exists.
- Prefer an update with supersedes when the new statement revises prior state.
- Stable personal facts like favourite ratios, purchase amounts, fundraising outcomes, reading progress, completed counts, and milestone dates are durable memory.
- Do not discard a later update just because it seems small. A new number often replaces an older one.
- When a later message changes a recurring cadence, schedule, count, price, bandwidth, screen size, or other exact attribute, preserve the new value explicitly and supersede the older one when appropriate.
- Do not round away specific attributes such as 55-inch, 500 Mbps, 8 am - 4 pm, or edition counts. Keep the exact value in the memory content.

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
- content:
  - for user and reference memories: direct factual prose that preserves the exact people, places, dates, relative time phrases, quantities, and historical events from the conversation. Prefer concrete statements over generic advice.
  - for feedback and project memories: structured with Why: and How to apply: lines
- index_entry: one-line entry for MEMORY.md (under 150 chars)
- supersedes (optional): when the user has corrected, updated, or contradicted an earlier stated fact for the same topic, set this to the filename of the earlier memory so it is retired. Only fill when you are confident the new fact replaces a specific older one; prefer leaving empty when unsure.

If nothing is worth saving, return: {"memories": []}

Respond with ONLY valid JSON: {"memories": [...]}`

type existingMemorySummary struct {
	Path        string
	Scope       string
	Name        string
	Description string
	Type        string
	Modified    string
	Content     string
}

// extractUserPrompt builds the user message for the extraction agent.
func extractUserPrompt(messages []Message, existingMemories []existingMemorySummary) string {
	var b strings.Builder

	if len(existingMemories) > 0 {
		b.WriteString("## Existing memories\n\n")
		for _, memory := range existingMemories {
			b.WriteString(fmt.Sprintf("### [%s] %s\n", memory.Scope, baseName(memory.Path)))
			if memory.Name != "" {
				b.WriteString(fmt.Sprintf("name: %s\n", memory.Name))
			}
			if memory.Description != "" {
				b.WriteString(fmt.Sprintf("description: %s\n", memory.Description))
			}
			if memory.Type != "" {
				b.WriteString(fmt.Sprintf("type: %s\n", memory.Type))
			}
			if memory.Modified != "" {
				b.WriteString(fmt.Sprintf("modified: %s\n", memory.Modified))
			}
			if memory.Content != "" {
				b.WriteString(fmt.Sprintf("content: %s\n", memory.Content))
			}
			b.WriteString("\n")
		}
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

	return b.String()
}

// extractionResult represents a parsed extraction response.
type extractionResult struct {
	Memories []ExtractedMemory `json:"memories"`
	Parsed   bool              `json:"-"`
}

type rawExtractionResult struct {
	Memories []rawExtractedMemory `json:"memories"`
}

type rawExtractedMemory struct {
	Action                string   `json:"action"`
	Filename              string   `json:"filename"`
	Name                  string   `json:"name"`
	Description           string   `json:"description"`
	Type                  string   `json:"type"`
	Content               string   `json:"content"`
	IndexEntry            string   `json:"indexEntry"`
	IndexEntrySnake       string   `json:"index_entry"`
	Scope                 string   `json:"scope"`
	Supersedes            string   `json:"supersedes"`
	Tags                  []string `json:"tags"`
	SessionID             string   `json:"sessionId"`
	SessionIDSnake        string   `json:"session_id"`
	ObservedOn            string   `json:"observedOn"`
	ObservedOnSnake       string   `json:"observed_on"`
	SessionDate           string   `json:"sessionDate"`
	SessionDateSnake      string   `json:"session_date"`
	ContextPrefix         string   `json:"contextPrefix"`
	ContextPrefixSnake    string   `json:"context_prefix"`
	ModifiedOverride      string   `json:"modifiedOverride"`
	ModifiedOverrideSnake string   `json:"modified_override"`
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

	existingMemories, _ := listExistingMemories(ctx, e.mem, projectPath)
	userPrompt := extractUserPrompt(recent, existingMemories)

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
	if !result.Parsed {
		slog.Warn("memory: extract response parse failed")
	}
	result.Memories = postProcessSessionExtractions(
		recent,
		appendHeuristicExtractions(recent, result.Memories, "", ""),
		"",
		"",
	)
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
		return
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
	return ExtractFromMessagesWithSession(ctx, provider, model, mem, projectPath, messages, "", "")
}

// ExtractFromMessagesWithSession runs the extraction LLM call and
// threads explicit session metadata through normalisation so preview
// handlers get the same observed/date shaping as persisted extraction.
func ExtractFromMessagesWithSession(
	ctx context.Context,
	provider llm.Provider,
	model string,
	mem *Memory,
	projectPath string,
	messages []Message,
	sessionID string,
	sessionDate string,
) ([]ExtractedMemory, error) {
	if len(messages) < 2 {
		return nil, nil
	}

	recent := messages
	if len(recent) > extractMaxRecent {
		recent = recent[len(recent)-extractMaxRecent:]
	}

	existingMemories, _ := listExistingMemories(ctx, mem, projectPath)
	userPrompt := extractUserPrompt(recent, existingMemories)

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
	if !result.Parsed {
		result = extractionResult{}
	}
	for i := range result.Memories {
		if strings.TrimSpace(sessionID) != "" && result.Memories[i].SessionID == "" {
			result.Memories[i].SessionID = strings.TrimSpace(sessionID)
		}
		if strings.TrimSpace(sessionDate) != "" && result.Memories[i].SessionDate == "" {
			result.Memories[i].SessionDate = strings.TrimSpace(sessionDate)
		}
	}
	return postProcessSessionExtractions(
		recent,
		appendHeuristicExtractions(recent, result.Memories, sessionID, sessionDate),
		sessionID,
		sessionDate,
	), nil
}

func listExistingMemories(ctx context.Context, mem *Memory, projectPath string) ([]existingMemorySummary, error) {
	projectTopics, err := mem.ListProjectTopics(ctx, projectPath)
	if err != nil {
		return nil, err
	}
	globalTopics, err := mem.ListGlobalTopics(ctx)
	if err != nil {
		return nil, err
	}

	topics := make([]TopicFile, 0, len(projectTopics)+len(globalTopics))
	topics = append(topics, projectTopics...)
	topics = append(topics, globalTopics...)

	summaries := make([]existingMemorySummary, 0, len(topics))
	for _, topic := range topics {
		raw, err := mem.ReadTopic(ctx, topic.Path)
		if err != nil {
			continue
		}
		fm, body := ParseFrontmatter(raw)
		summaries = append(summaries, existingMemorySummary{
			Path:        string(topic.Path),
			Scope:       topic.Scope,
			Name:        strings.TrimSpace(fm.Name),
			Description: strings.TrimSpace(fm.Description),
			Type:        strings.TrimSpace(fm.Type),
			Modified:    strings.TrimSpace(topic.Modified),
			Content:     truncatePromptContent(body),
		})
	}

	sort.Slice(summaries, func(i, j int) bool {
		left := memorySummaryTimestamp(summaries[i])
		right := memorySummaryTimestamp(summaries[j])
		if left != right {
			return left.After(right)
		}
		return summaries[i].Path < summaries[j].Path
	})

	if len(summaries) > existingMemoryLimit {
		summaries = summaries[:existingMemoryLimit]
	}

	return summaries, nil
}

func truncatePromptContent(content string) string {
	collapsed := strings.Join(strings.Fields(content), " ")
	if len(collapsed) <= existingMemoryPreviewLimit {
		return collapsed
	}
	return collapsed[:existingMemoryPreviewLimit] + "..."
}

func memorySummaryTimestamp(summary existingMemorySummary) time.Time {
	if summary.Modified == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339} {
		if parsed, err := time.Parse(layout, summary.Modified); err == nil {
			return parsed
		}
	}
	return time.Time{}
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

func postProcessSessionExtractions(messages []Message, extracted []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	if len(extracted) == 0 {
		return extracted
	}

	sessionRaw, anchor, hasAnchor := resolveHeuristicSessionMetadata(messages, sessionDate)
	sessionID = resolveHeuristicSessionID(messages, extracted, sessionID)
	modifiedOverride := ""
	sessionDateISO := ""
	datePrefix := ""
	if hasAnchor {
		modifiedOverride = anchor.UTC().Format(time.RFC3339)
		sessionDateISO = anchor.UTC().Format("2006-01-02")
		datePrefix = buildInlineDateTokens(modifiedOverride)
	}

	out := make([]ExtractedMemory, 0, len(extracted))
	for _, memory := range extracted {
		shaped := shapeExtractedMemory(memory)
		content := strings.TrimSpace(shaped.Content)
		if sessionRaw != "" && content != "" && !strings.HasPrefix(content, "[Date:") {
			content = datePrefix + "[Observed on " + sessionRaw + "]\n\n" + content
		}
		shaped.Content = content
		if shaped.ModifiedOverride == "" {
			shaped.ModifiedOverride = modifiedOverride
		}
		if shaped.ObservedOn == "" {
			shaped.ObservedOn = modifiedOverride
		}
		if sessionDateISO != "" {
			shaped.SessionDate = sessionDateISO
		}
		if shaped.SessionID == "" {
			shaped.SessionID = sessionID
		}
		shaped.Filename = RewriteHeuristicFilenameForSession(shaped.Filename, shaped.SessionID)
		shaped.Tags = mergeTags(shaped.Tags, autoFactTags(shaped.Content))
		out = append(out, shaped)
	}

	return out
}

func shapeExtractedMemory(memory ExtractedMemory) ExtractedMemory {
	if strings.TrimSpace(memory.Content) == "" {
		return memory
	}
	summary := inferSearchableSummary(memory.Content)
	if summary == "" {
		return memory
	}
	memory.Description = chooseMoreSpecificSummary(memory.Description, summary)
	memory.IndexEntry = chooseMoreSpecificIndexEntry(memory.IndexEntry, summary)
	return memory
}

func chooseMoreSpecificSummary(current, derived string) string {
	cleanedCurrent := strings.TrimSpace(current)
	cleanedDerived := truncateOneLine(derived, 140)
	if cleanedCurrent == "" {
		return cleanedDerived
	}
	if !isLessSpecificSummary(cleanedCurrent, derived) {
		return cleanedCurrent
	}
	return cleanedDerived
}

func chooseMoreSpecificIndexEntry(current, derived string) string {
	cleanedCurrent := strings.TrimSpace(current)
	cleanedDerived := truncateOneLine(derived, 140)
	if cleanedCurrent == "" {
		return cleanedDerived
	}
	if !isLessSpecificSummary(cleanedCurrent, derived) {
		return cleanedCurrent
	}
	if colonIndex := strings.Index(cleanedCurrent, ":"); strings.HasPrefix(cleanedCurrent, "-") && colonIndex > 0 {
		return truncateOneLine(cleanedCurrent[:colonIndex+1]+" "+stripTrailingFullStop(derived), 140)
	}
	return cleanedDerived
}

func isLessSpecificSummary(current, derived string) bool {
	currentTokens := informativeSummaryTokens(current)
	derivedTokens := informativeSummaryTokens(derived)
	if len(derivedTokens) == 0 {
		return false
	}
	if len(currentTokens) == 0 {
		return true
	}
	missing := 0
	for token := range derivedTokens {
		if !currentTokens[token] {
			missing++
		}
	}
	threshold := 2
	if half := (len(derivedTokens) + 1) / 2; half > threshold {
		threshold = half
	}
	return missing >= threshold
}

func informativeSummaryTokens(value string) map[string]bool {
	out := make(map[string]bool)
	for _, token := range summaryTokenRe.FindAllString(strings.ToLower(value), -1) {
		if summaryStopwords[token] {
			continue
		}
		out[token] = true
	}
	return out
}

func mergeTags(existing, inferred []string) []string {
	if len(existing) == 0 && len(inferred) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(existing)+len(inferred))
	out := make([]string, 0, len(existing)+len(inferred))
	for _, raw := range existing {
		tag := strings.TrimSpace(raw)
		if tag == "" || seen[tag] {
			continue
		}
		seen[tag] = true
		out = append(out, tag)
	}
	for _, raw := range inferred {
		tag := strings.TrimSpace(raw)
		if tag == "" || seen[tag] {
			continue
		}
		seen[tag] = true
		out = append(out, tag)
	}
	return out
}

func autoFactTags(content string) []string {
	if content == "" {
		return nil
	}
	body := content
	if len(body) > 4096 {
		body = body[:4096]
	}

	seen := make(map[string]bool)
	add := func(value string) {
		tag := strings.TrimSpace(value)
		if tag == "" || seen[tag] {
			return
		}
		seen[tag] = true
	}

	for _, match := range heuristicDateTagRe.FindAllString(body, -1) {
		add(match)
		if parsed, ok := parseDateInput(strings.ReplaceAll(match, "-", "/")); ok {
			add(parsed.Weekday().String())
			add(parsed.Month().String())
		}
	}
	for _, match := range autoTagWeekdayRe.FindAllString(body, -1) {
		lower := strings.ToLower(match)
		if lower == "" {
			continue
		}
		add(strings.ToUpper(lower[:1]) + lower[1:])
	}
	for _, match := range autoTagRelativeTemporalRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	for _, match := range autoTagClockTimeRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	for _, match := range autoTagMoneyRe.FindAllString(body, -1) {
		add(match)
	}
	for _, captured := range heuristicUnitQuantityRe.FindAllStringSubmatch(body, -1) {
		if len(captured) < 3 {
			continue
		}
		add(strings.TrimSpace(captured[1] + " " + captured[2]))
	}
	for _, match := range autoTagQuantityRe.FindAllString(body, -1) {
		add(match)
	}
	for _, match := range autoTagProperNounRe.FindAllString(body, -1) {
		if len(match) < 3 {
			continue
		}
		if autoTagStopNouns[strings.ToLower(match)] {
			continue
		}
		add(match)
	}
	for _, match := range autoTagPendingActionRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	for _, match := range autoTagMedicalRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	for _, match := range autoTagEventRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	for _, match := range autoTagEntertainmentRe.FindAllString(body, -1) {
		add(strings.ToLower(match))
	}
	if heuristicPendingActionLeadRe.MatchString(body) || autoTagPendingActionRe.MatchString(body) {
		add("pending")
		add("task")
	}
	if heuristicAppointmentRe.MatchString(body) {
		add("appointment")
	}
	if heuristicMedicalEntityRe.MatchString(body) || autoTagExtraMedicalEntityRe.MatchString(body) {
		add("medical")
	}
	if heuristicEventRe.MatchString(body) {
		add("event")
	}
	if heuristicRecommendationRequestRe.MatchString(body) {
		add("recommendation")
	}
	if autoTagEntertainmentRe.MatchString(body) {
		add("entertainment")
	}

	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for tag := range seen {
		out = append(out, tag)
	}
	sort.Strings(out)
	return out
}

func appendHeuristicExtractions(messages []Message, extracted []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	out := make([]ExtractedMemory, 0, len(extracted)+8)
	out = append(out, extracted...)
	out = append(out, deriveHeuristicUserFacts(messages, out, sessionID, sessionDate)...)
	out = append(out, deriveHeuristicMilestoneFacts(messages, out, sessionID, sessionDate)...)
	out = append(out, deriveHeuristicPendingFacts(messages, out, sessionID, sessionDate)...)
	out = append(out, deriveHeuristicEventFacts(messages, out, sessionID, sessionDate)...)
	out = append(out, deriveHeuristicAssistantTableFacts(messages, out, sessionID, sessionDate)...)
	out = append(out, deriveHeuristicPreferenceFacts(messages, out, sessionID, sessionDate)...)
	return out
}

func deriveHeuristicUserFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages, sessionDate)
	out := make([]ExtractedMemory, 0, heuristicUserFactLimit)

	for _, message := range messages {
		if message.Role != RoleUser {
			continue
		}
		for _, sentence := range heuristicUserFactCandidates(message.Content) {
			canonical := strings.ToLower(strings.TrimSpace(sentence))
			if canonical == "" || !firstPersonFactRe.MatchString(sentence) || !hasHeuristicUserFactSignal(sentence) || seen[canonical] {
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
				Filename:    buildHeuristicSessionFilename("user-fact", iso, sessionID, slug),
				Name:        "User Fact: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(sentence, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(sentence, observedOn),
				IndexEntry:  truncateOneLine(sentence, 140),
				ObservedOn:  observedOn,
				SessionID:   strings.TrimSpace(sessionID),
				SessionDate: strings.TrimSpace(sessionDate),
			})
			seen[canonical] = true
			if len(out) >= heuristicUserFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicMilestoneFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages, sessionDate)
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
				Filename:    buildHeuristicSessionFilename("user-fact", iso, sessionID, "milestone-"+slug),
				Name:        "User Fact: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(sentence, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(sentence, observedOn),
				IndexEntry:  truncateOneLine(sentence, 140),
				ObservedOn:  observedOn,
				SessionID:   strings.TrimSpace(sessionID),
				SessionDate: strings.TrimSpace(sessionDate),
			})
			seen[canonical] = true
			if len(out) >= heuristicMilestoneFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicEventFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages, sessionDate)
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
				Filename:    buildHeuristicSessionFilename("user-fact", iso, sessionID, "event-"+slug),
				Name:        "User Event: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(summary, 140),
				Type:        "user",
				Scope:       "global",
				Content:     withObservedDatePrefix(summary+"\n\nEvidence: "+sentence, observedOn),
				IndexEntry:  truncateOneLine(summary, 140),
				ObservedOn:  observedOn,
				SessionID:   strings.TrimSpace(sessionID),
				SessionDate: strings.TrimSpace(sessionDate),
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

func deriveHeuristicPreferenceFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
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
	iso := deriveHeuristicISODate(messages, sessionDate)
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
				Filename:    buildHeuristicSessionFilename("user-preference", iso, sessionID, slug),
				Name:        "User Preference: " + toTitleCase(strings.ReplaceAll(slug, "-", " ")),
				Description: truncateOneLine(candidate.summary, 140),
				Type:        "user",
				Scope:       "global",
				Content:     buildHeuristicPreferenceContent(candidate),
				IndexEntry:  truncateOneLine(candidate.summary, 140),
				SessionID:   strings.TrimSpace(sessionID),
				SessionDate: strings.TrimSpace(sessionDate),
			})
			seen[canonical] = true
			if len(out) >= heuristicPreferenceFactLimit {
				return out
			}
		}
	}

	return out
}

func deriveHeuristicPendingFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
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
					Filename:    buildHeuristicSessionFilename("user-fact", iso, sessionID, "task-"+slug),
					Name:        "User Task: " + toTitleCase(action),
					Description: truncateOneLine(summary, 140),
					Type:        "user",
					Scope:       "global",
					Content:     summary + "\n\nEvidence: " + sentence,
					IndexEntry:  truncateOneLine(summary, 140),
					SessionID:   strings.TrimSpace(sessionID),
					SessionDate: strings.TrimSpace(sessionDate),
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

type heuristicAssistantTableCandidate struct {
	summary string
}

func deriveHeuristicAssistantTableFacts(messages []Message, existing []ExtractedMemory, sessionID, sessionDate string) []ExtractedMemory {
	seen := buildExistingMemoryTextSet(existing)
	iso := deriveHeuristicISODate(messages, sessionDate)
	anchor, hasAnchor := deriveHeuristicSessionAnchor(messages, sessionDate)
	out := make([]ExtractedMemory, 0, heuristicAssistantFactLimit)

	for _, message := range messages {
		if message.Role != RoleAssistant {
			continue
		}
		for _, candidate := range buildHeuristicAssistantTableCandidates(message.Content) {
			canonical := normaliseMemoryText(candidate.summary)
			if canonical == "" || seen[canonical] {
				continue
			}
			slug := heuristicFactSlug(candidate.summary)
			if slug == "" {
				continue
			}
			observedOn := ""
			if hasAnchor {
				observedOn = anchor.UTC().Format(time.RFC3339)
			}
			out = append(out, ExtractedMemory{
				Action:      "create",
				Filename:    buildHeuristicSessionFilename("assistant-table", iso, sessionID, slug),
				Name:        "Assistant Table Row: " + strings.TrimSpace(strings.SplitN(candidate.summary, " ", 2)[0]),
				Description: truncateOneLine(candidate.summary, 140),
				Type:        "project",
				Scope:       "project",
				Content:     withObservedDatePrefix(candidate.summary, observedOn),
				IndexEntry:  truncateOneLine(candidate.summary, 140),
				ObservedOn:  observedOn,
				SessionID:   strings.TrimSpace(sessionID),
				SessionDate: strings.TrimSpace(sessionDate),
			})
			seen[canonical] = true
			if len(out) >= heuristicAssistantFactLimit {
				return out
			}
		}
	}

	return out
}

func buildHeuristicAssistantTableCandidates(content string) []heuristicAssistantTableCandidate {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	out := make([]heuristicAssistantTableCandidate, 0)
	for start := 0; start < len(lines); {
		if !heuristicMarkdownTableLineRe.MatchString(lines[start]) {
			start++
			continue
		}
		end := start
		for end < len(lines) && heuristicMarkdownTableLineRe.MatchString(lines[end]) {
			end++
		}
		out = append(out, parseHeuristicMarkdownTableBlock(lines[start:end])...)
		start = end
	}
	return out
}

func parseHeuristicMarkdownTableBlock(lines []string) []heuristicAssistantTableCandidate {
	if len(lines) < 3 {
		return nil
	}
	headers := parseHeuristicMarkdownTableCells(lines[0])
	separator := parseHeuristicMarkdownTableCells(lines[1])
	if len(headers) < 2 || len(headers) != len(separator) || !isHeuristicMarkdownSeparatorRow(separator) {
		return nil
	}

	out := make([]heuristicAssistantTableCandidate, 0, len(lines)-2)
	for _, line := range lines[2:] {
		cells := parseHeuristicMarkdownTableCells(line)
		if len(cells) == 0 {
			continue
		}
		if len(cells) < len(headers) {
			padded := make([]string, len(headers))
			copy(padded, cells)
			cells = padded
		}
		rowLabel := strings.TrimSpace(cells[0])
		if !heuristicWeekdayCellRe.MatchString(rowLabel) {
			continue
		}
		summary := buildHeuristicAssistantTableSummary(rowLabel, headers, cells)
		if summary == "" {
			continue
		}
		out = append(out, heuristicAssistantTableCandidate{
			summary: summary,
		})
	}
	return out
}

func parseHeuristicMarkdownTableCells(line string) []string {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return nil
	}
	trimmed = strings.TrimPrefix(trimmed, "|")
	trimmed = strings.TrimSuffix(trimmed, "|")
	parts := strings.Split(trimmed, "|")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		out = append(out, strings.TrimSpace(part))
	}
	return out
}

func isHeuristicMarkdownSeparatorRow(cells []string) bool {
	if len(cells) < 2 {
		return false
	}
	found := false
	for _, cell := range cells {
		trimmed := strings.TrimSpace(cell)
		if trimmed == "" {
			continue
		}
		if !heuristicMarkdownSeparatorCellRe.MatchString(trimmed) {
			return false
		}
		found = true
	}
	return found
}

func buildHeuristicAssistantTableSummary(weekday string, headers, cells []string) string {
	if strings.TrimSpace(weekday) == "" {
		return ""
	}

	assignments := make([]string, 0, len(headers)-1)
	hasShiftLabel := false
	for idx := 1; idx < len(headers) && idx < len(cells); idx++ {
		header := strings.Join(strings.Fields(strings.TrimSpace(headers[idx])), " ")
		value := strings.Join(strings.Fields(strings.TrimSpace(cells[idx])), " ")
		if value == "" || value == "-" {
			continue
		}
		formatted := formatHeuristicAssistantTableCell(header, value)
		if formatted == "" {
			continue
		}
		if heuristicAssistantShiftHeaderRe.MatchString(header) || heuristicAssistantShiftValueRe.MatchString(value) {
			hasShiftLabel = true
		}
		assignments = append(assignments, formatted)
	}
	if len(assignments) == 0 {
		return ""
	}

	noun := "row"
	if hasShiftLabel {
		noun = "roster"
	}
	return fmt.Sprintf("%s %s: %s.", toTitleCase(strings.ToLower(strings.TrimSpace(weekday))), noun, strings.Join(assignments, "; "))
}

func formatHeuristicAssistantTableCell(header, value string) string {
	header = strings.TrimSpace(header)
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if header == "" {
		return value
	}
	if heuristicAssistantShiftHeaderRe.MatchString(header) {
		return fmt.Sprintf("%s (%s)", value, header)
	}
	return fmt.Sprintf("%s: %s", header, value)
}

func splitIntoFactSentences(content string) []string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	content = protectSentenceSplitAbbreviations(content)

	var (
		out     []string
		current strings.Builder
	)
	flush := func() {
		part := strings.TrimSpace(current.String())
		if part != "" {
			out = append(out, restoreSentenceSplitAbbreviations(part))
		}
		current.Reset()
	}

	for i, r := range content {
		switch r {
		case '\n':
			flush()
			continue
		case ' ', '\t':
			current.WriteRune(r)
			if current.Len() == 0 {
				continue
			}
			prev := rune(0)
			trimmed := strings.TrimSpace(current.String())
			if trimmed != "" {
				prev = rune(trimmed[len(trimmed)-1])
			}
			if (prev == '.' || prev == '!' || prev == '?') && i+1 < len(content) {
				flush()
			}
			continue
		}
		current.WriteRune(r)
	}
	flush()
	return out
}

func protectSentenceSplitAbbreviations(content string) string {
	protected := content
	for _, replacement := range heuristicSentenceSplitProtections {
		protected = replacement.pattern.ReplaceAllString(protected, replacement.placeholder)
	}
	return protected
}

func restoreSentenceSplitAbbreviations(content string) string {
	restored := content
	for _, replacement := range heuristicSentenceSplitProtections {
		restored = strings.ReplaceAll(restored, replacement.placeholder, replacement.original)
	}
	return restored
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

func hasHeuristicUserFactSignal(sentence string) bool {
	return hasQuantifiedFact(sentence) ||
		heuristicCadenceFactRe.MatchString(sentence) ||
		heuristicStorageLocationFactRe.MatchString(sentence)
}

func heuristicUserFactCandidates(content string) []string {
	candidates := splitIntoFactSentences(content)
	trimmed := strings.TrimSpace(content)
	if trimmed == "" || (!heuristicCadenceFactRe.MatchString(trimmed) && !heuristicStorageLocationFactRe.MatchString(trimmed)) {
		return candidates
	}
	canonical := strings.ToLower(trimmed)
	for _, candidate := range candidates {
		if strings.ToLower(strings.TrimSpace(candidate)) == canonical {
			return candidates
		}
	}
	return append(candidates, trimmed)
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

func buildHeuristicSessionFilename(prefix, iso, sessionID, slug string) string {
	parts := []string{prefix}
	if strings.TrimSpace(iso) != "" {
		parts = append(parts, strings.TrimSpace(iso))
	}
	if sessionSegment := sanitiseHeuristicFileSegment(sessionID); sessionSegment != "" {
		parts = append(parts, sessionSegment)
	}
	parts = append(parts, slug)
	return strings.Join(parts, "-") + ".md"
}

// RewriteHeuristicFilenameForSession mirrors the TypeScript extractor's
// replay filename shape by inserting the session id into heuristic
// user-fact and user-preference filenames. This avoids distinct replay
// sessions overwriting each other when a derived slug and date match.
func RewriteHeuristicFilenameForSession(filename, sessionID string) string {
	sessionSegment := sanitiseHeuristicFileSegment(sessionID)
	if sessionSegment == "" {
		return filename
	}

	if matched := heuristicFilenameDatedRe.FindStringSubmatch(filename); len(matched) == 4 {
		rest := matched[3]
		if rest == sessionSegment || strings.HasPrefix(rest, sessionSegment+"-") {
			return filename
		}
		return fmt.Sprintf("%s-%s-%s-%s.md", matched[1], matched[2], sessionSegment, rest)
	}

	if matched := heuristicFilenameRe.FindStringSubmatch(filename); len(matched) == 3 {
		rest := matched[2]
		if rest == sessionSegment || strings.HasPrefix(rest, sessionSegment+"-") {
			return filename
		}
		return fmt.Sprintf("%s-%s-%s.md", matched[1], sessionSegment, rest)
	}

	return filename
}

func sanitiseHeuristicFileSegment(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	trimmed = heuristicFileSegmentUnsafeRe.ReplaceAllString(trimmed, "-")
	trimmed = heuristicFileSegmentDashRe.ReplaceAllString(trimmed, "-")
	return strings.Trim(trimmed, "-")
}

func deriveHeuristicISODate(messages []Message, sessionDate string) string {
	if anchor, ok := deriveHeuristicSessionAnchor(messages, sessionDate); ok {
		return anchor.UTC().Format("2006-01-02")
	}
	return ""
}

func deriveHeuristicSessionAnchor(messages []Message, sessionDate string) (time.Time, bool) {
	_, parsed, ok := resolveHeuristicSessionMetadata(messages, sessionDate)
	return parsed, ok
}

func resolveHeuristicSessionID(messages []Message, extracted []ExtractedMemory, sessionID string) string {
	if strings.TrimSpace(sessionID) != "" {
		return strings.TrimSpace(sessionID)
	}
	for _, memory := range extracted {
		sessionID := strings.TrimSpace(memory.SessionID)
		if sessionID != "" {
			return sessionID
		}
	}
	for _, message := range messages {
		if message.Role != RoleSystem {
			continue
		}
		match := heuristicSessionIDRe.FindStringSubmatch(message.Content)
		if len(match) != 2 {
			continue
		}
		sessionID := strings.TrimSpace(match[1])
		if sessionID != "" {
			return sessionID
		}
	}
	return ""
}

func resolveHeuristicSessionMetadata(messages []Message, sessionDate string) (string, time.Time, bool) {
	trimmed := strings.TrimSpace(sessionDate)
	if trimmed != "" {
		if parsed, ok := parseDateInput(trimmed); ok {
			return trimmed, parsed.UTC(), true
		}
	}
	return deriveHeuristicSessionMetadata(messages)
}

func deriveHeuristicSessionMetadata(messages []Message) (string, time.Time, bool) {
	for _, message := range messages {
		if message.Role != RoleSystem {
			continue
		}
		match := heuristicSessionDateRe.FindString(message.Content)
		if match == "" {
			continue
		}
		if parsed, ok := parseDateInput(match); ok {
			return match, parsed.UTC(), true
		}
	}
	return "", time.Time{}, false
}

func parseDateInput(value string) (time.Time, bool) {
	for _, layout := range []string{
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02 15:04",
		"2006-01-02",
		time.RFC3339,
	} {
		if parsed, err := time.Parse(layout, strings.TrimSpace(value)); err == nil {
			return parsed.UTC(), true
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
	memories, ok := parseRawExtractionMemories(content)
	if !ok {
		return extractionResult{}
	}
	out := make([]ExtractedMemory, 0, len(memories))
	for _, memory := range memories {
		out = append(out, normaliseExtractedMemory(memory))
	}
	return extractionResult{Memories: out, Parsed: true}
}

func parseRawExtractionMemories(content string) ([]rawExtractedMemory, bool) {
	for _, candidate := range extractionJSONCandidates(content) {
		if memories, ok := decodeRawExtractionCandidate(candidate); ok {
			return memories, true
		}
		repaired := repairExtractionJSONCandidate(candidate)
		if repaired == candidate {
			continue
		}
		if memories, ok := decodeRawExtractionCandidate(repaired); ok {
			return memories, true
		}
	}
	return nil, false
}

func extractionJSONCandidates(content string) []string {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil
	}
	out := make([]string, 0, 3)
	seen := map[string]bool{}
	add := func(candidate string) {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" || seen[candidate] {
			return
		}
		seen[candidate] = true
		out = append(out, candidate)
	}
	add(trimmed)
	add(bracketSlice(trimmed, '{', '}'))
	add(bracketSlice(trimmed, '[', ']'))
	return out
}

func bracketSlice(content string, open, close byte) string {
	start := strings.IndexByte(content, open)
	if start < 0 {
		return ""
	}
	end := strings.LastIndexByte(content, close)
	if end <= start {
		return ""
	}
	return content[start : end+1]
}

func repairExtractionJSONCandidate(content string) string {
	repaired := content
	for {
		next := extractionTrailingCommaRe.ReplaceAllString(repaired, "$1")
		if next == repaired {
			return repaired
		}
		repaired = next
	}
}

func decodeRawExtractionCandidate(content string) ([]rawExtractedMemory, bool) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, false
	}
	if strings.HasPrefix(trimmed, "[") {
		var list []rawExtractedMemory
		if err := json.Unmarshal([]byte(trimmed), &list); err != nil {
			return nil, false
		}
		return list, true
	}
	if !strings.HasPrefix(trimmed, "{") {
		return nil, false
	}

	var envelope struct {
		Memories []rawExtractedMemory `json:"memories"`
		Memory   rawExtractedMemory   `json:"memory"`
	}
	if err := json.Unmarshal([]byte(trimmed), &envelope); err == nil {
		if len(envelope.Memories) > 0 {
			return envelope.Memories, true
		}
		if looksLikeRawExtractedMemory(envelope.Memory) {
			return []rawExtractedMemory{envelope.Memory}, true
		}
	}

	var single rawExtractedMemory
	if err := json.Unmarshal([]byte(trimmed), &single); err != nil {
		return nil, false
	}
	if !looksLikeRawExtractedMemory(single) {
		return nil, false
	}
	return []rawExtractedMemory{single}, true
}

func looksLikeRawExtractedMemory(raw rawExtractedMemory) bool {
	return strings.TrimSpace(raw.Filename) != "" ||
		strings.TrimSpace(raw.Content) != "" ||
		strings.TrimSpace(raw.Name) != "" ||
		strings.TrimSpace(raw.Description) != "" ||
		strings.TrimSpace(raw.IndexEntry) != "" ||
		strings.TrimSpace(raw.IndexEntrySnake) != ""
}

func normaliseExtractedMemory(raw rawExtractedMemory) ExtractedMemory {
	action := "create"
	if raw.Action == "create" || raw.Action == "update" {
		action = raw.Action
	}

	memoryType := "project"
	switch raw.Type {
	case "user", "feedback", "project", "reference":
		memoryType = raw.Type
	}

	scope := "project"
	switch raw.Scope {
	case "global", "project":
		scope = raw.Scope
	}

	tags := make([]string, 0, len(raw.Tags))
	for _, tag := range raw.Tags {
		tag = strings.TrimSpace(tag)
		if tag != "" {
			tags = append(tags, tag)
		}
	}

	indexEntry := raw.IndexEntrySnake
	if strings.TrimSpace(raw.IndexEntry) != "" {
		indexEntry = raw.IndexEntry
	}

	sessionID := raw.SessionIDSnake
	if strings.TrimSpace(raw.SessionID) != "" {
		sessionID = raw.SessionID
	}

	observedOn := raw.ObservedOnSnake
	if strings.TrimSpace(raw.ObservedOn) != "" {
		observedOn = raw.ObservedOn
	}

	sessionDate := raw.SessionDateSnake
	if strings.TrimSpace(raw.SessionDate) != "" {
		sessionDate = raw.SessionDate
	}

	contextPrefix := raw.ContextPrefixSnake
	if strings.TrimSpace(raw.ContextPrefix) != "" {
		contextPrefix = raw.ContextPrefix
	}

	modifiedOverride := raw.ModifiedOverrideSnake
	if strings.TrimSpace(raw.ModifiedOverride) != "" {
		modifiedOverride = raw.ModifiedOverride
	}
	filename := RewriteHeuristicFilenameForSession(raw.Filename, sessionID)

	return ExtractedMemory{
		Action:           action,
		Filename:         filename,
		Name:             raw.Name,
		Description:      raw.Description,
		Type:             memoryType,
		Content:          raw.Content,
		IndexEntry:       indexEntry,
		Scope:            scope,
		Supersedes:       raw.Supersedes,
		Tags:             tags,
		SessionID:        sessionID,
		ObservedOn:       observedOn,
		SessionDate:      sessionDate,
		ContextPrefix:    contextPrefix,
		ModifiedOverride: modifiedOverride,
	}
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
			for _, candidatePath := range supersededCandidatePaths(projectSlug, em.Scope, oldSlug, oldPath) {
				stamped, err := stampSupersededBy(ctx, b, candidatePath, newFile)
				if err != nil {
					return err
				}
				if stamped {
					break
				}
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

func supersededCandidatePaths(projectSlug, scope, oldSlug string, primary brain.Path) []brain.Path {
	paths := []brain.Path{primary}
	switch scope {
	case "global":
		paths = append(paths, brain.MemoryProjectTopic(projectSlug, oldSlug))
	default:
		paths = append(paths, brain.MemoryGlobalTopic(oldSlug))
	}
	return paths
}

// stampSupersededBy rewrites an existing memory file's frontmatter with
// a superseded_by pointer to the new file. It reports whether the
// target file existed and was updated.
func stampSupersededBy(ctx context.Context, b brain.Batch, oldPath brain.Path, newFile string) (bool, error) {
	raw, err := b.Read(ctx, oldPath)
	if err != nil {
		return false, nil
	}
	content := string(raw)
	lines := strings.Split(content, "\n")
	if len(lines) < 2 || strings.TrimSpace(lines[0]) != "---" {
		return false, nil
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return false, nil
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
	if err := b.Write(ctx, oldPath, []byte(strings.Join(lines, "\n"))); err != nil {
		return false, err
	}
	return true, nil
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

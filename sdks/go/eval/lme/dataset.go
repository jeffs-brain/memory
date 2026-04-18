// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Dataset holds a parsed LongMemEval dataset with precomputed metadata.
type Dataset struct {
	Questions  []Question
	SHA256     string
	Categories []string
}

// SessionMessage is a single message within an LME haystack session.
type SessionMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	HasAnswer bool   `json:"has_answer,omitempty"`
}

// Question represents a single LME question with its haystack sessions
// and ground-truth answer.
type Question struct {
	ID               string             `json:"question_id"`
	Category         string             `json:"question_type"`
	Question         string             `json:"question"`
	Answer           string             `json:"answer"`
	QuestionDate     string             `json:"question_date,omitempty"`
	HaystackDates    []string           `json:"haystack_dates,omitempty"`
	SessionIDs       []string           `json:"haystack_session_ids"`
	AnswerSessionIDs []string           `json:"answer_session_ids,omitempty"`
	HaystackSessions [][]SessionMessage `json:"haystack_sessions,omitempty"`
}

// UnmarshalJSON handles the LME dataset's mixed-type answer field,
// which can be a string or a number.
func (q *Question) UnmarshalJSON(data []byte) error {
	type Alias Question
	aux := &struct {
		Answer json.RawMessage `json:"answer"`
		*Alias
	}{
		Alias: (*Alias)(q),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if len(aux.Answer) > 0 {
		var s string
		if err := json.Unmarshal(aux.Answer, &s); err != nil {
			q.Answer = strings.Trim(string(aux.Answer), "\" ")
		} else {
			q.Answer = s
		}
	}
	return nil
}

// HaystackText returns the concatenated text of all haystack sessions
// for this question, suitable for bulk ingest.
func (q Question) HaystackText() string {
	var b strings.Builder
	for i, session := range q.HaystackSessions {
		if i > 0 {
			b.WriteString("\n---\n\n")
		}
		for _, msg := range session {
			fmt.Fprintf(&b, "[%s]: %s\n\n", msg.Role, msg.Content)
		}
	}
	return b.String()
}

// SessionText returns the rendered text of a single haystack session at
// index i, using the same [role]: content shape as HaystackText. Returns
// an empty string if i is out of range.
func (q Question) SessionText(i int) string {
	if i < 0 || i >= len(q.HaystackSessions) {
		return ""
	}
	var b strings.Builder
	for _, msg := range q.HaystackSessions[i] {
		fmt.Fprintf(&b, "[%s]: %s\n\n", msg.Role, msg.Content)
	}
	return b.String()
}

// LoadDataset reads and validates a LongMemEval JSON file.
func LoadDataset(path string) (*Dataset, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("lme: read dataset: %w", err)
	}

	hash := sha256.Sum256(raw)
	sha := hex.EncodeToString(hash[:])

	var questions []Question
	if err := json.Unmarshal(raw, &questions); err != nil {
		return nil, fmt.Errorf("lme: parse dataset: %w", err)
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("lme: dataset contains no questions")
	}

	catSet := make(map[string]bool)
	for i, q := range questions {
		if q.ID == "" {
			return nil, fmt.Errorf("lme: question %d has empty ID", i)
		}
		if q.Category == "" {
			return nil, fmt.Errorf("lme: question %q has empty category", q.ID)
		}
		if q.Question == "" {
			return nil, fmt.Errorf("lme: question %q has empty question text", q.ID)
		}
		if q.Answer == "" {
			return nil, fmt.Errorf("lme: question %q has empty answer", q.ID)
		}
		catSet[q.Category] = true
	}

	cats := make([]string, 0, len(catSet))
	for c := range catSet {
		cats = append(cats, c)
	}
	sort.Strings(cats)

	return &Dataset{
		Questions:  questions,
		SHA256:     sha,
		Categories: cats,
	}, nil
}

// ByCategory returns questions grouped by their category label.
func (d *Dataset) ByCategory() map[string][]Question {
	out := make(map[string][]Question)
	for _, q := range d.Questions {
		out[q.Category] = append(out[q.Category], q)
	}
	return out
}

// Sample returns a deterministic subset of n questions from the dataset.
// The selection is stratified by category to preserve distribution.
func (d *Dataset) Sample(n int, seed int64) []Question {
	if n >= len(d.Questions) {
		return d.Questions
	}

	byCategory := d.ByCategory()
	total := len(d.Questions)

	var result []Question
	remaining := n
	catOrder := make([]string, 0, len(byCategory))
	for c := range byCategory {
		catOrder = append(catOrder, c)
	}
	sort.Strings(catOrder)

	for _, cat := range catOrder {
		qs := byCategory[cat]
		alloc := (len(qs) * n) / total
		if alloc == 0 && remaining > 0 {
			alloc = 1
		}
		if alloc > remaining {
			alloc = remaining
		}
		if alloc > len(qs) {
			alloc = len(qs)
		}

		selected := deterministicSelect(qs, alloc, seed)
		result = append(result, selected...)
		remaining -= len(selected)
	}

	return result
}

// deterministicSelect picks n items from qs using a seed-based permutation.
func deterministicSelect(qs []Question, n int, seed int64) []Question {
	if n >= len(qs) {
		out := make([]Question, len(qs))
		copy(out, qs)
		return out
	}

	indices := make([]int, len(qs))
	for i := range indices {
		indices[i] = i
	}
	state := uint64(seed)
	for i := len(indices) - 1; i > 0; i-- {
		state = state*6364136223846793005 + 1442695040888963407
		j := int(state>>33) % (i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	out := make([]Question, n)
	for i := range n {
		out[i] = qs[indices[i]]
	}
	return out
}

// VerifySHA checks that the dataset's computed SHA256 matches the
// expected value.
func (d *Dataset) VerifySHA(expected string) error {
	if d.SHA256 != expected {
		return fmt.Errorf("lme: dataset SHA mismatch: got %s, want %s", d.SHA256, expected)
	}
	return nil
}

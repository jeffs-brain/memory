// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func makeOutcome(id, category, judge, groundTruth, agent string) QuestionOutcome {
	return QuestionOutcome{
		ID:           id,
		Category:     category,
		Question:     "Q: " + id,
		GroundTruth:  groundTruth,
		AgentAnswer:  agent,
		JudgeVerdict: judge,
	}
}

func writeTempReport(t *testing.T, result *LMEResult) (string, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")
	envelope := map[string]any{
		"generated_at":   "2026-04-15T12:00:00Z",
		"schema_version": 1,
		"lme":            result,
	}
	data, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write report: %v", err)
	}
	return dir, path
}

func TestApplyReviewFilter_IncludeIncorrect(t *testing.T) {
	outcomes := []QuestionOutcome{
		makeOutcome("q1", "a", "correct", "red", "red"),
		makeOutcome("q2", "a", "incorrect", "red", "blue"),
		makeOutcome("q3", "b", "partial", "red", "crimson"),
		makeOutcome("q4", "b", "abstain_incorrect", "n/a", "purple"),
	}

	filter := ReviewFilter{IncludeIncorrect: true}
	got := applyReviewFilter(outcomes, filter)

	if len(got) != 2 {
		t.Fatalf("got %d outcomes, want 2", len(got))
	}
	ids := []string{got[0].ID, got[1].ID}
	if !containsAll(ids, []string{"q2", "q4"}) {
		t.Errorf("unexpected ids %v", ids)
	}
}

func TestApplyReviewFilter_SkipAlreadyReviewed(t *testing.T) {
	outcomes := []QuestionOutcome{
		makeOutcome("q1", "a", "incorrect", "red", "blue"),
		makeOutcome("q2", "a", "incorrect", "red", "blue"),
	}
	outcomes[0].HumanVerdict = "correct"

	got := applyReviewFilter(outcomes, ReviewFilter{IncludeIncorrect: true})
	if len(got) != 1 || got[0].ID != "q2" {
		t.Fatalf("got %+v, want only q2", got)
	}
}

func TestApplyReviewFilter_Disagreement(t *testing.T) {
	outcomes := []QuestionOutcome{
		makeOutcome("q1", "a", "correct", "red", "red"),
		makeOutcome("q2", "a", "correct", "red", "crimson"),
		makeOutcome("q3", "a", "incorrect", "red", "it is red"),
		makeOutcome("q4", "a", "partial", "red", "crimson"),
	}

	got := applyReviewFilter(outcomes, ReviewFilter{IncludeDisagreement: true})
	ids := make([]string, 0, len(got))
	for _, o := range got {
		ids = append(ids, o.ID)
	}
	if !containsAll(ids, []string{"q2", "q3", "q4"}) || len(ids) != 3 {
		t.Fatalf("unexpected ids %v", ids)
	}
}

func TestStratifiedSample_EvenSplit(t *testing.T) {
	outcomes := []QuestionOutcome{}
	for _, cat := range []string{"cat-a", "cat-b", "cat-c"} {
		for i := 0; i < 10; i++ {
			outcomes = append(outcomes, makeOutcome(fmt.Sprintf("%s-%d", cat, i), cat, "incorrect", "x", "y"))
		}
	}

	got := stratifiedSample(outcomes, 6, 42)
	if len(got) != 6 {
		t.Fatalf("got %d samples, want 6", len(got))
	}

	counts := map[string]int{}
	for _, o := range got {
		counts[o.Category]++
	}
	for cat, c := range counts {
		if c != 2 {
			t.Errorf("category %s has %d samples, want 2", cat, c)
		}
	}
}

func TestStratifiedSample_UnevenAllocation(t *testing.T) {
	outcomes := []QuestionOutcome{}
	for _, cat := range []string{"cat-a", "cat-b", "cat-c"} {
		for i := 0; i < 10; i++ {
			outcomes = append(outcomes, makeOutcome(fmt.Sprintf("%s-%d", cat, i), cat, "incorrect", "x", "y"))
		}
	}

	got := stratifiedSample(outcomes, 7, 1)
	if len(got) != 7 {
		t.Fatalf("got %d samples, want 7", len(got))
	}

	counts := map[string]int{}
	for _, o := range got {
		counts[o.Category]++
	}
	if counts["cat-a"] != 3 || counts["cat-b"] != 2 || counts["cat-c"] != 2 {
		t.Errorf("unexpected distribution: %+v", counts)
	}
}

func TestNewReviewSession_FilterIncorrect(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 4,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "a", "correct", "red", "red"),
			makeOutcome("q2", "a", "incorrect", "red", "blue"),
			makeOutcome("q3", "a", "incorrect", "red", "green"),
			makeOutcome("q4", "b", "partial", "yes", "maybe"),
		},
	}
	_, reportPath := writeTempReport(t, result)

	session, err := NewReviewSession(reportPath, ReviewFilter{IncludeIncorrect: true})
	if err != nil {
		t.Fatalf("NewReviewSession: %v", err)
	}
	if len(session.Outcomes) != 2 {
		t.Fatalf("got %d outcomes, want 2", len(session.Outcomes))
	}
}

func scriptedEditor(t *testing.T, verdicts []ArbitrationVerdict, notes []string) EditorFunc {
	t.Helper()
	i := 0
	return func(path string) error {
		if i >= len(verdicts) {
			t.Fatalf("editor invoked %d times, script only has %d", i+1, len(verdicts))
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		body := stripComments(string(data))
		var s scratchpad
		if err := yaml.Unmarshal([]byte(body), &s); err != nil {
			return fmt.Errorf("scripted editor parse: %w", err)
		}
		s.HumanVerdict = string(verdicts[i])
		if i < len(notes) {
			s.Notes = notes[i]
		}
		out, err := yaml.Marshal(s)
		if err != nil {
			return err
		}
		i++
		return os.WriteFile(path, out, 0o644)
	}
}

func stripComments(input string) string {
	scanner := bufio.NewScanner(strings.NewReader(input))
	var out []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func TestReviewSession_Run_AggregatesScore(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 4,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
			makeOutcome("q2", "cat-a", "incorrect", "red", "green"),
			makeOutcome("q3", "cat-b", "incorrect", "yes", "no"),
			makeOutcome("q4", "cat-b", "incorrect", "yes", "maybe"),
		},
	}
	dir, reportPath := writeTempReport(t, result)

	session, err := NewReviewSession(reportPath, ReviewFilter{IncludeIncorrect: true})
	if err != nil {
		t.Fatalf("NewReviewSession: %v", err)
	}
	session.Editor = scriptedEditor(t,
		[]ArbitrationVerdict{HumanCorrect, HumanPartial, HumanIncorrect, HumanCorrect},
		[]string{"good", "halfway", "off", "fine"},
	)

	got, err := session.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got.Reviewed != 4 {
		t.Errorf("Reviewed = %d, want 4", got.Reviewed)
	}

	if math.Abs(got.ArbitratedScore-0.625) > 1e-9 {
		t.Errorf("ArbitratedScore = %v, want 0.625", got.ArbitratedScore)
	}

	if math.Abs(got.ArbitratedCategoryScores["cat-a"]-0.75) > 1e-9 {
		t.Errorf("cat-a score = %v, want 0.75", got.ArbitratedCategoryScores["cat-a"])
	}
	if math.Abs(got.ArbitratedCategoryScores["cat-b"]-0.5) > 1e-9 {
		t.Errorf("cat-b score = %v, want 0.5", got.ArbitratedCategoryScores["cat-b"])
	}

	if math.Abs(got.JudgeAgreement-0.25) > 1e-9 {
		t.Errorf("JudgeAgreement = %v, want 0.25", got.JudgeAgreement)
	}

	jsonlPath := filepath.Join(dir, "arbitration.jsonl")
	lines, err := readJSONL(jsonlPath)
	if err != nil {
		t.Fatalf("read jsonl: %v", err)
	}
	if len(lines) != 4 {
		t.Fatalf("got %d jsonl lines, want 4", len(lines))
	}
}

func TestReviewSession_Run_Skip(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 2,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
			makeOutcome("q2", "cat-a", "incorrect", "red", "green"),
		},
	}
	_, reportPath := writeTempReport(t, result)
	session, err := NewReviewSession(reportPath, ReviewFilter{IncludeIncorrect: true})
	if err != nil {
		t.Fatalf("NewReviewSession: %v", err)
	}
	session.Editor = scriptedEditor(t,
		[]ArbitrationVerdict{HumanSkip, HumanCorrect},
		nil,
	)

	got, err := session.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got.Reviewed != 1 {
		t.Errorf("Reviewed = %d, want 1", got.Reviewed)
	}
	if math.Abs(got.ArbitratedScore-1.0) > 1e-9 {
		t.Errorf("ArbitratedScore = %v, want 1.0", got.ArbitratedScore)
	}
}

func TestReviewSession_Run_InvalidYAMLRetries(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 1,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
		},
	}
	_, reportPath := writeTempReport(t, result)
	session, err := NewReviewSession(reportPath, ReviewFilter{IncludeIncorrect: true})
	if err != nil {
		t.Fatalf("NewReviewSession: %v", err)
	}

	attempts := 0
	session.Editor = func(path string) error {
		attempts++
		bad := `question_id: q1
human_verdict: bogus
`
		return os.WriteFile(path, []byte(bad), 0o644)
	}

	_, runErr := session.Run(context.Background())
	if runErr == nil {
		t.Fatalf("Run: want error after retries, got nil")
	}
	if attempts != maxEditorReopens {
		t.Errorf("editor invoked %d times, want %d", attempts, maxEditorReopens)
	}
}

func TestReviewSession_Run_InvalidThenValid(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 1,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
		},
	}
	_, reportPath := writeTempReport(t, result)
	session, err := NewReviewSession(reportPath, ReviewFilter{IncludeIncorrect: true})
	if err != nil {
		t.Fatalf("NewReviewSession: %v", err)
	}

	attempts := 0
	session.Editor = func(path string) error {
		attempts++
		if attempts == 1 {
			return os.WriteFile(path, []byte("question_id: q1\nhuman_verdict: bogus\n"), 0o644)
		}
		s := scratchpad{QuestionID: "q1", HumanVerdict: string(HumanCorrect)}
		out, _ := yaml.Marshal(s)
		return os.WriteFile(path, out, 0o644)
	}

	got, err := session.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
	if got.Entries[0].HumanVerdict != HumanCorrect {
		t.Errorf("verdict = %v, want correct", got.Entries[0].HumanVerdict)
	}
}

func TestWriteArbitration_AdditivePersist(t *testing.T) {
	result := &LMEResult{
		QuestionsRun: 2,
		OverallScore: 0.5,
		ByCategory: map[string]Category{
			"cat-a": {Run: 1, Incorrect: 1, Score: 0.0},
			"cat-b": {Run: 1, Correct: 1, Score: 1.0},
		},
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
			makeOutcome("q2", "cat-b", "correct", "yes", "yes"),
		},
	}
	dir, reportPath := writeTempReport(t, result)

	arb := ArbitrationResult{
		Entries: []ArbitrationEntry{
			{QuestionID: "q1", Category: "cat-a", JudgeVerdict: "incorrect", HumanVerdict: HumanCorrect},
		},
		ArbitratedScore:          1.0,
		ArbitratedCategoryScores: map[string]float64{"cat-a": 1.0},
		Reviewed:                 1,
	}

	if err := WriteArbitration(dir, arb); err != nil {
		t.Fatalf("WriteArbitration: %v", err)
	}

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var envelope struct {
		LME *LMEResult `json:"lme"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if envelope.LME.OverallScore != 0.5 {
		t.Errorf("OverallScore corrupted: %v", envelope.LME.OverallScore)
	}
	if math.Abs(envelope.LME.ArbitratedScore-1.0) > 1e-9 {
		t.Errorf("ArbitratedScore = %v, want 1.0", envelope.LME.ArbitratedScore)
	}
	if envelope.LME.ArbitratedReviewed != 1 {
		t.Errorf("ArbitratedReviewed = %d, want 1", envelope.LME.ArbitratedReviewed)
	}
	if envelope.LME.ArbitratedCategoryScores["cat-a"] != 1.0 {
		t.Errorf("cat-a arbitrated = %v, want 1.0", envelope.LME.ArbitratedCategoryScores["cat-a"])
	}
	if envelope.LME.ByCategory["cat-a"].Score != 0.0 {
		t.Errorf("original cat-a score mutated: %v", envelope.LME.ByCategory["cat-a"].Score)
	}
	if envelope.LME.Questions[0].HumanVerdict != string(HumanCorrect) {
		t.Errorf("human verdict not mirrored, got %q", envelope.LME.Questions[0].HumanVerdict)
	}
}

func TestResolveReportPath_Latest(t *testing.T) {
	base := t.TempDir()
	runDir := filepath.Join(base, "123456")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "report.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.Symlink("123456", filepath.Join(base, "latest")); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	got, err := ResolveReportPath(base, "latest")
	if err != nil {
		t.Fatalf("ResolveReportPath: %v", err)
	}
	want := filepath.Join(runDir, "report.json")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func readJSONL(path string) ([]ArbitrationEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []ArbitrationEntry
	dec := json.NewDecoder(f)
	for dec.More() {
		var e ArbitrationEntry
		if err := dec.Decode(&e); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}

func containsAll(got, want []string) bool {
	set := map[string]bool{}
	for _, g := range got {
		set[g] = true
	}
	for _, w := range want {
		if !set[w] {
			return false
		}
	}
	return true
}

// ---- Diff (before/after) tests ----

func TestDiffReports_Regressions(t *testing.T) {
	before := &LMEResult{
		QuestionsRun: 3,
		OverallScore: 1.0,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "correct", "red", "red"),
			makeOutcome("q2", "cat-b", "correct", "yes", "yes"),
			makeOutcome("q3", "cat-a", "incorrect", "maybe", "no"),
		},
	}
	after := &LMEResult{
		QuestionsRun: 3,
		OverallScore: 0.333,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "incorrect", "red", "blue"),
			makeOutcome("q2", "cat-b", "correct", "yes", "yes"),
			makeOutcome("q3", "cat-a", "correct", "maybe", "maybe"),
		},
	}

	md, err := DiffReports(before, after)
	if err != nil {
		t.Fatalf("DiffReports: %v", err)
	}

	mustContain := []string{
		"# LME Benchmark Diff",
		"Overall score",
		"Regressions (1)",
		"**q1** (`cat-a`): correct -> incorrect",
		"Improvements (1)",
		"**q3** (`cat-a`): incorrect -> correct",
	}
	for _, s := range mustContain {
		if !strings.Contains(md, s) {
			t.Errorf("diff markdown missing %q; full output:\n%s", s, md)
		}
	}
}

func TestDiffReports_NoQuestionsSide(t *testing.T) {
	before := &LMEResult{
		QuestionsRun: 2,
		OverallScore: 1.0,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "correct", "red", "red"),
			makeOutcome("q2", "cat-b", "correct", "yes", "yes"),
		},
	}
	after := &LMEResult{
		QuestionsRun: 2,
		OverallScore: 1.0,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "cat-a", "correct", "red", "red"),
			makeOutcome("q3", "cat-a", "correct", "blue", "blue"),
		},
	}

	md, err := DiffReports(before, after)
	if err != nil {
		t.Fatalf("DiffReports: %v", err)
	}

	if !strings.Contains(md, "Only in before (1)") {
		t.Error("expected 'Only in before (1)' section header")
	}
	if !strings.Contains(md, "Only in after (1)") {
		t.Error("expected 'Only in after (1)' section header")
	}
	if !strings.Contains(md, "`q2`") {
		t.Error("expected q2 listed as only-in-before")
	}
	if !strings.Contains(md, "`q3`") {
		t.Error("expected q3 listed as only-in-after")
	}
}

func TestDiffReports_RequiresBothReports(t *testing.T) {
	_, err := DiffReports(nil, &LMEResult{})
	if err == nil {
		t.Error("expected error for nil before")
	}
	_, err = DiffReports(&LMEResult{}, nil)
	if err == nil {
		t.Error("expected error for nil after")
	}
}

func TestLoadReport_Envelope(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")
	envelope := map[string]any{
		"lme": &LMEResult{
			QuestionsRun: 1,
			Questions: []QuestionOutcome{
				makeOutcome("q1", "a", "correct", "r", "r"),
			},
		},
	}
	data, _ := json.MarshalIndent(envelope, "", "  ")
	_ = os.WriteFile(path, data, 0o644)

	got, err := LoadReport(path)
	if err != nil {
		t.Fatalf("LoadReport: %v", err)
	}
	if got.QuestionsRun != 1 {
		t.Errorf("QuestionsRun = %d, want 1", got.QuestionsRun)
	}
}

func TestLoadReport_Bare(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")
	bare := &LMEResult{
		QuestionsRun: 2,
		Questions: []QuestionOutcome{
			makeOutcome("q1", "a", "correct", "r", "r"),
			makeOutcome("q2", "a", "incorrect", "r", "b"),
		},
	}
	data, _ := json.MarshalIndent(bare, "", "  ")
	_ = os.WriteFile(path, data, 0o644)

	got, err := LoadReport(path)
	if err != nil {
		t.Fatalf("LoadReport: %v", err)
	}
	if got.QuestionsRun != 2 {
		t.Errorf("QuestionsRun = %d, want 2", got.QuestionsRun)
	}
}

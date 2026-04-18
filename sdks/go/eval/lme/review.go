// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// ArbitrationVerdict is a human-supplied verdict captured during the
// `memory eval lme review` flow. It parallels the JudgeVerdict
// categories but is recorded separately so the judge score is never
// overwritten.
type ArbitrationVerdict string

const (
	HumanCorrect          ArbitrationVerdict = "correct"
	HumanPartial          ArbitrationVerdict = "partial"
	HumanIncorrect        ArbitrationVerdict = "incorrect"
	HumanAbstainCorrect   ArbitrationVerdict = "abstain_correct"
	HumanAbstainIncorrect ArbitrationVerdict = "abstain_incorrect"
	HumanSkip             ArbitrationVerdict = "skip"
)

// validHumanVerdicts enumerates the verdict strings accepted back from
// the editor scratchpad.
var validHumanVerdicts = map[ArbitrationVerdict]bool{
	HumanCorrect:          true,
	HumanPartial:          true,
	HumanIncorrect:        true,
	HumanAbstainCorrect:   true,
	HumanAbstainIncorrect: true,
	HumanSkip:             true,
}

// maxEditorReopens caps the retry loop when the user saves an invalid
// scratchpad.
const maxEditorReopens = 3

// ReviewFilter configures the subset of outcomes surfaced to the human
// reviewer. Each include flag widens the candidate pool; stratified
// sampling kicks in when Stratified is set and SampleSize > 0.
type ReviewFilter struct {
	IncludeIncorrect    bool
	IncludePartial      bool
	IncludeDisagreement bool
	Stratified          bool
	SampleSize          int
	Seed                int64
}

// EditorFunc opens path in an editor and blocks until it closes. The
// indirection keeps the pure-logic tests independent of $EDITOR.
type EditorFunc func(path string) error

// ReviewSession holds the loaded report plus the filtered and sampled
// outcomes.
type ReviewSession struct {
	Report     *LMEResult
	ReportPath string
	Filter     ReviewFilter
	Editor     EditorFunc
	Outcomes   []QuestionOutcome
}

// ArbitrationEntry is the persisted record of a single human verdict.
type ArbitrationEntry struct {
	QuestionID   string             `json:"question_id"`
	Category     string             `json:"category"`
	JudgeVerdict string             `json:"judge_verdict,omitempty"`
	HumanVerdict ArbitrationVerdict `json:"human_verdict"`
	Notes        string             `json:"notes,omitempty"`
	Timestamp    time.Time          `json:"timestamp"`
}

// ArbitrationResult aggregates the outcome of a review session. Scores
// exclude skips from the denominator so noisy questions can be
// deferred without distorting the headline number.
type ArbitrationResult struct {
	Entries                  []ArbitrationEntry
	ArbitratedScore          float64
	ArbitratedCategoryScores map[string]float64
	Reviewed                 int
	JudgeAgreement           float64
}

// scratchpad is the YAML structure the reviewer edits on disk.
type scratchpad struct {
	QuestionID     string `yaml:"question_id"`
	Category       string `yaml:"category"`
	Question       string `yaml:"question"`
	GroundTruth    string `yaml:"ground_truth"`
	AgentAnswer    string `yaml:"agent_answer"`
	JudgeVerdict   string `yaml:"judge_verdict"`
	JudgeRationale string `yaml:"judge_rationale,omitempty"`
	HumanVerdict   string `yaml:"human_verdict"`
	Notes          string `yaml:"notes,omitempty"`
}

// ResolveReportPath turns a CLI identifier into an absolute report.json
// path. "latest" resolves through the evals symlink; "<ts>" is looked
// up under evalsDir; anything else is treated as a direct filesystem
// path to either a report.json file or a directory containing one.
func ResolveReportPath(evalsDir, identifier string) (string, error) {
	switch identifier {
	case "":
		return "", errors.New("empty report identifier")
	case "latest":
		link := filepath.Join(evalsDir, "latest")
		target, err := os.Readlink(link)
		if err != nil {
			return "", fmt.Errorf("resolve latest: %w", err)
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(link), target)
		}
		return filepath.Join(target, "report.json"), nil
	}

	if info, err := os.Stat(identifier); err == nil {
		if info.IsDir() {
			return filepath.Join(identifier, "report.json"), nil
		}
		return identifier, nil
	}

	candidate := filepath.Join(evalsDir, identifier, "report.json")
	if _, err := os.Stat(candidate); err == nil {
		return candidate, nil
	}
	return "", fmt.Errorf("report %q not found (checked direct path and %s)", identifier, candidate)
}

// NewReviewSession loads a report and prepares a filtered slice of
// outcomes. Accepts both an [eval.Report] envelope (`{"lme":{...}}`)
// and a bare [LMEResult].
func NewReviewSession(reportPath string, filter ReviewFilter) (*ReviewSession, error) {
	if reportPath == "" {
		return nil, errors.New("review session: empty report path")
	}

	data, err := os.ReadFile(reportPath)
	if err != nil {
		return nil, fmt.Errorf("read report: %w", err)
	}

	lmeResult, err := extractLMEResult(data)
	if err != nil {
		return nil, err
	}
	if lmeResult == nil || len(lmeResult.Questions) == 0 {
		return nil, errors.New("review session: report has no LME outcomes to review")
	}

	candidates := applyReviewFilter(lmeResult.Questions, filter)
	if filter.Stratified && filter.SampleSize > 0 {
		candidates = stratifiedSample(candidates, filter.SampleSize, filter.Seed)
	} else if filter.SampleSize > 0 && len(candidates) > filter.SampleSize {
		candidates = uniformSample(candidates, filter.SampleSize, filter.Seed)
	}

	return &ReviewSession{
		Report:     lmeResult,
		ReportPath: reportPath,
		Filter:     filter,
		Outcomes:   candidates,
	}, nil
}

// extractLMEResult peels the LMEResult out of either a full
// eval.Report envelope (`{"lme":{...}}`) or a bare LMEResult JSON.
func extractLMEResult(data []byte) (*LMEResult, error) {
	var envelope struct {
		LME *LMEResult `json:"lme"`
	}
	if err := json.Unmarshal(data, &envelope); err == nil && envelope.LME != nil {
		return envelope.LME, nil
	}

	var bare LMEResult
	if err := json.Unmarshal(data, &bare); err != nil {
		return nil, fmt.Errorf("parse report: %w", err)
	}
	if bare.QuestionsRun == 0 && len(bare.Questions) == 0 {
		return nil, errors.New("report payload does not contain LME outcomes")
	}
	return &bare, nil
}

// applyReviewFilter walks outcomes and keeps those matching any
// requested predicate. Already-reviewed outcomes are skipped so
// repeated sessions accumulate verdicts rather than re-asking.
func applyReviewFilter(outcomes []QuestionOutcome, f ReviewFilter) []QuestionOutcome {
	anyFilter := f.IncludeIncorrect || f.IncludePartial || f.IncludeDisagreement
	if !anyFilter {
		return nil
	}

	out := make([]QuestionOutcome, 0, len(outcomes))
	for _, o := range outcomes {
		if o.HumanVerdict != "" {
			continue
		}

		keep := false
		if f.IncludeIncorrect && isIncorrectVerdict(o.JudgeVerdict) {
			keep = true
		}
		if !keep && f.IncludePartial && o.JudgeVerdict == "partial" {
			keep = true
		}
		if !keep && f.IncludeDisagreement && judgeDisagreesWithExact(o) {
			keep = true
		}
		if keep {
			out = append(out, o)
		}
	}
	return out
}

// isIncorrectVerdict normalises the judge verdict space so the filter
// captures the full "looks wrong" set.
func isIncorrectVerdict(v string) bool {
	switch v {
	case "incorrect", "abstain_incorrect", "error":
		return true
	default:
		return false
	}
}

// judgeDisagreesWithExact compares the judge verdict against what
// exact-match would have produced.
func judgeDisagreesWithExact(o QuestionOutcome) bool {
	exact := exactMatch(o.AgentAnswer, o.GroundTruth)
	switch o.JudgeVerdict {
	case "correct":
		return !exact
	case "incorrect":
		return exact
	case "partial":
		return true
	default:
		return false
	}
}

// stratifiedSample returns a sampled slice with questions drawn evenly
// from each category in the input. When the allocation cannot be
// shared exactly, the remainder is distributed across the largest
// categories in alphabetical order so the policy is deterministic.
func stratifiedSample(outcomes []QuestionOutcome, sampleSize int, seed int64) []QuestionOutcome {
	if sampleSize <= 0 || len(outcomes) == 0 {
		return outcomes
	}
	if sampleSize >= len(outcomes) {
		return outcomes
	}

	buckets := map[string][]QuestionOutcome{}
	for _, o := range outcomes {
		buckets[o.Category] = append(buckets[o.Category], o)
	}

	categories := make([]string, 0, len(buckets))
	for cat := range buckets {
		categories = append(categories, cat)
	}
	sort.Strings(categories)

	perCat := sampleSize / len(categories)
	remainder := sampleSize - (perCat * len(categories))

	rng := rand.New(rand.NewSource(seed))

	var out []QuestionOutcome
	for i, cat := range categories {
		bucket := buckets[cat]
		alloc := perCat
		if i < remainder {
			alloc++
		}
		if alloc > len(bucket) {
			alloc = len(bucket)
		}
		if alloc == 0 {
			continue
		}

		shuffled := make([]QuestionOutcome, len(bucket))
		copy(shuffled, bucket)
		rng.Shuffle(len(shuffled), func(a, b int) {
			shuffled[a], shuffled[b] = shuffled[b], shuffled[a]
		})
		out = append(out, shuffled[:alloc]...)
	}

	return out
}

// uniformSample returns a simple random sample of n outcomes.
func uniformSample(outcomes []QuestionOutcome, n int, seed int64) []QuestionOutcome {
	if n >= len(outcomes) {
		return outcomes
	}
	rng := rand.New(rand.NewSource(seed))
	idx := rng.Perm(len(outcomes))
	out := make([]QuestionOutcome, n)
	for i := 0; i < n; i++ {
		out[i] = outcomes[idx[i]]
	}
	return out
}

// Run walks each outcome, writes a YAML scratchpad, invokes the
// editor, parses the reviewer's verdict, and persists the running
// arbitration.jsonl as it goes.
func (r *ReviewSession) Run(ctx context.Context) (ArbitrationResult, error) {
	if r == nil {
		return ArbitrationResult{}, errors.New("nil review session")
	}
	if r.Editor == nil {
		return ArbitrationResult{}, errors.New("review session: editor func not configured")
	}

	reportDir := filepath.Dir(r.ReportPath)
	jsonlPath := filepath.Join(reportDir, "arbitration.jsonl")

	f, err := os.OpenFile(jsonlPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return ArbitrationResult{}, fmt.Errorf("open arbitration.jsonl: %w", err)
	}
	defer f.Close()

	var entries []ArbitrationEntry

	for i, outcome := range r.Outcomes {
		if err := ctx.Err(); err != nil {
			return aggregate(entries, r.Outcomes), err
		}

		entry, err := r.reviewOne(ctx, outcome, i+1, len(r.Outcomes))
		if err != nil {
			return aggregate(entries, r.Outcomes), err
		}
		entries = append(entries, entry)

		line, err := json.Marshal(entry)
		if err != nil {
			return aggregate(entries, r.Outcomes), fmt.Errorf("marshal entry: %w", err)
		}
		if _, err := f.Write(append(line, '\n')); err != nil {
			return aggregate(entries, r.Outcomes), fmt.Errorf("append arbitration.jsonl: %w", err)
		}
		if err := f.Sync(); err != nil {
			return aggregate(entries, r.Outcomes), fmt.Errorf("fsync arbitration.jsonl: %w", err)
		}
	}

	return aggregate(entries, r.Outcomes), nil
}

// reviewOne drives a single scratchpad edit-loop with up to
// maxEditorReopens retries when the reviewer supplies an invalid
// verdict or malformed YAML.
func (r *ReviewSession) reviewOne(_ context.Context, outcome QuestionOutcome, idx, total int) (ArbitrationEntry, error) {
	scratch := scratchpad{
		QuestionID:     outcome.ID,
		Category:       outcome.Category,
		Question:       outcome.Question,
		GroundTruth:    outcome.GroundTruth,
		AgentAnswer:    outcome.AgentAnswer,
		JudgeVerdict:   outcome.JudgeVerdict,
		JudgeRationale: outcome.JudgeRationale,
		HumanVerdict:   "",
	}

	var lastErr error
	for attempt := 0; attempt < maxEditorReopens; attempt++ {
		path, cleanup, err := writeScratchpad(scratch, attempt, idx, total, lastErr)
		if err != nil {
			return ArbitrationEntry{}, err
		}

		if err := r.Editor(path); err != nil {
			cleanup()
			return ArbitrationEntry{}, fmt.Errorf("editor for %s: %w", outcome.ID, err)
		}

		edited, readErr := readScratchpad(path)
		cleanup()
		if readErr != nil {
			lastErr = readErr
			continue
		}

		verdict := ArbitrationVerdict(strings.TrimSpace(edited.HumanVerdict))
		if verdict == "" {
			verdict = HumanSkip
		}
		if !validHumanVerdicts[verdict] {
			lastErr = fmt.Errorf("invalid human_verdict %q (must be one of correct, partial, incorrect, abstain_correct, abstain_incorrect, skip)", edited.HumanVerdict)
			scratch.Notes = edited.Notes
			continue
		}

		return ArbitrationEntry{
			QuestionID:   outcome.ID,
			Category:     outcome.Category,
			JudgeVerdict: outcome.JudgeVerdict,
			HumanVerdict: verdict,
			Notes:        strings.TrimSpace(edited.Notes),
			Timestamp:    time.Now().UTC(),
		}, nil
	}

	return ArbitrationEntry{}, fmt.Errorf("arbitration for %s: gave up after %d attempts: %w", outcome.ID, maxEditorReopens, lastErr)
}

// writeScratchpad renders the YAML scratch file. When lastErr is
// non-nil a header comment surfaces the reason so the reviewer can fix
// their mistake without re-reading the whole file.
func writeScratchpad(s scratchpad, attempt, idx, total int, lastErr error) (string, func(), error) {
	tmp, err := os.CreateTemp("", "memory-lme-review-*.yaml")
	if err != nil {
		return "", func() {}, fmt.Errorf("create scratchpad: %w", err)
	}
	path := tmp.Name()
	cleanup := func() { _ = os.Remove(path) }

	var header strings.Builder
	fmt.Fprintf(&header, "# Question %d of %d, attempt %d\n", idx, total, attempt+1)
	header.WriteString("# Set human_verdict to one of: correct, partial, incorrect, abstain_correct, abstain_incorrect, skip\n")
	header.WriteString("# Leave blank or use 'skip' to defer this question.\n")
	if lastErr != nil {
		fmt.Fprintf(&header, "# ERROR from previous attempt: %s\n", lastErr)
	}
	header.WriteString("\n")

	body, err := yaml.Marshal(s)
	if err != nil {
		tmp.Close()
		cleanup()
		return "", func() {}, fmt.Errorf("marshal scratchpad: %w", err)
	}

	if _, err := tmp.WriteString(header.String()); err != nil {
		tmp.Close()
		cleanup()
		return "", func() {}, fmt.Errorf("write header: %w", err)
	}
	if _, err := tmp.Write(body); err != nil {
		tmp.Close()
		cleanup()
		return "", func() {}, fmt.Errorf("write scratchpad: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("close scratchpad: %w", err)
	}
	return path, cleanup, nil
}

// readScratchpad re-parses the scratchpad after the editor exits.
func readScratchpad(path string) (scratchpad, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return scratchpad{}, fmt.Errorf("read scratchpad: %w", err)
	}
	var s scratchpad
	if err := yaml.Unmarshal(data, &s); err != nil {
		return scratchpad{}, fmt.Errorf("parse scratchpad: %w", err)
	}
	return s, nil
}

// aggregate rolls the recorded entries into an ArbitrationResult.
// Skips are excluded from the denominator.
func aggregate(entries []ArbitrationEntry, reviewed []QuestionOutcome) ArbitrationResult {
	if len(entries) == 0 {
		return ArbitrationResult{Entries: entries, ArbitratedCategoryScores: map[string]float64{}}
	}

	type counter struct {
		scored  int
		correct float64
	}

	var overall counter
	byCat := map[string]*counter{}
	agreements := 0
	agreementDenom := 0

	outcomeByID := map[string]QuestionOutcome{}
	for _, o := range reviewed {
		outcomeByID[o.ID] = o
	}
	_ = outcomeByID

	for _, e := range entries {
		if e.HumanVerdict == HumanSkip {
			continue
		}

		cat := e.Category
		c, ok := byCat[cat]
		if !ok {
			c = &counter{}
			byCat[cat] = c
		}

		weight := scoreWeight(e.HumanVerdict)
		overall.scored++
		overall.correct += weight
		c.scored++
		c.correct += weight

		if j := strings.TrimSpace(e.JudgeVerdict); j != "" {
			agreementDenom++
			if verdictsAgree(j, e.HumanVerdict) {
				agreements++
			}
		}
	}

	arbScore := 0.0
	if overall.scored > 0 {
		arbScore = overall.correct / float64(overall.scored)
	}

	catScores := make(map[string]float64, len(byCat))
	for cat, c := range byCat {
		if c.scored == 0 {
			continue
		}
		catScores[cat] = c.correct / float64(c.scored)
	}

	agreement := 0.0
	if agreementDenom > 0 {
		agreement = float64(agreements) / float64(agreementDenom)
	}

	return ArbitrationResult{
		Entries:                  entries,
		ArbitratedScore:          arbScore,
		ArbitratedCategoryScores: catScores,
		Reviewed:                 overall.scored,
		JudgeAgreement:           agreement,
	}
}

// scoreWeight maps a verdict to its numeric contribution.
func scoreWeight(v ArbitrationVerdict) float64 {
	switch v {
	case HumanCorrect, HumanAbstainCorrect:
		return 1.0
	case HumanPartial:
		return 0.5
	default:
		return 0.0
	}
}

// verdictsAgree decides whether a judge verdict and a human verdict
// end up on the same side of the correct / not-correct line.
func verdictsAgree(judge string, human ArbitrationVerdict) bool {
	if human == HumanPartial {
		return judge == "partial"
	}
	judgeCorrect := judge == "correct" || judge == "abstain_correct"
	humanCorrect := human == HumanCorrect || human == HumanAbstainCorrect
	return judgeCorrect == humanCorrect
}

// WriteArbitration persists the result back into the run directory
// alongside report.json.
func WriteArbitration(reportDir string, result ArbitrationResult) error {
	if reportDir == "" {
		return errors.New("write arbitration: empty report dir")
	}

	reportPath := filepath.Join(reportDir, "report.json")
	data, err := os.ReadFile(reportPath)
	if err != nil {
		return fmt.Errorf("read report.json: %w", err)
	}

	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(data, &envelope); err != nil {
		return fmt.Errorf("parse report.json: %w", err)
	}

	lmeRaw, ok := envelope["lme"]
	if !ok {
		if err := patchBareLMEResult(reportPath, data, result); err != nil {
			return err
		}
		return nil
	}

	var lmeResult LMEResult
	if err := json.Unmarshal(lmeRaw, &lmeResult); err != nil {
		return fmt.Errorf("parse lme block: %w", err)
	}
	applyArbitrationToLME(&lmeResult, result)

	patched, err := json.Marshal(&lmeResult)
	if err != nil {
		return fmt.Errorf("marshal lme block: %w", err)
	}
	envelope["lme"] = patched

	out, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}

	return atomicWrite(reportPath, out)
}

// patchBareLMEResult handles the rarer case where report.json is
// itself a bare LMEResult.
func patchBareLMEResult(path string, data []byte, result ArbitrationResult) error {
	var lmeResult LMEResult
	if err := json.Unmarshal(data, &lmeResult); err != nil {
		return fmt.Errorf("parse bare report: %w", err)
	}
	applyArbitrationToLME(&lmeResult, result)
	out, err := json.MarshalIndent(&lmeResult, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal bare report: %w", err)
	}
	return atomicWrite(path, out)
}

// applyArbitrationToLME folds the arbitration result into the LMEResult
// additively: the judge score, category map, and exact-match score are
// never overwritten.
func applyArbitrationToLME(lmeResult *LMEResult, result ArbitrationResult) {
	lmeResult.ArbitratedScore = result.ArbitratedScore
	lmeResult.ArbitratedCategoryScores = result.ArbitratedCategoryScores
	lmeResult.ArbitratedReviewed = result.Reviewed

	if len(result.Entries) > 0 && len(lmeResult.Questions) > 0 {
		index := map[string]int{}
		for i, q := range lmeResult.Questions {
			index[q.ID] = i
		}
		for _, e := range result.Entries {
			if e.HumanVerdict == HumanSkip {
				continue
			}
			if i, ok := index[e.QuestionID]; ok {
				lmeResult.Questions[i].HumanVerdict = string(e.HumanVerdict)
			}
		}
	}
}

// atomicWrite stages data in a sibling temp file, fsyncs it, and
// renames it into place.
func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".report-*.json")
	if err != nil {
		return fmt.Errorf("create tmp: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write tmp: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("fsync tmp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close tmp: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename tmp: %w", err)
	}
	return nil
}

// ---- before/after diff ----

// DiffReports compares two LME reports and emits a markdown summary
// highlighting regressions (questions that previously passed, now
// fail) and improvements (previously failed, now pass). The intended
// use is PR review: run the benchmark before and after a change, then
// diff the two reports.
//
// Questions are matched by ID. Questions present in only one side
// are reported separately. Category aggregates and overall score
// deltas round out the summary header.
func DiffReports(before, after *LMEResult) (string, error) {
	if before == nil || after == nil {
		return "", errors.New("diff reports: both reports are required")
	}

	beforeByID := map[string]QuestionOutcome{}
	for _, o := range before.Questions {
		beforeByID[o.ID] = o
	}
	afterByID := map[string]QuestionOutcome{}
	for _, o := range after.Questions {
		afterByID[o.ID] = o
	}

	var regressions, improvements, flaky []DiffEntry
	var onlyBefore, onlyAfter []QuestionOutcome

	for id, b := range beforeByID {
		a, ok := afterByID[id]
		if !ok {
			onlyBefore = append(onlyBefore, b)
			continue
		}
		bc := isCorrectVerdict(b.JudgeVerdict)
		ac := isCorrectVerdict(a.JudgeVerdict)
		if bc && !ac {
			regressions = append(regressions, DiffEntry{Before: b, After: a})
		}
		if !bc && ac {
			improvements = append(improvements, DiffEntry{Before: b, After: a})
		}
		if b.JudgeVerdict != a.JudgeVerdict && bc == ac {
			flaky = append(flaky, DiffEntry{Before: b, After: a})
		}
	}
	for id, a := range afterByID {
		if _, ok := beforeByID[id]; !ok {
			onlyAfter = append(onlyAfter, a)
		}
	}

	sort.Slice(regressions, func(i, j int) bool { return regressions[i].After.ID < regressions[j].After.ID })
	sort.Slice(improvements, func(i, j int) bool { return improvements[i].After.ID < improvements[j].After.ID })
	sort.Slice(flaky, func(i, j int) bool { return flaky[i].After.ID < flaky[j].After.ID })
	sort.Slice(onlyBefore, func(i, j int) bool { return onlyBefore[i].ID < onlyBefore[j].ID })
	sort.Slice(onlyAfter, func(i, j int) bool { return onlyAfter[i].ID < onlyAfter[j].ID })

	var b strings.Builder
	fmt.Fprintln(&b, "# LME Benchmark Diff")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Overall")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| Metric | Before | After | Delta |")
	fmt.Fprintln(&b, "| --- | --- | --- | --- |")
	fmt.Fprintf(&b, "| Questions run | %d | %d | %+d |\n", before.QuestionsRun, after.QuestionsRun, after.QuestionsRun-before.QuestionsRun)
	fmt.Fprintf(&b, "| Overall score | %.3f | %.3f | %+.3f |\n", before.OverallScore, after.OverallScore, after.OverallScore-before.OverallScore)
	fmt.Fprintf(&b, "| Task-avg score | %.3f | %.3f | %+.3f |\n", before.TaskAvgScore, after.TaskAvgScore, after.TaskAvgScore-before.TaskAvgScore)
	fmt.Fprintf(&b, "| Exact match | %.3f | %.3f | %+.3f |\n", before.ExactMatchScore, after.ExactMatchScore, after.ExactMatchScore-before.ExactMatchScore)
	fmt.Fprintf(&b, "| Latency p50 (ms) | %d | %d | %+d |\n", before.LatencyP50Ms, after.LatencyP50Ms, after.LatencyP50Ms-before.LatencyP50Ms)
	fmt.Fprintf(&b, "| Latency p95 (ms) | %d | %d | %+d |\n", before.LatencyP95Ms, after.LatencyP95Ms, after.LatencyP95Ms-before.LatencyP95Ms)
	fmt.Fprintf(&b, "| Total USD | $%.4f | $%.4f | %+.4f |\n", before.CostAccounting.TotalUSD, after.CostAccounting.TotalUSD, after.CostAccounting.TotalUSD-before.CostAccounting.TotalUSD)

	writeCategoryTable(&b, before.ByCategory, after.ByCategory)

	writeDiffSection(&b, "Regressions", "Questions that previously passed and now fail.", regressions)
	writeDiffSection(&b, "Improvements", "Questions that previously failed and now pass.", improvements)
	writeDiffSection(&b, "Verdict churn", "Questions whose verdict changed but the binary outcome did not.", flaky)

	writeOnlySection(&b, "Only in before", onlyBefore)
	writeOnlySection(&b, "Only in after", onlyAfter)

	return b.String(), nil
}

// DiffEntry pairs the same question's before and after outcomes.
type DiffEntry struct {
	Before QuestionOutcome
	After  QuestionOutcome
}

// writeCategoryTable renders category-level score deltas. Missing
// categories on either side land as blank cells so the reviewer can
// spot newly-introduced or removed categories.
func writeCategoryTable(b *strings.Builder, before, after map[string]Category) {
	cats := map[string]bool{}
	for c := range before {
		cats[c] = true
	}
	for c := range after {
		cats[c] = true
	}
	if len(cats) == 0 {
		return
	}
	ordered := make([]string, 0, len(cats))
	for c := range cats {
		ordered = append(ordered, c)
	}
	sort.Strings(ordered)

	fmt.Fprintln(b)
	fmt.Fprintln(b, "## Per-category score")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Category | Before | After | Delta |")
	fmt.Fprintln(b, "| --- | --- | --- | --- |")
	for _, c := range ordered {
		bScore, bOk := before[c]
		aScore, aOk := after[c]
		switch {
		case bOk && aOk:
			fmt.Fprintf(b, "| %s | %.3f | %.3f | %+.3f |\n", c, bScore.Score, aScore.Score, aScore.Score-bScore.Score)
		case bOk:
			fmt.Fprintf(b, "| %s | %.3f | | |\n", c, bScore.Score)
		case aOk:
			fmt.Fprintf(b, "| %s | | %.3f | |\n", c, aScore.Score)
		}
	}
}

// writeDiffSection renders one of the regression / improvement /
// verdict-churn lists.
func writeDiffSection(b *strings.Builder, title, description string, entries []DiffEntry) {
	fmt.Fprintln(b)
	fmt.Fprintf(b, "## %s (%d)\n", title, len(entries))
	fmt.Fprintln(b)
	fmt.Fprintln(b, description)
	fmt.Fprintln(b)
	if len(entries) == 0 {
		fmt.Fprintln(b, "_None._")
		return
	}
	for _, e := range entries {
		id := e.After.ID
		if id == "" {
			id = e.Before.ID
		}
		cat := e.After.Category
		if cat == "" {
			cat = e.Before.Category
		}
		fmt.Fprintf(b, "- **%s** (`%s`): %s -> %s\n", id, cat, e.Before.JudgeVerdict, e.After.JudgeVerdict)
		if q := strings.TrimSpace(firstNonEmpty(e.After.Question, e.Before.Question)); q != "" {
			fmt.Fprintf(b, "  - Q: %s\n", singleLine(q, 240))
		}
		if gt := strings.TrimSpace(firstNonEmpty(e.After.GroundTruth, e.Before.GroundTruth)); gt != "" {
			fmt.Fprintf(b, "  - Truth: %s\n", singleLine(gt, 200))
		}
		if before := strings.TrimSpace(e.Before.AgentAnswer); before != "" {
			fmt.Fprintf(b, "  - Before: %s\n", singleLine(before, 200))
		}
		if after := strings.TrimSpace(e.After.AgentAnswer); after != "" {
			fmt.Fprintf(b, "  - After: %s\n", singleLine(after, 200))
		}
	}
}

// writeOnlySection renders outcomes that appear on only one side of
// the diff.
func writeOnlySection(b *strings.Builder, title string, outcomes []QuestionOutcome) {
	fmt.Fprintln(b)
	fmt.Fprintf(b, "## %s (%d)\n", title, len(outcomes))
	fmt.Fprintln(b)
	if len(outcomes) == 0 {
		fmt.Fprintln(b, "_None._")
		return
	}
	for _, o := range outcomes {
		fmt.Fprintf(b, "- `%s` (%s): %s\n", o.ID, o.Category, o.JudgeVerdict)
	}
}

// firstNonEmpty returns the first non-empty string.
func firstNonEmpty(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

// singleLine collapses whitespace so diff entries stay on one bullet.
func singleLine(s string, max int) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if max > 0 && len(s) > max {
		s = s[:max] + "..."
	}
	return s
}

// LoadReport reads a report file and extracts the inner LMEResult.
// Convenience wrapper for [DiffReports] callers.
func LoadReport(path string) (*LMEResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read report: %w", err)
	}
	return extractLMEResult(data)
}

// WriteDiff writes the markdown diff to w and returns the number of
// bytes written. Separated from [DiffReports] so CLI callers can also
// stream to stdout without buffering the full string twice.
func WriteDiff(w io.Writer, before, after *LMEResult) (int, error) {
	md, err := DiffReports(before, after)
	if err != nil {
		return 0, err
	}
	return io.WriteString(w, md)
}

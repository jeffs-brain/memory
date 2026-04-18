// SPDX-License-Identifier: Apache-2.0

package lme

import "testing"

func TestLatencyPercentile_KnownDistribution(t *testing.T) {
	lats := []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	if got := LatencyPercentile(lats, 50); got != 50 {
		t.Errorf("p50 = %d, want 50", got)
	}
	if got := LatencyPercentile(lats, 95); got != 100 {
		t.Errorf("p95 = %d, want 100", got)
	}
	if got := LatencyPercentile(lats, 0); got != 10 {
		t.Errorf("p0 = %d, want 10", got)
	}
	if got := LatencyPercentile(lats, 100); got != 100 {
		t.Errorf("p100 = %d, want 100", got)
	}
}

func TestLatencyPercentile_Unsorted(t *testing.T) {
	lats := []int{50, 10, 90, 30, 70, 20, 100, 40, 80, 60}
	if got := LatencyPercentile(lats, 50); got != 50 {
		t.Errorf("p50 on unsorted = %d, want 50", got)
	}
	if got := LatencyPercentile(lats, 95); got != 100 {
		t.Errorf("p95 on unsorted = %d, want 100", got)
	}
}

func TestLatencyPercentile_DoesNotMutateInput(t *testing.T) {
	lats := []int{5, 1, 3, 2, 4}
	original := append([]int(nil), lats...)
	_ = LatencyPercentile(lats, 50)
	for i := range lats {
		if lats[i] != original[i] {
			t.Fatalf("input mutated at %d: got %d, want %d", i, lats[i], original[i])
		}
	}
}

func TestLatencyPercentile_EmptyInput(t *testing.T) {
	if got := LatencyPercentile(nil, 50); got != 0 {
		t.Errorf("empty p50 = %d, want 0", got)
	}
	if got := LatencyPercentile([]int{}, 95); got != 0 {
		t.Errorf("empty p95 = %d, want 0", got)
	}
}

func TestLatencyPercentile_SingleValue(t *testing.T) {
	lats := []int{42}
	if got := LatencyPercentile(lats, 50); got != 42 {
		t.Errorf("single p50 = %d, want 42", got)
	}
	if got := LatencyPercentile(lats, 95); got != 42 {
		t.Errorf("single p95 = %d, want 42", got)
	}
}

func TestBootstrapCI_BracketsTrueRate(t *testing.T) {
	outcomes := make([]bool, 100)
	for i := 0; i < 60; i++ {
		outcomes[i] = true
	}

	ci := BootstrapCI(outcomes, 42, 1000)

	if ci[0] > 0.6 || ci[1] < 0.6 {
		t.Errorf("CI %v does not bracket 0.6", ci)
	}
	width := ci[1] - ci[0]
	if width > 0.25 {
		t.Errorf("CI width %.3f unexpectedly wide for n=100 p=0.6", width)
	}
	if ci[0] < 0 || ci[1] > 1 {
		t.Errorf("CI %v escapes [0,1]", ci)
	}
}

func TestBootstrapCI_Deterministic(t *testing.T) {
	outcomes := make([]bool, 50)
	for i := 0; i < 30; i++ {
		outcomes[i] = true
	}

	a := BootstrapCI(outcomes, 123, 500)
	b := BootstrapCI(outcomes, 123, 500)

	if a != b {
		t.Errorf("same seed yields different bounds: a=%v b=%v", a, b)
	}

	c := BootstrapCI(outcomes, 456, 500)
	if a == c {
		t.Logf("warning: different seeds produced identical CI (possible but unlikely): %v", a)
	}
}

func TestBootstrapCI_AllTrue(t *testing.T) {
	outcomes := make([]bool, 20)
	for i := range outcomes {
		outcomes[i] = true
	}
	ci := BootstrapCI(outcomes, 1, 200)
	if ci[0] != 1.0 || ci[1] != 1.0 {
		t.Errorf("all-true CI = %v, want [1,1]", ci)
	}
}

func TestBootstrapCI_AllFalse(t *testing.T) {
	outcomes := make([]bool, 20)
	ci := BootstrapCI(outcomes, 1, 200)
	if ci[0] != 0.0 || ci[1] != 0.0 {
		t.Errorf("all-false CI = %v, want [0,0]", ci)
	}
}

func TestBootstrapCI_EmptyInput(t *testing.T) {
	ci := BootstrapCI(nil, 1, 200)
	if ci != [2]float64{0, 0} {
		t.Errorf("empty CI = %v, want [0,0]", ci)
	}
}

func TestBootstrapCI_DefaultResamples(t *testing.T) {
	outcomes := []bool{true, false, true, true, false}
	ci := BootstrapCI(outcomes, 7, 0)
	if ci[0] < 0 || ci[1] > 1 || ci[0] > ci[1] {
		t.Errorf("default-resample CI malformed: %v", ci)
	}
}

func TestBootstrapCategoryCI_Basic(t *testing.T) {
	outcomes := []QuestionOutcome{
		{ID: "1", Category: "a", JudgeVerdict: "correct"},
		{ID: "2", Category: "a", JudgeVerdict: "correct"},
		{ID: "3", Category: "a", JudgeVerdict: "incorrect"},
		{ID: "4", Category: "b", JudgeVerdict: "correct"},
		{ID: "5", Category: "b", JudgeVerdict: "incorrect"},
		{ID: "6", Category: "b", JudgeVerdict: "incorrect"},
	}

	cis := BootstrapCategoryCI(
		outcomes,
		func(q QuestionOutcome) string { return q.Category },
		func(q QuestionOutcome) bool { return q.JudgeVerdict == "correct" },
		42,
		500,
	)

	if len(cis) != 2 {
		t.Fatalf("expected 2 categories, got %d", len(cis))
	}
	for name, ci := range cis {
		if ci[0] < 0 || ci[1] > 1 || ci[0] > ci[1] {
			t.Errorf("category %q CI malformed: %v", name, ci)
		}
	}
}

func TestBootstrapCategoryCI_Deterministic(t *testing.T) {
	outcomes := []QuestionOutcome{
		{ID: "1", Category: "a", JudgeVerdict: "correct"},
		{ID: "2", Category: "a", JudgeVerdict: "incorrect"},
		{ID: "3", Category: "b", JudgeVerdict: "correct"},
	}
	keyCat := func(q QuestionOutcome) string { return q.Category }
	keyOk := func(q QuestionOutcome) bool { return q.JudgeVerdict == "correct" }

	a := BootstrapCategoryCI(outcomes, keyCat, keyOk, 99, 200)
	b := BootstrapCategoryCI(outcomes, keyCat, keyOk, 99, 200)

	if len(a) != len(b) {
		t.Fatalf("determinism length mismatch: %d vs %d", len(a), len(b))
	}
	for k, v := range a {
		if b[k] != v {
			t.Errorf("category %q differs: a=%v b=%v", k, v, b[k])
		}
	}
}

func TestBootstrapCategoryCI_EmptyInput(t *testing.T) {
	cis := BootstrapCategoryCI(
		nil,
		func(q QuestionOutcome) string { return q.Category },
		func(q QuestionOutcome) bool { return false },
		1,
		100,
	)
	if len(cis) != 0 {
		t.Errorf("expected empty map, got %d entries", len(cis))
	}
}

func TestPopulateStats_WiresEverything(t *testing.T) {
	result := &LMEResult{
		Questions: []QuestionOutcome{
			{ID: "1", Category: "a", JudgeVerdict: "correct", LatencyMs: 10},
			{ID: "2", Category: "a", JudgeVerdict: "incorrect", LatencyMs: 20},
			{ID: "3", Category: "b", JudgeVerdict: "correct", LatencyMs: 30},
			{ID: "4", Category: "b", JudgeVerdict: "abstain_correct", LatencyMs: 40},
		},
	}

	populateStats(result, 42)

	if result.LatencyP50Ms == 0 {
		t.Error("LatencyP50Ms not populated")
	}
	if result.LatencyP95Ms == 0 {
		t.Error("LatencyP95Ms not populated")
	}
	if result.OverallScoreCI == ([2]float64{0, 0}) {
		t.Error("OverallScoreCI not populated")
	}
	if len(result.PerCategoryCI) != 2 {
		t.Errorf("PerCategoryCI has %d entries, want 2", len(result.PerCategoryCI))
	}
}

func TestPopulateStats_NilAndEmpty(t *testing.T) {
	populateStats(nil, 1)

	empty := &LMEResult{}
	populateStats(empty, 1)
	if empty.LatencyP50Ms != 0 || empty.LatencyP95Ms != 0 {
		t.Errorf("empty result should have zero latencies, got p50=%d p95=%d",
			empty.LatencyP50Ms, empty.LatencyP95Ms)
	}
}

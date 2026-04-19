// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunQuestionsConcurrent_ParallelismReducesWallClock(t *testing.T) {
	const (
		n           = 40
		perQuery    = 50 * time.Millisecond
		lowWorkers  = 1
		highWorkers = 8
	)

	qs := make([]Question, n)
	for i := range qs {
		qs[i] = Question{ID: fmt.Sprintf("q%d", i), Category: "single-session", Question: "?", Answer: ""}
	}

	process := func(ctx context.Context, _ int, q Question) QuestionOutcome {
		select {
		case <-time.After(perQuery):
		case <-ctx.Done():
		}
		return QuestionOutcome{ID: q.ID}
	}

	serialStart := time.Now()
	_ = runQuestionsConcurrent(context.Background(), qs, lowWorkers, process)
	serialElapsed := time.Since(serialStart)

	parallelStart := time.Now()
	_ = runQuestionsConcurrent(context.Background(), qs, highWorkers, process)
	parallelElapsed := time.Since(parallelStart)

	if parallelElapsed >= 600*time.Millisecond {
		t.Fatalf("parallel run took %s, expected < 600ms at 8 workers (serial was %s)", parallelElapsed, serialElapsed)
	}
	if serialElapsed < time.Duration(n-2)*perQuery {
		t.Fatalf("serial run too fast: %s, expected >= %s", serialElapsed, time.Duration(n-2)*perQuery)
	}
}

func TestRunQuestionsConcurrent_OrderStable(t *testing.T) {
	qs := make([]Question, 20)
	for i := range qs {
		qs[i] = Question{ID: fmt.Sprintf("q%02d", i), Category: "single-session", Question: "?", Answer: ""}
	}

	process := func(_ context.Context, idx int, q Question) QuestionOutcome {
		if idx%2 == 0 {
			time.Sleep(20 * time.Millisecond)
		} else {
			time.Sleep(5 * time.Millisecond)
		}
		return QuestionOutcome{ID: q.ID, Category: q.Category}
	}

	outcomes := runQuestionsConcurrent(context.Background(), qs, 8, process)

	if len(outcomes) != len(qs) {
		t.Fatalf("len(outcomes) = %d, want %d", len(outcomes), len(qs))
	}
	for i, o := range outcomes {
		if o.ID != qs[i].ID {
			t.Fatalf("outcomes[%d].ID = %q, want %q", i, o.ID, qs[i].ID)
		}
	}
}

func TestRunQuestionsConcurrent_CostSumInvariant(t *testing.T) {
	const (
		n       = 80
		workers = 8
	)
	costs := &CostAccumulator{}

	qs := make([]Question, n)
	for i := range qs {
		qs[i] = Question{ID: fmt.Sprintf("q%d", i)}
	}

	process := func(_ context.Context, _ int, q Question) QuestionOutcome {
		costs.AddAgent(0.01)
		return QuestionOutcome{ID: q.ID}
	}

	_ = runQuestionsConcurrent(context.Background(), qs, workers, process)

	snap := costs.Snapshot()
	want := 0.80
	if math.Abs(snap.AgentUSD-want) > 1e-6 {
		t.Fatalf("AgentUSD = %.10f, want %.10f", snap.AgentUSD, want)
	}
	if math.Abs(snap.TotalUSD-want) > 1e-6 {
		t.Fatalf("TotalUSD = %.10f, want %.10f", snap.TotalUSD, want)
	}
}

func TestRunQuestionsConcurrent_RaceClean(t *testing.T) {
	const n = 100
	qs := make([]Question, n)
	for i := range qs {
		qs[i] = Question{ID: fmt.Sprintf("q%d", i), Category: "temporal"}
	}

	costs := &CostAccumulator{}
	var counter atomic.Int64

	process := func(_ context.Context, idx int, q Question) QuestionOutcome {
		counter.Add(1)
		costs.AddAgent(0.0001)
		return QuestionOutcome{
			ID:          q.ID,
			Category:    q.Category,
			AgentAnswer: fmt.Sprintf("out-%d", idx),
			LatencyMs:   idx,
		}
	}

	outcomes := runQuestionsConcurrent(context.Background(), qs, 16, process)

	if got := counter.Load(); got != int64(n) {
		t.Fatalf("counter = %d, want %d", got, n)
	}
	if len(outcomes) != n {
		t.Fatalf("len(outcomes) = %d, want %d", len(outcomes), n)
	}
	for i, o := range outcomes {
		if o.ID != qs[i].ID {
			t.Fatalf("outcomes[%d].ID = %q, want %q", i, o.ID, qs[i].ID)
		}
	}
}

func TestRunQuestionsConcurrent_CancelledContext(t *testing.T) {
	qs := []Question{
		{ID: "q1", Category: "single-session", Question: "?", Answer: ""},
		{ID: "q2", Category: "single-session", Question: "?", Answer: ""},
		{ID: "q3", Category: "single-session", Question: "?", Answer: ""},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var invoked atomic.Int64
	process := func(_ context.Context, _ int, q Question) QuestionOutcome {
		invoked.Add(1)
		return QuestionOutcome{ID: q.ID}
	}

	outcomes := runQuestionsConcurrent(ctx, qs, 4, process)

	if len(outcomes) != len(qs) {
		t.Fatalf("len(outcomes) = %d, want %d", len(outcomes), len(qs))
	}
	for i, o := range outcomes {
		if o.Error != "context cancelled" {
			t.Fatalf("outcomes[%d].Error = %q, want %q", i, o.Error, "context cancelled")
		}
	}
	if got := invoked.Load(); got != 0 {
		t.Fatalf("process invoked %d times on a pre-cancelled ctx, want 0", got)
	}
}

func TestClampConcurrency(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, defaultConcurrency},
		{-5, defaultConcurrency},
		{1, 1},
		{8, 8},
		{64, 64},
		{256, 256},
		{257, 256},
		{1_000_000, 256},
	}
	for _, c := range cases {
		if got := clampConcurrency(c.in); got != c.want {
			t.Fatalf("clampConcurrency(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

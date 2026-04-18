// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"os"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

// questionProgressInterval matches the judge cadence so a single eval
// run's stderr has a predictable heartbeat regardless of which phase is
// running.
const questionProgressInterval = 50

// runQuestionsConcurrent fans out a per-question processor across a
// bounded worker pool. Per-question errors live on the returned
// outcomes, so the errgroup is never asked to fail-fast; it exists only
// to pipe parent-context cancellation into every in-flight worker.
// Results are written at their source index, preserving the caller's
// order.
func runQuestionsConcurrent(
	ctx context.Context,
	items []Question,
	workers int,
	process func(ctx context.Context, idx int, q Question) QuestionOutcome,
) []QuestionOutcome {
	outcomes := make([]QuestionOutcome, len(items))
	if len(items) == 0 {
		return outcomes
	}

	workers = clampConcurrency(workers)
	sem := make(chan struct{}, workers)
	g, gctx := errgroup.WithContext(ctx)

	start := time.Now()
	retriesBaseline := TransientRetriesTotal()
	var completed atomic.Int64

	for i := range items {
		i := i
		q := items[i]

		if gctx.Err() != nil {
			outcomes[i] = QuestionOutcome{
				ID:           q.ID,
				Category:     q.Category,
				Question:     q.Question,
				QuestionDate: q.QuestionDate,
				GroundTruth:  q.Answer,
				Error:        "context cancelled",
			}
			continue
		}

		select {
		case sem <- struct{}{}:
		case <-gctx.Done():
			outcomes[i] = QuestionOutcome{
				ID:           q.ID,
				Category:     q.Category,
				Question:     q.Question,
				QuestionDate: q.QuestionDate,
				GroundTruth:  q.Answer,
				Error:        "context cancelled",
			}
			continue
		}

		g.Go(func() error {
			defer func() { <-sem }()
			outcomes[i] = process(gctx, i, q)
			done := completed.Add(1)
			logQuestionProgress(int(done), len(items), start, retriesBaseline, outcomes[i])
			return nil
		})
	}
	_ = g.Wait()
	total := int(completed.Load())
	retries := TransientRetriesTotal() - retriesBaseline
	fmt.Fprintf(os.Stderr, "[questions] done %d/%d in %s (transient_retries=%d)\n",
		total, len(items), time.Since(start).Truncate(time.Millisecond), retries)
	return outcomes
}

// logQuestionProgress emits one [questions] line per questionProgressInterval
// completions plus the very first and last.
func logQuestionProgress(done, total int, start time.Time, retriesBaseline int64, o QuestionOutcome) {
	if done != 1 && done%questionProgressInterval != 0 && done != total {
		return
	}
	elapsed := time.Since(start)
	rate := 0.0
	if elapsed > 0 {
		rate = float64(done) / elapsed.Seconds()
	}
	eta := "n/a"
	remaining := total - done
	if rate > 0 && remaining > 0 {
		eta = time.Duration(float64(remaining)/rate * float64(time.Second)).Truncate(time.Second).String()
	}
	status := "ok"
	if o.Error != "" {
		status = "err"
	}
	short := o.ID
	if len(short) > 16 {
		short = short[:16] + "..."
	}
	retries := TransientRetriesTotal() - retriesBaseline
	fmt.Fprintf(os.Stderr, "[questions] %d/%d q=%s %s (%dms) rate=%.1f/s eta=%s retries=%d\n",
		done, total, short, status, o.LatencyMs, rate, eta, retries)
}

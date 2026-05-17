// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

func TestPool_ProcessesJobsConcurrently(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(4, "brain-a")...)
	var processing atomic.Int32
	var maxConcurrent atomic.Int32

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     4,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 5 * time.Second,
		Processor: func(_ context.Context, _ Job) error {
			cur := processing.Add(1)
			for {
				old := maxConcurrent.Load()
				if cur <= old {
					break
				}
				if maxConcurrent.CompareAndSwap(old, cur) {
					break
				}
			}
			time.Sleep(50 * time.Millisecond)
			processing.Add(-1)
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	// Wait for all jobs to complete.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for jobs to complete")
		default:
		}
		if len(adapter.completedIDs()) >= 4 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	if got := maxConcurrent.Load(); got < 2 {
		t.Fatalf("expected at least 2 concurrent workers, got %d", got)
	}
	if got := len(adapter.completedIDs()); got != 4 {
		t.Fatalf("expected 4 completed jobs, got %d", got)
	}
}

func TestPool_PerBrainConcurrencyLimit(t *testing.T) {
	t.Parallel()
	jobs := makeJobs(6, "brain-x")
	adapter := newFakeAdapter(jobs...)
	var brainConcurrent atomic.Int32
	var maxBrainConcurrent atomic.Int32

	pool := NewPool(PoolConfig{
		Queue:               adapter,
		Concurrency:         6,
		PerBrainConcurrency: 2,
		PollInterval:        10 * time.Millisecond,
		ShutdownTimeout:     5 * time.Second,
		Processor: func(_ context.Context, _ Job) error {
			cur := brainConcurrent.Add(1)
			for {
				old := maxBrainConcurrent.Load()
				if cur <= old {
					break
				}
				if maxBrainConcurrent.CompareAndSwap(old, cur) {
					break
				}
			}
			time.Sleep(50 * time.Millisecond)
			brainConcurrent.Add(-1)
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	// Wait for all 6 jobs to complete. Jobs that hit the per-brain
	// concurrency limit are requeued (not failed), so they re-enter
	// the pending pool and are eventually processed.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for jobs")
		default:
		}
		completed := len(adapter.completedIDs())
		if completed >= 6 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	if got := maxBrainConcurrent.Load(); got > 2 {
		t.Fatalf("per-brain concurrency exceeded limit: max observed %d, limit 2", got)
	}

	// Verify requeue was used instead of fail for over-limit jobs.
	if len(adapter.requeuedIDs()) == 0 {
		t.Fatal("expected requeued jobs for per-brain concurrency rejection")
	}
	if len(adapter.failedIDs()) != 0 {
		t.Fatalf("expected no failed jobs from concurrency rejection, got %d", len(adapter.failedIDs()))
	}
}

func TestPool_BackpressureDetection(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.pendingDepth = 1500

	pool := NewPool(PoolConfig{
		Queue:         adapter,
		Concurrency:   2,
		MaxQueueDepth: 1000,
		PollInterval:  10 * time.Millisecond,
		Processor:     func(_ context.Context, _ Job) error { return nil },
	})

	ctx := context.Background()
	pool.refreshBackpressure(ctx)

	if !pool.IsBackpressured() {
		t.Fatal("expected pool to be backpressured when depth exceeds threshold")
	}

	adapter.mu.Lock()
	adapter.pendingDepth = 500
	adapter.mu.Unlock()

	pool.refreshBackpressure(ctx)
	if pool.IsBackpressured() {
		t.Fatal("expected pool not to be backpressured when depth is below threshold")
	}
}

func TestPool_ProcessorErrorMarksJobFailed(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(1, "brain-err")...)

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     1,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 5 * time.Second,
		Processor: func(_ context.Context, _ Job) error {
			return fmt.Errorf("extraction failed: unsupported mime type")
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for failed job")
		default:
		}
		if len(adapter.failedIDs()) >= 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	failedList := adapter.failedIDs()
	if len(failedList) != 1 {
		t.Fatalf("expected 1 failed job, got %d", len(failedList))
	}
	if failedList[0] != "job-0" {
		t.Fatalf("expected failed job ID 'job-0', got %q", failedList[0])
	}
	if len(adapter.completedIDs()) != 0 {
		t.Fatal("expected no completed jobs when processor returns error")
	}
}

func TestPool_GracefulShutdownWaitsForInflight(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(1, "brain-slow")...)
	processingStarted := make(chan struct{})
	processingDone := make(chan struct{})

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     1,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 5 * time.Second,
		Processor: func(_ context.Context, _ Job) error {
			close(processingStarted)
			time.Sleep(200 * time.Millisecond)
			close(processingDone)
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	select {
	case <-processingStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for processing to start")
	}

	err := pool.Stop()
	if err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	select {
	case <-processingDone:
		// The inflight job completed before Stop returned.
	default:
		t.Fatal("Stop returned before inflight job completed")
	}
}

func TestPool_ShutdownTimeoutReturnsError(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(1, "brain-stuck")...)

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     1,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 100 * time.Millisecond,
		Processor: func(ctx context.Context, _ Job) error {
			// Simulate a stuck job that ignores context cancellation.
			time.Sleep(2 * time.Second)
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	// Wait for the job to be claimed.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for claim")
		default:
		}
		adapter.mu.Lock()
		claimed := len(adapter.claimed)
		adapter.mu.Unlock()
		if claimed > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	err := pool.Stop()
	if err == nil {
		t.Fatal("expected timeout error from Stop, got nil")
	}
}

func TestPool_MetricsReflectCurrentState(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(3, "brain-m")...)
	processing := make(chan struct{})

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     2,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 5 * time.Second,
		Processor: func(_ context.Context, _ Job) error {
			<-processing
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	// Wait until at least 2 jobs are being processed.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for active workers")
		default:
		}
		m := pool.Metrics()
		if m.ActiveWorkers >= 2 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	metrics := pool.Metrics()
	if metrics.ActiveWorkers < 2 {
		t.Fatalf("expected at least 2 active workers, got %d", metrics.ActiveWorkers)
	}

	brainCount, hasBrain := metrics.PerBrainActive["brain-m"]
	if !hasBrain {
		t.Fatal("expected brain-m in PerBrainActive map")
	}
	if brainCount < 2 {
		t.Fatalf("expected at least 2 active for brain-m, got %d", brainCount)
	}

	// Unblock processing and let jobs complete.
	close(processing)

	deadline = time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for completion")
		default:
		}
		if len(adapter.completedIDs()) >= 3 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	final := pool.Metrics()
	if final.ProcessedTotal != 3 {
		t.Fatalf("expected ProcessedTotal=3, got %d", final.ProcessedTotal)
	}
}

func TestPool_MultipleBrainsProcessInParallel(t *testing.T) {
	t.Parallel()
	jobs := makeMultiBrainJobs(2, "brain-alpha", "brain-beta")
	adapter := newFakeAdapter(jobs...)
	var brainAlpha, brainBeta atomic.Int32
	var bothActive atomic.Bool

	pool := NewPool(PoolConfig{
		Queue:               adapter,
		Concurrency:         4,
		PerBrainConcurrency: 1,
		PollInterval:        10 * time.Millisecond,
		ShutdownTimeout:     5 * time.Second,
		Processor: func(_ context.Context, job Job) error {
			switch job.BrainID {
			case "brain-alpha":
				brainAlpha.Add(1)
			case "brain-beta":
				brainBeta.Add(1)
			}
			if brainAlpha.Load() > 0 && brainBeta.Load() > 0 {
				bothActive.Store(true)
			}
			time.Sleep(50 * time.Millisecond)
			switch job.BrainID {
			case "brain-alpha":
				brainAlpha.Add(-1)
			case "brain-beta":
				brainBeta.Add(-1)
			}
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for all jobs")
		default:
		}
		completed := len(adapter.completedIDs())
		if completed >= 4 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}

	if !bothActive.Load() {
		t.Fatal("expected both brains to be processed in parallel")
	}
}

func TestPool_HealthyReportsCorrectly(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     2,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 1 * time.Second,
		Processor:       func(_ context.Context, _ Job) error { return nil },
	})

	ctx := context.Background()
	pool.Start(ctx)

	// The pool should be healthy with idle workers.
	time.Sleep(50 * time.Millisecond)
	if !pool.Healthy() {
		t.Fatal("expected pool to be healthy after start")
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}
}

func TestPool_EnvironmentVariableOverrides(t *testing.T) {
	tests := []struct {
		name            string
		envConcurrency  string
		cfgConcurrency  int
		wantConcurrency int
	}{
		{
			name:            "config takes precedence over env",
			envConcurrency:  "8",
			cfgConcurrency:  3,
			wantConcurrency: 3,
		},
		{
			name:            "env used when config is zero",
			envConcurrency:  "12",
			cfgConcurrency:  0,
			wantConcurrency: 12,
		},
		{
			name:            "invalid env falls back to default",
			envConcurrency:  "not-a-number",
			cfgConcurrency:  0,
			wantConcurrency: defaultConcurrency,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envWorkerCount, tc.envConcurrency)
			got := resolveConcurrency(tc.cfgConcurrency)
			if got != tc.wantConcurrency {
				t.Fatalf("resolveConcurrency(%d) with env=%q = %d, want %d",
					tc.cfgConcurrency, tc.envConcurrency, got, tc.wantConcurrency)
			}
		})
	}
}

func TestPool_PollIntervalEnvironmentVariable(t *testing.T) {
	tests := []struct {
		name    string
		envVal  string
		cfgVal  time.Duration
		wantVal time.Duration
	}{
		{
			name:    "config takes precedence",
			envVal:  "5000",
			cfgVal:  2 * time.Second,
			wantVal: 2 * time.Second,
		},
		{
			name:    "env used when config is zero",
			envVal:  "30000",
			cfgVal:  0,
			wantVal: 30 * time.Second,
		},
		{
			name:    "invalid env falls back to default",
			envVal:  "bad",
			cfgVal:  0,
			wantVal: defaultPollInterval,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envPollInterval, tc.envVal)
			got := resolvePollInterval(tc.cfgVal)
			if got != tc.wantVal {
				t.Fatalf("resolvePollInterval(%v) with env=%q = %v, want %v",
					tc.cfgVal, tc.envVal, got, tc.wantVal)
			}
		})
	}
}

func TestPool_WorkerCrashRecovery(t *testing.T) {
	t.Parallel()
	var crashCount atomic.Int32
	adapter := newFakeAdapter(makeJobs(2, "brain-crash")...)

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     1,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 5 * time.Second,
		Processor: func(_ context.Context, job Job) error {
			if job.ID == "job-0" && crashCount.Add(1) == 1 {
				panic("simulated worker crash")
			}
			return nil
		},
	})

	ctx := context.Background()
	pool.Start(ctx)

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for recovery job to complete")
		default:
		}
		// job-0 triggers a panic then completes on retry, job-1 completes normally.
		if len(adapter.completedIDs()) >= 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}
}

func TestPool_BackpressureAutoRefresh(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.mu.Lock()
	adapter.pendingDepth = 1500
	adapter.mu.Unlock()

	pool := NewPool(PoolConfig{
		Queue:           adapter,
		Concurrency:     1,
		MaxQueueDepth:   1000,
		PollInterval:    10 * time.Millisecond,
		ShutdownTimeout: 2 * time.Second,
		Processor:       func(_ context.Context, _ Job) error { return nil },
	})

	ctx := context.Background()
	pool.Start(ctx)

	// Wait for the worker to idle-poll and auto-refresh backpressure.
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for backpressure auto-refresh")
		default:
		}
		if pool.IsBackpressured() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	if !pool.IsBackpressured() {
		t.Fatal("expected pool to be auto-backpressured after idle poll")
	}

	// Verify QueueDepth is populated in metrics.
	m := pool.Metrics()
	if m.QueueDepth != 1500 {
		t.Fatalf("expected QueueDepth=1500, got %d", m.QueueDepth)
	}

	if err := pool.Stop(); err != nil {
		t.Fatalf("pool.Stop() returned error: %v", err)
	}
}

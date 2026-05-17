// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeAdapter is a test double for the queue Adapter interface. It
// dispenses pre-loaded jobs from a thread-safe slice and tracks
// completion/failure calls.
type fakeAdapter struct {
	mu         sync.Mutex
	jobs       []Job
	claimed    []string
	completed  []string
	failed     []string
	failReason map[string]string
	depth      int64
	claimErr   error
	claimDelay time.Duration
}

func newFakeAdapter(jobs ...Job) *fakeAdapter {
	return &fakeAdapter{
		jobs:       jobs,
		failReason: make(map[string]string),
	}
}

func (f *fakeAdapter) Claim(_ context.Context, workerID string) (*Job, error) {
	if f.claimDelay > 0 {
		time.Sleep(f.claimDelay)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.claimErr != nil {
		return nil, f.claimErr
	}
	if len(f.jobs) == 0 {
		return nil, nil
	}
	job := f.jobs[0]
	f.jobs = f.jobs[1:]
	job.Status = JobStatusRunning
	job.ClaimedAt = time.Now()
	f.claimed = append(f.claimed, job.ID)
	return &job, nil
}

func (f *fakeAdapter) Complete(_ context.Context, jobID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.completed = append(f.completed, jobID)
	return nil
}

func (f *fakeAdapter) Fail(_ context.Context, jobID string, reason string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failed = append(f.failed, jobID)
	f.failReason[jobID] = reason
	return nil
}

func (f *fakeAdapter) Heartbeat(_ context.Context, _ string) error {
	return nil
}

func (f *fakeAdapter) Depth(_ context.Context, _ string) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.depth, nil
}

func (f *fakeAdapter) completedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]string, len(f.completed))
	copy(cp, f.completed)
	return cp
}

func (f *fakeAdapter) failedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]string, len(f.failed))
	copy(cp, f.failed)
	return cp
}

func makeJobs(count int, brainID string) []Job {
	jobs := make([]Job, count)
	for i := range count {
		jobs[i] = Job{
			ID:       fmt.Sprintf("job-%d", i),
			BrainID:  brainID,
			Payload:  []byte(fmt.Sprintf(`{"doc":"%d"}`, i)),
			Status:   JobStatusPending,
			Attempts: 0,
		}
	}
	return jobs
}

func makeMultiBrainJobs(perBrain int, brainIDs ...string) []Job {
	jobs := make([]Job, 0, perBrain*len(brainIDs))
	seq := 0
	for _, brainID := range brainIDs {
		for range perBrain {
			jobs = append(jobs, Job{
				ID:      fmt.Sprintf("job-%d", seq),
				BrainID: brainID,
				Payload: []byte(`{}`),
				Status:  JobStatusPending,
			})
			seq++
		}
	}
	return jobs
}

func TestPool_ProcessesJobsConcurrently(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter(makeJobs(4, "brain-a")...)
	var processing atomic.Int32
	var maxConcurrent atomic.Int32

	pool := NewPool(PoolConfig{
		Queue:       adapter,
		Concurrency: 4,
		PollInterval: 10 * time.Millisecond,
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

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for jobs")
		default:
		}
		completed := len(adapter.completedIDs())
		failed := len(adapter.failedIDs())
		if completed+failed >= 6 {
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
}

func TestPool_BackpressureDetection(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.depth = 1500

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
	adapter.depth = 500
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
		Queue:        adapter,
		Concurrency:  1,
		PollInterval: 10 * time.Millisecond,
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
		Queue:        adapter,
		Concurrency:  2,
		PollInterval: 10 * time.Millisecond,
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
		failed := len(adapter.failedIDs())
		if completed+failed >= 4 {
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
		name     string
		envVal   string
		cfgVal   time.Duration
		wantVal  time.Duration
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

func TestBackpressureChecker_DefaultThreshold(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	checker := NewBackpressureChecker(adapter, 0)
	if checker.MaxDepth() != defaultMaxQueueDepth {
		t.Fatalf("expected default threshold %d, got %d", defaultMaxQueueDepth, checker.MaxDepth())
	}
}

func TestBackpressureChecker_CustomThreshold(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	checker := NewBackpressureChecker(adapter, 500)
	if checker.MaxDepth() != 500 {
		t.Fatalf("expected threshold 500, got %d", checker.MaxDepth())
	}
}

func TestBackpressureChecker_CheckUpdatesState(t *testing.T) {
	t.Parallel()
	adapter := newFakeAdapter()
	adapter.depth = 100
	checker := NewBackpressureChecker(adapter, 50)

	ctx := context.Background()
	pressured, err := checker.Check(ctx, "")
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if !pressured {
		t.Fatal("expected backpressured when depth 100 >= threshold 50")
	}
	if !checker.IsBackpressured() {
		t.Fatal("IsBackpressured should reflect last check")
	}

	adapter.mu.Lock()
	adapter.depth = 30
	adapter.mu.Unlock()

	pressured, err = checker.Check(ctx, "")
	if err != nil {
		t.Fatalf("Check returned error: %v", err)
	}
	if pressured {
		t.Fatal("expected not backpressured when depth 30 < threshold 50")
	}
}

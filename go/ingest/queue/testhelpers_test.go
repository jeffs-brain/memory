// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// fakeAdapter is a test double for the queue Adapter interface. It
// dispenses pre-loaded jobs from a thread-safe slice and tracks
// completion/failure/requeue calls.
type fakeAdapter struct {
	mu           sync.Mutex
	jobs         []Job
	claimed      []string
	completed    []string
	failed       []string
	requeued     []string
	failReason   map[string]string
	claimedJobs  map[string]Job // track full job details by ID for requeue
	pendingDepth int64
	claimErr     error
	claimDelay   time.Duration
}

func newFakeAdapter(jobs ...Job) *fakeAdapter {
	return &fakeAdapter{
		jobs:        jobs,
		failReason:  make(map[string]string),
		claimedJobs: make(map[string]Job),
	}
}

func (f *fakeAdapter) Enqueue(_ context.Context, input EnqueueInput) (Job, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	job := Job{
		ID:         fmt.Sprintf("enqueued-%d", len(f.jobs)),
		BrainID:    input.BrainID,
		Status:     StatusPending,
		Payload:    input.Payload,
		MaxRetries: input.MaxRetries,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	f.jobs = append(f.jobs, job)
	return job, nil
}

func (f *fakeAdapter) Claim(_ context.Context, opts ClaimOptions) ([]Job, error) {
	if f.claimDelay > 0 {
		time.Sleep(f.claimDelay)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.claimErr != nil {
		return nil, f.claimErr
	}

	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}

	var result []Job
	var remaining []Job
	now := time.Now()
	for _, job := range f.jobs {
		if job.Status == StatusPending && len(result) < batchSize {
			job.Status = StatusProcessing
			job.ClaimedBy = opts.WorkerID
			job.ClaimedAt = &now
			f.claimed = append(f.claimed, job.ID)
			f.claimedJobs[job.ID] = job
			result = append(result, job)
		} else {
			remaining = append(remaining, job)
		}
	}
	f.jobs = remaining
	return result, nil
}

func (f *fakeAdapter) Complete(_ context.Context, jobID string, _ map[string]string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.completed = append(f.completed, jobID)
	return nil
}

func (f *fakeAdapter) Fail(_ context.Context, jobID string, reason string, _ bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failed = append(f.failed, jobID)
	f.failReason[jobID] = reason
	return nil
}

func (f *fakeAdapter) Requeue(_ context.Context, jobID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requeued = append(f.requeued, jobID)
	// Return the job to the pending pool so another worker can claim it.
	original, ok := f.claimedJobs[jobID]
	if ok {
		original.Status = StatusPending
		original.ClaimedBy = ""
		original.ClaimedAt = nil
		f.jobs = append(f.jobs, original)
		delete(f.claimedJobs, jobID)
	}
	return nil
}

func (f *fakeAdapter) Heartbeat(_ context.Context, _ string) error {
	return nil
}

func (f *fakeAdapter) RecoverStale(_ context.Context, _ time.Duration) (int, error) {
	return 0, nil
}

func (f *fakeAdapter) CountByStatus(_ context.Context, _ string) (map[JobStatus]int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return map[JobStatus]int{
		StatusPending: int(f.pendingDepth),
	}, nil
}

func (f *fakeAdapter) Close() error {
	return nil
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

func (f *fakeAdapter) requeuedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	cp := make([]string, len(f.requeued))
	copy(cp, f.requeued)
	return cp
}

func makeJobs(count int, brainID string) []Job {
	jobs := make([]Job, count)
	for i := range count {
		jobs[i] = Job{
			ID:      fmt.Sprintf("job-%d", i),
			BrainID: brainID,
			Payload: JobPayload{
				Kind:    "raw",
				Content: fmt.Sprintf(`{"doc":"%d"}`, i),
			},
			Status:     StatusPending,
			RetryCount: 0,
			MaxRetries: defaultMaxRetries,
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
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
				Payload: JobPayload{
					Kind: "raw",
				},
				Status:     StatusPending,
				MaxRetries: defaultMaxRetries,
				CreatedAt:  time.Now(),
				UpdatedAt:  time.Now(),
			})
			seq++
		}
	}
	return jobs
}

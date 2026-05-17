// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// defaultConcurrency is 4x the CPU count, following the Celery/Airflow
// convention for I/O-bound workloads where workers spend most time
// waiting on network calls (embedding, LLM).
var defaultConcurrency = 4 * runtime.NumCPU()

// defaultPerBrainConcurrency limits how many workers can process jobs
// for the same brain simultaneously. Follows the AWS fair scheduling
// pattern for multi-tenant systems, preventing a single large brain
// from monopolising all workers.
const defaultPerBrainConcurrency = 5

// defaultPollInterval is how often idle workers poll the queue for new
// jobs. 15 seconds balances responsiveness against database load.
const defaultPollInterval = 15 * time.Second

// defaultShutdownTimeout is the maximum time Stop waits for in-flight
// jobs before cancelling them. 2 minutes follows Google Cloud K8s best
// practice for ingestion pipeline drain windows.
const defaultShutdownTimeout = 120 * time.Second

// envWorkerCount is the environment variable that overrides concurrency.
const envWorkerCount = "MEMORY_WORKER_COUNT"

// envPollInterval is the environment variable that overrides the poll
// interval in milliseconds.
const envPollInterval = "MEMORY_INGEST_WORKER_INTERVAL_MS"

// PoolConfig configures the worker pool behaviour. All fields are
// optional; zero values fall back to documented defaults.
type PoolConfig struct {
	Queue               Adapter
	Processor           func(ctx context.Context, job Job) error
	Concurrency         int
	PerBrainConcurrency int
	PollInterval        time.Duration
	ShutdownTimeout     time.Duration
	MaxQueueDepth       int64
	WorkerID            string
	Logger              Logger
}

// PoolMetrics captures a point-in-time snapshot of pool health and
// throughput counters. All fields are safe to read concurrently.
type PoolMetrics struct {
	ActiveWorkers  int
	IdleWorkers    int
	QueueDepth     int64
	ProcessedTotal int64
	FailedTotal    int64
	PerBrainActive map[string]int
}

// Pool manages a set of concurrent workers that claim and process
// ingestion jobs from a queue adapter. It enforces per-brain concurrency
// limits, detects backpressure, and supports graceful shutdown.
//
// Concurrency model: mu guards brainActive and workerStates. The
// atomic counters (processed, failed, active) are lock-free.
type Pool struct {
	cfg         PoolConfig
	logger      Logger
	backpressure *BackpressureChecker

	// mu guards brainActive and workerStates.
	mu           sync.Mutex
	brainActive  map[string]int
	workerStates []workerState

	processed atomic.Int64
	failed    atomic.Int64
	active    atomic.Int32

	cancel context.CancelFunc
	wg     sync.WaitGroup
	once   sync.Once
}

type workerState struct {
	id     int
	status string // "idle", "processing", "stopped"
}

// NewPool creates a pool with the given configuration. Call Start to
// begin processing. The pool reads MEMORY_WORKER_COUNT and
// MEMORY_INGEST_WORKER_INTERVAL_MS from the environment if the
// corresponding config fields are zero.
func NewPool(cfg PoolConfig) *Pool {
	concurrency := resolveConcurrency(cfg.Concurrency)
	pollInterval := resolvePollInterval(cfg.PollInterval)
	perBrain := cfg.PerBrainConcurrency
	if perBrain <= 0 {
		perBrain = defaultPerBrainConcurrency
	}
	shutdownTimeout := cfg.ShutdownTimeout
	if shutdownTimeout <= 0 {
		shutdownTimeout = defaultShutdownTimeout
	}
	workerID := cfg.WorkerID
	if workerID == "" {
		workerID = uuid.New().String()
	}
	logger := cfg.Logger
	if logger == nil {
		logger = noopLogger{}
	}

	resolved := PoolConfig{
		Queue:               cfg.Queue,
		Processor:           cfg.Processor,
		Concurrency:         concurrency,
		PerBrainConcurrency: perBrain,
		PollInterval:        pollInterval,
		ShutdownTimeout:     shutdownTimeout,
		MaxQueueDepth:       cfg.MaxQueueDepth,
		WorkerID:            workerID,
		Logger:              logger,
	}

	states := make([]workerState, concurrency)
	for i := range states {
		states[i] = workerState{id: i, status: "idle"}
	}

	return &Pool{
		cfg:          resolved,
		logger:       logger,
		backpressure: NewBackpressureChecker(cfg.Queue, cfg.MaxQueueDepth),
		brainActive:  make(map[string]int, perBrain),
		workerStates: states,
	}
}

// Start launches all worker goroutines. Each worker loops: claim a job,
// check per-brain limits, process, complete/fail, repeat. The pool
// stops when ctx is cancelled or Stop is called.
func (p *Pool) Start(ctx context.Context) {
	workerCtx, cancel := context.WithCancel(ctx)
	p.cancel = cancel

	p.logger.Info("pool starting",
		"concurrency", p.cfg.Concurrency,
		"perBrainConcurrency", p.cfg.PerBrainConcurrency,
		"pollInterval", p.cfg.PollInterval.String(),
		"workerID", p.cfg.WorkerID,
	)

	for i := range p.cfg.Concurrency {
		p.wg.Add(1)
		go p.runWorker(workerCtx, i)
	}
}

// Stop signals all workers to cease claiming new jobs and waits for
// in-flight work to complete, up to the configured shutdown timeout.
// Returns an error if the timeout is exceeded.
func (p *Pool) Stop() error {
	var timedOut bool
	p.once.Do(func() {
		p.logger.Info("pool stopping", "shutdownTimeout", p.cfg.ShutdownTimeout.String())
		p.cancel()

		done := make(chan struct{})
		go func() {
			p.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			p.logger.Info("pool stopped gracefully")
		case <-time.After(p.cfg.ShutdownTimeout):
			p.logger.Warn("pool shutdown timed out, some workers may still be running")
			timedOut = true
		}
	})
	if timedOut {
		return fmt.Errorf("queue: pool shutdown exceeded %s timeout", p.cfg.ShutdownTimeout)
	}
	return nil
}

// Metrics returns a point-in-time snapshot of the pool's health and
// throughput counters.
func (p *Pool) Metrics() PoolMetrics {
	p.mu.Lock()
	brainCopy := make(map[string]int, len(p.brainActive))
	for k, v := range p.brainActive {
		brainCopy[k] = v
	}
	p.mu.Unlock()

	activeCount := int(p.active.Load())
	return PoolMetrics{
		ActiveWorkers:  activeCount,
		IdleWorkers:    p.cfg.Concurrency - activeCount,
		ProcessedTotal: p.processed.Load(),
		FailedTotal:    p.failed.Load(),
		PerBrainActive: brainCopy,
	}
}

// IsBackpressured returns whether the queue depth exceeds the
// configured threshold. Uses the cached value from the most recent
// backpressure check.
func (p *Pool) IsBackpressured() bool {
	return p.backpressure.IsBackpressured()
}

// Healthy returns true when at least one worker is running and the
// pool has not been stopped.
func (p *Pool) Healthy() bool {
	return p.active.Load() > 0 || p.idleCount() > 0
}

func (p *Pool) idleCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	count := 0
	for _, ws := range p.workerStates {
		if ws.status == "idle" {
			count++
		}
	}
	return count
}

func (p *Pool) runWorker(ctx context.Context, workerIdx int) {
	defer p.wg.Done()
	qualifiedID := fmt.Sprintf("%s-%d", p.cfg.WorkerID, workerIdx)
	p.logger.Debug("worker started", "worker", qualifiedID)

	for {
		select {
		case <-ctx.Done():
			p.setWorkerStatus(workerIdx, "stopped")
			p.logger.Debug("worker stopped", "worker", qualifiedID)
			return
		default:
		}

		claimed := p.claimAndProcess(ctx, qualifiedID, workerIdx)
		if !claimed {
			p.pollWait(ctx)
		}
	}
}

func (p *Pool) claimAndProcess(ctx context.Context, qualifiedID string, workerIdx int) bool {
	job, err := p.cfg.Queue.Claim(ctx, qualifiedID)
	if err != nil {
		if ctx.Err() != nil {
			return false
		}
		p.logger.Warn("claim failed", "worker", qualifiedID, "error", err.Error())
		return false
	}
	if job == nil {
		return false
	}

	if !p.acquireBrainSlot(job.BrainID) {
		p.logger.Debug("per-brain concurrency limit reached, releasing job",
			"worker", qualifiedID, "brainID", job.BrainID)
		_ = p.cfg.Queue.Fail(ctx, job.ID, "per-brain concurrency limit reached")
		return true
	}

	p.setWorkerStatus(workerIdx, "processing")
	p.active.Add(1)

	processErr := p.cfg.Processor(ctx, *job)

	p.active.Add(-1)
	p.releaseBrainSlot(job.BrainID)
	p.setWorkerStatus(workerIdx, "idle")

	if processErr != nil {
		p.failed.Add(1)
		failErr := p.cfg.Queue.Fail(ctx, job.ID, processErr.Error())
		if failErr != nil {
			p.logger.Error("failed to mark job as failed",
				"worker", qualifiedID, "jobID", job.ID, "error", failErr.Error())
		}
		p.logger.Warn("job processing failed",
			"worker", qualifiedID, "jobID", job.ID, "brainID", job.BrainID,
			"error", processErr.Error())
		return true
	}

	p.processed.Add(1)
	completeErr := p.cfg.Queue.Complete(ctx, job.ID)
	if completeErr != nil {
		p.logger.Error("failed to mark job as completed",
			"worker", qualifiedID, "jobID", job.ID, "error", completeErr.Error())
	}
	return true
}

func (p *Pool) acquireBrainSlot(brainID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	current := p.brainActive[brainID]
	if current >= p.cfg.PerBrainConcurrency {
		return false
	}
	p.brainActive[brainID] = current + 1
	return true
}

func (p *Pool) releaseBrainSlot(brainID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	current := p.brainActive[brainID]
	if current <= 1 {
		delete(p.brainActive, brainID)
		return
	}
	p.brainActive[brainID] = current - 1
}

func (p *Pool) setWorkerStatus(idx int, status string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if idx < len(p.workerStates) {
		p.workerStates[idx].status = status
	}
}

func (p *Pool) pollWait(ctx context.Context) {
	timer := time.NewTimer(p.cfg.PollInterval)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

// refreshBackpressure updates the cached backpressure state by
// querying the queue adapter for current depth.
func (p *Pool) refreshBackpressure(ctx context.Context) {
	_, err := p.backpressure.Check(ctx, "")
	if err != nil {
		p.logger.Warn("backpressure check failed", "error", err.Error())
	}
}

func resolveConcurrency(configured int) int {
	if configured > 0 {
		return configured
	}
	if envVal := os.Getenv(envWorkerCount); envVal != "" {
		parsed, err := strconv.Atoi(envVal)
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	return defaultConcurrency
}

func resolvePollInterval(configured time.Duration) time.Duration {
	if configured > 0 {
		return configured
	}
	if envVal := os.Getenv(envPollInterval); envVal != "" {
		parsed, err := strconv.Atoi(envVal)
		if err == nil && parsed > 0 {
			return time.Duration(parsed) * time.Millisecond
		}
	}
	return defaultPollInterval
}

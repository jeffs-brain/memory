// SPDX-License-Identifier: Apache-2.0
package schedule

import (
	"context"
	"sync"
	"time"
)

const defaultPollInterval = 30 * time.Second

// SchedulerOptions configures the scheduler.
type SchedulerOptions struct {
	Store        Store
	Dispatch     DispatchFunc
	PollInterval time.Duration
	Logger       Logger
	// Now is an injectable clock for testing. Defaults to time.Now.
	Now func() time.Time
	// CronEngine overrides the built-in cron parser. When nil, the
	// default engine (ParseCron + NextOccurrence) is used.
	CronEngine CronEngine
}

// Scheduler polls the schedule store for due jobs and fires them.
type Scheduler struct {
	opts   SchedulerOptions
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewScheduler creates a scheduler. Call Start to begin polling.
func NewScheduler(opts SchedulerOptions) *Scheduler {
	if opts.PollInterval == 0 {
		opts.PollInterval = defaultPollInterval
	}
	if opts.Now == nil {
		opts.Now = func() time.Time { return time.Now().UTC() }
	}
	if opts.CronEngine == nil {
		opts.CronEngine = NewDefaultCronEngine()
	}
	return &Scheduler{opts: opts}
}

// Start begins the polling loop in a background goroutine.
func (s *Scheduler) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	s.wg.Add(1)
	go s.poll(ctx)
}

// Stop signals the scheduler to stop and waits for the current poll to
// complete.
func (s *Scheduler) Stop() error {
	if s.cancel != nil {
		s.cancel()
	}
	s.wg.Wait()
	return nil
}

// RunDueJobs is exposed for testing. It checks for due jobs and fires
// them, returning the number of jobs fired.
func (s *Scheduler) RunDueJobs(ctx context.Context) (int, error) {
	now := s.opts.Now()
	jobs, err := s.opts.Store.FindDue(ctx, now)
	if err != nil {
		return 0, err
	}

	fired := 0
	for _, job := range jobs {
		if err := s.opts.Dispatch(ctx, job); err != nil {
			if s.opts.Logger != nil {
				s.opts.Logger.Error("schedule: dispatch failed", map[string]string{
					"jobId": job.ID,
					"error": err.Error(),
				})
			}
			// Continue with next job — one failure does not block others.
			continue
		}

		// Reschedule using the configured cron engine.
		nextRun, cronErr := s.opts.CronEngine.NextOccurrence(job.CronExpression, now)
		if cronErr != nil {
			if s.opts.Logger != nil {
				s.opts.Logger.Error("schedule: cron next-occurrence failed", map[string]string{
					"jobId":          job.ID,
					"cronExpression": job.CronExpression,
					"error":          cronErr.Error(),
				})
			}
			continue
		}
		if err := s.opts.Store.MarkRun(ctx, job.ID, now, nextRun); err != nil {
			if s.opts.Logger != nil {
				s.opts.Logger.Error("schedule: markRun failed", map[string]string{
					"jobId": job.ID,
					"error": err.Error(),
				})
			}
		}
		fired++
	}

	return fired, nil
}

func (s *Scheduler) poll(ctx context.Context) {
	defer s.wg.Done()
	ticker := time.NewTicker(s.opts.PollInterval)
	defer ticker.Stop()

	// Run immediately on start.
	if _, err := s.RunDueJobs(ctx); err != nil {
		if s.opts.Logger != nil {
			s.opts.Logger.Error("schedule: poll error", map[string]string{
				"error": err.Error(),
			})
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := s.RunDueJobs(ctx); err != nil {
				if s.opts.Logger != nil {
					s.opts.Logger.Error("schedule: poll error", map[string]string{
						"error": err.Error(),
					})
				}
			}
		}
	}
}

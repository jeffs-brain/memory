// SPDX-License-Identifier: Apache-2.0
package schedule

import (
	"context"
	"database/sql"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestStoreCRUD(t *testing.T) {
	db := openTestDB(t)
	store, err := NewSQLiteStore(db)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	// Create.
	job, err := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "daily re-ingest",
		CronExpression: "0 2 * * *",
		Target:         Target{Kind: "url", URL: "https://example.com/feed"},
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if job.ID == "" {
		t.Fatal("expected non-empty ID")
	}
	if !job.Enabled {
		t.Fatal("expected enabled")
	}

	// Get.
	got, err := store.Get(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if got.Name != "daily re-ingest" {
		t.Fatalf("expected 'daily re-ingest', got %q", got.Name)
	}

	// List.
	jobs, err := store.List(context.Background(), "brain-1")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(jobs) != 1 {
		t.Fatalf("expected 1 job, got %d", len(jobs))
	}

	// Update.
	newName := "weekly re-ingest"
	updated, err := store.Update(context.Background(), job.ID, UpdatePatch{Name: &newName})
	if err != nil {
		t.Fatalf("update: %v", err)
	}
	if updated.Name != "weekly re-ingest" {
		t.Fatalf("expected 'weekly re-ingest', got %q", updated.Name)
	}

	// Delete.
	err = store.Delete(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	jobs, _ = store.List(context.Background(), "brain-1")
	if len(jobs) != 0 {
		t.Fatalf("expected 0 jobs after delete, got %d", len(jobs))
	}
}

func TestFindDueReturnsOnlyEnabledDueJobs(t *testing.T) {
	db := openTestDB(t)
	store, err := NewSQLiteStore(db)
	if err != nil {
		t.Fatalf("new store: %v", err)
	}

	// Create a due job.
	job1, err := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "due job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/a.md"},
	})
	if err != nil {
		t.Fatalf("create job1: %v", err)
	}

	// Force next_run_at to the past.
	past := time.Now().UTC().Add(-1 * time.Hour)
	if err := store.MarkRun(context.Background(), job1.ID, past.Add(-2*time.Hour), past); err != nil {
		t.Fatalf("mark run job1: %v", err)
	}

	// Create a disabled job.
	job2, err := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "disabled job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/b.md"},
	})
	if err != nil {
		t.Fatalf("create job2: %v", err)
	}
	enabled := false
	if _, err := store.Update(context.Background(), job2.ID, UpdatePatch{Enabled: &enabled}); err != nil {
		t.Fatalf("update job2: %v", err)
	}

	due, err := store.FindDue(context.Background(), time.Now().UTC())
	if err != nil {
		t.Fatalf("findDue: %v", err)
	}

	if len(due) != 1 {
		t.Fatalf("expected 1 due job, got %d", len(due))
	}
	if due[0].ID != job1.ID {
		t.Fatalf("expected job %s, got %s", job1.ID, due[0].ID)
	}
}

func TestSchedulerRunDueJobs(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	var dispatched atomic.Int32

	job, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "test job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/test.md"},
	})

	// Force next_run_at to the past.
	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job.ID, past.Add(-2*time.Hour), past)

	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store: store,
		Dispatch: func(_ context.Context, j Job) error {
			dispatched.Add(1)
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, err := scheduler.RunDueJobs(context.Background())
	if err != nil {
		t.Fatalf("runDueJobs: %v", err)
	}
	if fired != 1 {
		t.Fatalf("expected 1 fired, got %d", fired)
	}
	if got := dispatched.Load(); got != 1 {
		t.Fatalf("expected 1 dispatch, got %d", got)
	}

	// After running, the job should be rescheduled (nextRunAt in the future).
	updated, _ := store.Get(context.Background(), job.ID)
	if updated.NextRunAt.Before(time.Now().UTC()) {
		t.Fatal("expected nextRunAt in the future after run")
	}
}

func TestSchedulerDisabledJobSkipped(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	var dispatched atomic.Int32

	job, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "disabled job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/test.md"},
	})

	enabled := false
	store.Update(context.Background(), job.ID, UpdatePatch{Enabled: &enabled})

	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store: store,
		Dispatch: func(_ context.Context, _ Job) error {
			dispatched.Add(1)
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, _ := scheduler.RunDueJobs(context.Background())
	if fired != 0 {
		t.Fatalf("expected 0 fired for disabled job, got %d", fired)
	}
}

func TestSchedulerJobFailureDoesNotBlockOthers(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	job1, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "failing job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/fail.md"},
	})
	job2, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "succeeding job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/ok.md"},
	})

	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job1.ID, past.Add(-2*time.Hour), past)
	store.MarkRun(context.Background(), job2.ID, past.Add(-2*time.Hour), past)

	var succeeded atomic.Int32
	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store:  store,
		Logger: &testLogger{},
		Dispatch: func(_ context.Context, j Job) error {
			if j.Name == "failing job" {
				return errors.New("dispatch error")
			}
			succeeded.Add(1)
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, _ := scheduler.RunDueJobs(context.Background())
	// Only the successful one counts as fired.
	if fired != 1 {
		t.Fatalf("expected 1 fired (the successful one), got %d", fired)
	}
	if got := succeeded.Load(); got != 1 {
		t.Fatalf("expected 1 succeeded, got %d", got)
	}
}

func TestSchedulerNoDueJobs(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	var dispatched atomic.Int32
	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store: store,
		Dispatch: func(_ context.Context, _ Job) error {
			dispatched.Add(1)
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, _ := scheduler.RunDueJobs(context.Background())
	if fired != 0 {
		t.Fatalf("expected 0 fired, got %d", fired)
	}
}

func TestSchedulerPerBrainIsolation(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	// Create jobs for different brains.
	job1, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-a",
		Name:           "job-a",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/a.md"},
	})
	job2, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-b",
		Name:           "job-b",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/b.md"},
	})

	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job1.ID, past.Add(-2*time.Hour), past)
	store.MarkRun(context.Background(), job2.ID, past.Add(-2*time.Hour), past)

	brains := map[string]int{}
	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store: store,
		Dispatch: func(_ context.Context, j Job) error {
			brains[j.BrainID]++
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, _ := scheduler.RunDueJobs(context.Background())
	if fired != 2 {
		t.Fatalf("expected 2 fired, got %d", fired)
	}
	if brains["brain-a"] != 1 || brains["brain-b"] != 1 {
		t.Fatalf("expected 1 per brain, got %v", brains)
	}
}

func TestInvalidCronExpressionRejected(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	_, err := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "bad cron",
		CronExpression: "invalid",
		Target:         Target{Kind: "file", Path: "/data/test.md"},
	})
	if err == nil {
		t.Fatal("expected error for invalid cron expression")
	}
}

func TestSchedulerStartStop(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	var dispatched atomic.Int32

	job, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "lifecycle job",
		CronExpression: "* * * * *",
		Target:         Target{Kind: "file", Path: "/data/lifecycle.md"},
	})

	// Force next_run_at to the past.
	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job.ID, past.Add(-2*time.Hour), past)

	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store:        store,
		PollInterval: 50 * time.Millisecond,
		Dispatch: func(_ context.Context, _ Job) error {
			dispatched.Add(1)
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	scheduler.Start()
	// Wait until at least one dispatch occurs (bounded to prevent CI stalls).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if dispatched.Load() > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	err := scheduler.Stop()
	if err != nil {
		t.Fatalf("stop: %v", err)
	}

	if got := dispatched.Load(); got < 1 {
		t.Fatalf("expected at least 1 dispatch after start, got %d", got)
	}
}

func TestSchedulerCustomCronEngine(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	fixedNext := time.Date(2099, 1, 1, 0, 0, 0, 0, time.UTC)
	engine := &stubCronEngine{
		nextTime: fixedNext,
		valid:    true,
	}

	job, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "custom engine job",
		CronExpression: "0 3 * * *",
		Target:         Target{Kind: "file", Path: "/data/custom.md"},
	})

	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job.ID, past.Add(-2*time.Hour), past)

	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store:      store,
		CronEngine: engine,
		Dispatch: func(_ context.Context, _ Job) error {
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, err := scheduler.RunDueJobs(context.Background())
	if err != nil {
		t.Fatalf("runDueJobs: %v", err)
	}
	if fired != 1 {
		t.Fatalf("expected 1 fired, got %d", fired)
	}

	// Verify the custom engine was used: nextRunAt should be our fixed time.
	updated, _ := store.Get(context.Background(), job.ID)
	if !updated.NextRunAt.Equal(fixedNext) {
		t.Fatalf("expected nextRunAt %v from custom engine, got %v", fixedNext, updated.NextRunAt)
	}

	if engine.callCount == 0 {
		t.Fatal("expected custom cron engine to be called")
	}
}

func TestSchedulerDefaultCronEngine(t *testing.T) {
	db := openTestDB(t)
	store, _ := NewSQLiteStore(db)

	job, _ := store.Create(context.Background(), CreateInput{
		BrainID:        "brain-1",
		Name:           "default engine job",
		CronExpression: "0 * * * *",
		Target:         Target{Kind: "file", Path: "/data/default.md"},
	})

	past := time.Now().UTC().Add(-1 * time.Hour)
	store.MarkRun(context.Background(), job.ID, past.Add(-2*time.Hour), past)

	scheduler, sErr := NewScheduler(SchedulerOptions{
		Store: store,
		Dispatch: func(_ context.Context, _ Job) error {
			return nil
		},
	})
	if sErr != nil {
		t.Fatalf("new scheduler: %v", sErr)
	}

	fired, err := scheduler.RunDueJobs(context.Background())
	if err != nil {
		t.Fatalf("runDueJobs: %v", err)
	}
	if fired != 1 {
		t.Fatalf("expected 1 fired, got %d", fired)
	}

	// Verify the default engine computed a future nextRunAt.
	updated, _ := store.Get(context.Background(), job.ID)
	if updated.NextRunAt.Before(time.Now().UTC()) {
		t.Fatal("expected nextRunAt in the future with default engine")
	}
}

// stubCronEngine is a test double for the CronEngine interface.
type stubCronEngine struct {
	nextTime  time.Time
	valid     bool
	callCount int
}

func (s *stubCronEngine) NextOccurrence(_ string, _ time.Time) (time.Time, error) {
	s.callCount++
	return s.nextTime, nil
}

func (s *stubCronEngine) IsValid(_ string) bool {
	return s.valid
}

type testLogger struct{}

func (l *testLogger) Debug(_ string, _ ...map[string]string) {}
func (l *testLogger) Info(_ string, _ ...map[string]string)  {}
func (l *testLogger) Warn(_ string, _ ...map[string]string)  {}
func (l *testLogger) Error(_ string, _ ...map[string]string) {}

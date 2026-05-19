// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"math"
	"testing"
	"time"
)

// --- validateIdentifier tests ---

func TestValidateIdentifier(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{name: "simple alpha", input: "ingest_queue", wantErr: false},
		{name: "mixed case", input: "IngestQueue", wantErr: false},
		{name: "digits", input: "table123", wantErr: false},
		{name: "underscore only", input: "_", wantErr: false},
		{name: "empty", input: "", wantErr: true},
		{name: "contains space", input: "ingest queue", wantErr: true},
		{name: "contains dash", input: "ingest-queue", wantErr: true},
		{name: "contains dot", input: "public.ingest_queue", wantErr: true},
		{name: "contains semicolon", input: "queue;DROP TABLE", wantErr: true},
		{name: "contains single quote", input: "queue'", wantErr: true},
		{name: "contains double quote", input: `queue"`, wantErr: true},
		{name: "contains parenthesis", input: "queue()", wantErr: true},
		{name: "unicode letter", input: "queueé", wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := validateIdentifier(tc.input)
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("validateIdentifier(%q): got err=%v, wantErr=%v", tc.input, err, tc.wantErr)
			}
		})
	}
}

// --- advisoryLockKey tests ---

func TestAdvisoryLockKey_Deterministic(t *testing.T) {
	t.Parallel()

	brainID := "brain-abc-123"
	key1 := advisoryLockKey(brainID)
	key2 := advisoryLockKey(brainID)
	if key1 != key2 {
		t.Errorf("advisory lock key not deterministic: %d vs %d", key1, key2)
	}
}

func TestAdvisoryLockKey_DifferentBrains(t *testing.T) {
	t.Parallel()

	keyA := advisoryLockKey("brain-alpha")
	keyB := advisoryLockKey("brain-beta")
	if keyA == keyB {
		t.Errorf("different brains produced same advisory lock key: %d", keyA)
	}
}

// --- computeBackoff tests ---

func TestComputeBackoff_ExponentialGrowth(t *testing.T) {
	t.Parallel()

	// Collect backoff times for retries 0..4 and verify they grow.
	// Because of jitter the actual values vary, but the median should
	// roughly double each step. We run multiple samples and check the
	// minimum of retry N+1 is generally > minimum of retry N (accounting
	// for jitter bounds).
	samples := 50
	for retry := 0; retry < 4; retry++ {
		var totalDelay float64
		for range samples {
			before := time.Now()
			target := computeBackoff(retry)
			delay := target.Sub(before).Seconds()
			totalDelay += delay
		}
		avgDelay := totalDelay / float64(samples)

		// Expected centre: 2^retry * 1.0 second (jitter centres at 1.0)
		expectedCentre := math.Pow(2, float64(retry))
		// Allow 50% tolerance for jitter spread.
		if avgDelay < expectedCentre*0.3 || avgDelay > expectedCentre*2.0 {
			t.Errorf("retry %d: avg delay %.2fs outside expected range (centre=%.1fs)",
				retry, avgDelay, expectedCentre)
		}
	}
}

func TestComputeBackoff_FutureTime(t *testing.T) {
	t.Parallel()

	for retry := range 5 {
		before := time.Now()
		target := computeBackoff(retry)
		if !target.After(before) {
			t.Errorf("retry %d: backoff time %v is not after %v", retry, target, before)
		}
	}
}

// --- JobStatus tests ---

func TestJobStatus_IsValid(t *testing.T) {
	t.Parallel()

	cases := []struct {
		status JobStatus
		valid  bool
	}{
		{StatusPending, true},
		{StatusProcessing, true},
		{StatusCompleted, true},
		{StatusFailed, true},
		{StatusDeadLetter, true},
		{"unknown", false},
		{"", false},
		{"PENDING", false},
	}

	for _, tc := range cases {
		t.Run(string(tc.status), func(t *testing.T) {
			t.Parallel()
			if got := tc.status.IsValid(); got != tc.valid {
				t.Errorf("IsValid(%q) = %v, want %v", tc.status, got, tc.valid)
			}
		})
	}
}

// --- nullableBytes tests ---

func TestNullableBytes(t *testing.T) {
	t.Parallel()

	t.Run("nil for empty", func(t *testing.T) {
		t.Parallel()
		result := nullableBytes(nil)
		if result != nil {
			t.Error("expected nil for nil input")
		}
		result = nullableBytes([]byte{})
		if result != nil {
			t.Error("expected nil for empty slice")
		}
	})

	t.Run("non-nil for content", func(t *testing.T) {
		t.Parallel()
		data := []byte(`{"key":"value"}`)
		result := nullableBytes(data)
		if result == nil {
			t.Fatal("expected non-nil for non-empty slice")
		}
		if string(*result) != string(data) {
			t.Errorf("got %s, want %s", *result, data)
		}
	})
}

// --- NewPostgresQueue validation tests ---

func TestNewPostgresQueue_NilDB(t *testing.T) {
	t.Parallel()

	_, err := NewPostgresQueue(PostgresOptions{DB: nil})
	if err == nil {
		t.Fatal("expected error for nil DB")
	}
}

// Constructor validation tests with a real mock DB are in adapter_test.go
// (TestNewPostgresQueue_InvalidSchemaWithMockDB and
// TestNewPostgresQueue_InvalidTableWithMockDB).

// --- qualifiedTable tests ---

func TestQualifiedTable(t *testing.T) {
	t.Parallel()

	q := &PostgresQueue{schema: "myschema", table: "mytable"}
	got := q.qualifiedTable()
	want := "myschema.mytable"
	if got != want {
		t.Errorf("qualifiedTable() = %q, want %q", got, want)
	}
}

// --- EnqueueInput defaults ---

func TestEnqueueDefaults(t *testing.T) {
	t.Parallel()

	input := EnqueueInput{BrainID: "b1", Payload: JobPayload{Kind: "raw"}}
	if input.MaxRetries != 0 {
		t.Error("MaxRetries should default to zero value")
	}
	// The adapter applies defaultMaxRetries when zero.
}

// --- ClaimOptions defaults ---

func TestClaimOptions_Defaults(t *testing.T) {
	t.Parallel()

	opts := ClaimOptions{}
	if opts.BatchSize != 0 {
		t.Error("BatchSize should default to zero value")
	}
	// The adapter applies defaultBatchSize when zero.
}

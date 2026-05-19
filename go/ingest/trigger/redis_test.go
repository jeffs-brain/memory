// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockRedisSubscriber simulates a Redis pub/sub client for testing.
type mockRedisSubscriber struct {
	messages []string
}

func (m *mockRedisSubscriber) Subscribe(ctx context.Context, _ string, onMessage func(payload string)) error {
	for _, msg := range m.messages {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			onMessage(msg)
		}
	}
	// Block until cancelled to simulate long-running subscription.
	<-ctx.Done()
	return ctx.Err()
}

func TestRedisBridgeValidJSON(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	event := validEvent("redis-1")
	event.Source = SourceRedis
	data, _ := json.Marshal(event)

	mock := &mockRedisSubscriber{messages: []string{string(data)}}

	var received atomic.Int32
	var wg sync.WaitGroup
	wg.Add(1)
	bus.Subscribe(func(evt IngestTriggerEvent) error {
		if evt.ID == "redis-1" {
			received.Add(1)
			wg.Done()
		}
		return nil
	})

	bridge := NewRedisBridge(RedisBridgeOptions{
		Subscriber: mock,
		Bus:        bus,
	})
	defer bridge.Close()

	waitWithTimeout(t, &wg, 5*time.Second)
	if got := received.Load(); got != 1 {
		t.Fatalf("expected 1 event, got %d", got)
	}
}

func TestNewRedisBridge_NilSubscriberPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil Subscriber, got none")
		}
		msg, ok := r.(string)
		if !ok || msg != "trigger: NewRedisBridge requires a non-nil Subscriber" {
			t.Fatalf("unexpected panic value: %v", r)
		}
	}()

	bus := NewBus(nil)
	defer bus.Close()
	NewRedisBridge(RedisBridgeOptions{Subscriber: nil, Bus: bus})
}

func TestNewRedisBridge_NilBusPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil Bus, got none")
		}
		msg, ok := r.(string)
		if !ok || msg != "trigger: NewRedisBridge requires a non-nil Bus" {
			t.Fatalf("unexpected panic value: %v", r)
		}
	}()

	NewRedisBridge(RedisBridgeOptions{Subscriber: &mockRedisSubscriber{}, Bus: nil})
}

func TestRedisBridgeInvalidJSON(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	var warnings atomic.Int32
	var wg sync.WaitGroup
	wg.Add(1)
	logger := &testLogger{onWarn: func(_ string, _ ...map[string]string) {
		if warnings.Add(1) == 1 {
			wg.Done()
		}
	}}

	mock := &mockRedisSubscriber{messages: []string{"not-json"}}

	bridge := NewRedisBridge(RedisBridgeOptions{
		Subscriber: mock,
		Bus:        bus,
		Logger:     logger,
	})

	waitWithTimeout(t, &wg, 5*time.Second)
	bridge.Close()

	if got := warnings.Load(); got < 1 {
		t.Fatalf("expected at least 1 warning for invalid JSON, got %d", got)
	}
}

// waitWithTimeout waits for a WaitGroup with a bounded timeout to prevent
// CI stalls from blocking indefinitely on missing events.
func waitWithTimeout(t *testing.T, wg *sync.WaitGroup, timeout time.Duration) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		t.Fatal("timed out waiting for WaitGroup")
	}
}

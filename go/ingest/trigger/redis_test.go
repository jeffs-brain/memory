// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
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

	wg.Wait()
	if got := received.Load(); got != 1 {
		t.Fatalf("expected 1 event, got %d", got)
	}
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

	wg.Wait()
	bridge.Close()

	if got := warnings.Load(); got < 1 {
		t.Fatalf("expected at least 1 warning for invalid JSON, got %d", got)
	}
}

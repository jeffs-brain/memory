// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockPgListener simulates a PostgreSQL LISTEN/NOTIFY client.
type mockPgListener struct {
	payloads []string
	errOnce  error // if set, first call returns this error
}

func (m *mockPgListener) Listen(ctx context.Context, _ string, onNotify func(payload string)) error {
	if m.errOnce != nil {
		err := m.errOnce
		m.errOnce = nil
		return err
	}
	for _, p := range m.payloads {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			onNotify(p)
		}
	}
	<-ctx.Done()
	return ctx.Err()
}

func TestPostgresBridgeValidPayload(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	event := validEvent("pg-1")
	event.Source = SourcePostgres
	data, _ := json.Marshal(event)

	mock := &mockPgListener{payloads: []string{string(data)}}

	var received atomic.Int32
	var wg sync.WaitGroup
	wg.Add(1)
	bus.Subscribe(func(evt IngestTriggerEvent) error {
		if evt.ID == "pg-1" {
			received.Add(1)
			wg.Done()
		}
		return nil
	})

	bridge := NewPostgresBridge(PostgresBridgeOptions{
		Listener: mock,
		Bus:      bus,
	})
	defer bridge.Close()

	wg.Wait()
	if got := received.Load(); got != 1 {
		t.Fatalf("expected 1 event, got %d", got)
	}
}

func TestPostgresBridgeInvalidJSON(t *testing.T) {
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

	mock := &mockPgListener{payloads: []string{`{broken`}}
	bridge := NewPostgresBridge(PostgresBridgeOptions{
		Listener: mock,
		Bus:      bus,
		Logger:   logger,
	})

	wg.Wait()
	bridge.Close()

	if got := warnings.Load(); got < 1 {
		t.Fatalf("expected warning for invalid JSON, got %d", got)
	}
}

func TestPostgresBridgeReconnects(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	event := validEvent("pg-reconnect")
	data, _ := json.Marshal(event)

	mock := &mockPgListener{
		payloads: []string{string(data)},
		errOnce:  errors.New("connection dropped"),
	}

	var received atomic.Int32
	var wg sync.WaitGroup
	wg.Add(1)
	bus.Subscribe(func(evt IngestTriggerEvent) error {
		if evt.ID == "pg-reconnect" {
			received.Add(1)
			wg.Done()
		}
		return nil
	})

	bridge := NewPostgresBridge(PostgresBridgeOptions{
		Listener:       mock,
		Bus:            bus,
		Logger:         &testLogger{},
		ReconnectDelay: 10 * time.Millisecond,
	})
	defer bridge.Close()

	wg.Wait()
	if got := received.Load(); got != 1 {
		t.Fatalf("expected 1 event after reconnect, got %d", got)
	}
}

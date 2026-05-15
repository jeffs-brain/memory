// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func validEvent(id string) IngestTriggerEvent {
	return IngestTriggerEvent{
		ID:        id,
		BrainID:   "brain-1",
		Source:    SourceEventBus,
		Payload:   TriggerPayload{Kind: PayloadFile, Path: "/docs/readme.md"},
		Timestamp: time.Now(),
	}
}

func TestPublishSubscribeReceives(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	var received IngestTriggerEvent
	var wg sync.WaitGroup
	wg.Add(1)

	bus.Subscribe(func(event IngestTriggerEvent) error {
		received = event
		wg.Done()
		return nil
	})

	event := validEvent("e1")
	if err := bus.Publish(event); err != nil {
		t.Fatalf("publish: %v", err)
	}

	wg.Wait()
	if received.ID != "e1" {
		t.Fatalf("expected event e1, got %q", received.ID)
	}
}

func TestMultipleSubscribersReceiveSameEvent(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	var count atomic.Int32
	var wg sync.WaitGroup
	wg.Add(3)

	for range 3 {
		bus.Subscribe(func(_ IngestTriggerEvent) error {
			count.Add(1)
			wg.Done()
			return nil
		})
	}

	if err := bus.Publish(validEvent("e2")); err != nil {
		t.Fatalf("publish: %v", err)
	}

	wg.Wait()
	if got := count.Load(); got != 3 {
		t.Fatalf("expected 3 deliveries, got %d", got)
	}
}

func TestUnsubscribeStopsDelivery(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	var firstCount atomic.Int32
	var secondCount atomic.Int32

	// WaitGroup tracks when both handlers have processed the first event.
	var wg1 sync.WaitGroup
	wg1.Add(2)

	unsub := bus.Subscribe(func(_ IngestTriggerEvent) error {
		firstCount.Add(1)
		wg1.Done()
		return nil
	})

	// WaitGroup for second event (only second handler receives).
	var wg2 sync.WaitGroup
	wg2.Add(1)

	bus.Subscribe(func(_ IngestTriggerEvent) error {
		cur := secondCount.Add(1)
		if cur == 1 {
			wg1.Done()
		} else {
			wg2.Done()
		}
		return nil
	})

	// Publish once: both receive.
	if err := bus.Publish(validEvent("e3")); err != nil {
		t.Fatalf("publish: %v", err)
	}
	wg1.Wait()

	unsub()

	// Publish again: only second receives.
	if err := bus.Publish(validEvent("e4")); err != nil {
		t.Fatalf("publish: %v", err)
	}
	wg2.Wait()

	if got := firstCount.Load(); got != 1 {
		t.Fatalf("first handler: expected 1 delivery, got %d", got)
	}
	if got := secondCount.Load(); got != 2 {
		t.Fatalf("second handler: expected 2 deliveries, got %d", got)
	}
}

func TestQueueDepthExceeded(t *testing.T) {
	var warnings atomic.Int32
	logger := &testLogger{onWarn: func(_ string, _ ...map[string]string) {
		warnings.Add(1)
	}}

	bus := NewBus(&BusOptions{MaxQueueDepth: 2, Logger: logger})

	// Do not subscribe — events accumulate in queue.
	if err := bus.Publish(validEvent("q1")); err != nil {
		t.Fatalf("publish q1: %v", err)
	}
	if err := bus.Publish(validEvent("q2")); err != nil {
		t.Fatalf("publish q2: %v", err)
	}
	// Third publish should trigger backpressure (drop oldest).
	if err := bus.Publish(validEvent("q3")); err != nil {
		t.Fatalf("publish q3: %v", err)
	}

	bus.Close()

	if got := warnings.Load(); got < 1 {
		t.Fatalf("expected at least 1 warning, got %d", got)
	}
}

func TestMalformedEventRejected(t *testing.T) {
	bus := NewBus(nil)
	defer bus.Close()

	tests := []struct {
		name  string
		event IngestTriggerEvent
		err   error
	}{
		{
			name:  "missing id",
			event: IngestTriggerEvent{BrainID: "b", Payload: TriggerPayload{Kind: PayloadFile, Path: "/x"}, Timestamp: time.Now()},
			err:   ErrMissingID,
		},
		{
			name:  "missing brain id",
			event: IngestTriggerEvent{ID: "1", Payload: TriggerPayload{Kind: PayloadFile, Path: "/x"}, Timestamp: time.Now()},
			err:   ErrMissingBrainID,
		},
		{
			name:  "missing timestamp",
			event: IngestTriggerEvent{ID: "1", BrainID: "b", Payload: TriggerPayload{Kind: PayloadFile, Path: "/x"}},
			err:   ErrMissingTimestamp,
		},
		{
			name:  "invalid payload kind",
			event: IngestTriggerEvent{ID: "1", BrainID: "b", Payload: TriggerPayload{Kind: "banana"}, Timestamp: time.Now()},
			err:   ErrInvalidPayloadKind,
		},
		{
			name:  "file payload missing path",
			event: IngestTriggerEvent{ID: "1", BrainID: "b", Payload: TriggerPayload{Kind: PayloadFile}, Timestamp: time.Now()},
			err:   ErrMissingPayloadPath,
		},
		{
			name:  "url payload missing url",
			event: IngestTriggerEvent{ID: "1", BrainID: "b", Payload: TriggerPayload{Kind: PayloadURL}, Timestamp: time.Now()},
			err:   ErrMissingPayloadURL,
		},
		{
			name:  "raw payload missing content",
			event: IngestTriggerEvent{ID: "1", BrainID: "b", Payload: TriggerPayload{Kind: PayloadRaw}, Timestamp: time.Now()},
			err:   ErrMissingPayloadContent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := bus.Publish(tt.event)
			if !errors.Is(err, tt.err) {
				t.Fatalf("expected %v, got %v", tt.err, err)
			}
		})
	}
}

func TestCloseDrainsPending(t *testing.T) {
	bus := NewBus(nil)

	var received atomic.Int32
	bus.Subscribe(func(_ IngestTriggerEvent) error {
		received.Add(1)
		return nil
	})

	for i := range 5 {
		if err := bus.Publish(validEvent("drain-" + strconv.Itoa(i))); err != nil {
			t.Fatalf("publish: %v", err)
		}
	}

	bus.Close()

	if got := received.Load(); got != 5 {
		t.Fatalf("expected 5 drained events, got %d", got)
	}
}

func TestPublishAfterCloseReturnsError(t *testing.T) {
	bus := NewBus(nil)
	bus.Close()

	err := bus.Publish(validEvent("late"))
	if !errors.Is(err, ErrBusClosed) {
		t.Fatalf("expected ErrBusClosed, got %v", err)
	}
}

func TestHandlerErrorDoesNotRemoveSubscription(t *testing.T) {
	bus := NewBus(&BusOptions{Logger: &testLogger{}})
	defer bus.Close()

	var count atomic.Int32
	var wg sync.WaitGroup
	wg.Add(3)
	bus.Subscribe(func(_ IngestTriggerEvent) error {
		count.Add(1)
		wg.Done()
		return errors.New("handler fail")
	})

	for range 3 {
		if err := bus.Publish(validEvent("err")); err != nil {
			t.Fatalf("publish: %v", err)
		}
	}
	wg.Wait()

	if got := count.Load(); got != 3 {
		t.Fatalf("handler should still receive despite errors; got %d calls", got)
	}
}

// testLogger is a minimal Logger implementation for tests.
type testLogger struct {
	onWarn func(string, ...map[string]string)
}

func (l *testLogger) Debug(_ string, _ ...map[string]string) {}
func (l *testLogger) Info(_ string, _ ...map[string]string)  {}
func (l *testLogger) Warn(msg string, ctx ...map[string]string) {
	if l.onWarn != nil {
		l.onWarn(msg, ctx...)
	}
}
func (l *testLogger) Error(_ string, _ ...map[string]string) {}

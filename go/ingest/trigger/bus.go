// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"strconv"
	"sync"
	"sync/atomic"
)

const defaultMaxQueueDepth = 1000

// subscriber wraps a handler with an ID for removal and an optional
// event filter predicate.
type subscriber struct {
	id      string
	handler TriggerHandler
	filter  func(IngestTriggerEvent) bool
}

// inProcessBus is a channels-based event bus that dispatches events to
// registered handlers in-process. It is the default implementation and
// has zero external dependencies.
//
// Subscriber storage uses a map for O(1) unsubscribe and a copy-on-write
// snapshot slice for lock-free iteration during delivery.
type inProcessBus struct {
	mu            sync.Mutex
	subscribers   map[string]subscriber
	snapshot      []subscriber // copy-on-write: rebuilt on subscribe/unsubscribe
	nextID        uint64
	maxQueueDepth int
	logger        Logger
	closed        atomic.Bool

	eventCh chan IngestTriggerEvent
	wg      sync.WaitGroup
	done    chan struct{}
}

// NewBus creates an in-process event bus. All events are delivered to
// subscribers asynchronously via a buffered channel. When opts is nil,
// defaults are used.
func NewBus(opts *BusOptions) Bus {
	maxQ := defaultMaxQueueDepth
	var log Logger
	if opts != nil {
		if opts.MaxQueueDepth > 0 {
			maxQ = opts.MaxQueueDepth
		}
		log = opts.Logger
	}

	b := &inProcessBus{
		subscribers:   make(map[string]subscriber),
		maxQueueDepth: maxQ,
		logger:        log,
		eventCh:       make(chan IngestTriggerEvent, maxQ),
		done:          make(chan struct{}),
	}
	b.wg.Add(1)
	go b.dispatch()
	return b
}

// Publish validates and enqueues an event. Returns an error if the bus
// is closed or the event is malformed. When the queue is full, the
// oldest event is dropped and a warning is logged.
func (b *inProcessBus) Publish(event IngestTriggerEvent) error {
	if b.closed.Load() {
		return ErrBusClosed
	}
	if err := event.Validate(); err != nil {
		return err
	}

	select {
	case b.eventCh <- event:
		return nil
	default:
		// Queue full — drop oldest and enqueue new.
		select {
		case <-b.eventCh:
			if b.logger != nil {
				b.logger.Warn("trigger: queue full, dropping oldest event", map[string]string{
					"maxQueueDepth": strconv.Itoa(b.maxQueueDepth),
				})
			}
		default:
		}
		select {
		case b.eventCh <- event:
			return nil
		default:
			return ErrQueueFull
		}
	}
}

// Subscribe registers a handler. Returns a function to unsubscribe.
func (b *inProcessBus) Subscribe(handler TriggerHandler) (unsubscribe func()) {
	return b.SubscribeWithFilter(handler, SubscribeOptions{})
}

// SubscribeWithFilter registers a handler with an optional event filter.
// When opts.Filter is non-nil, only events for which the filter returns
// true are delivered. When opts.Filter is nil, all events are delivered.
func (b *inProcessBus) SubscribeWithFilter(handler TriggerHandler, opts SubscribeOptions) (unsubscribe func()) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.nextID++
	id := strconv.FormatUint(b.nextID, 10)
	b.subscribers[id] = subscriber{id: id, handler: handler, filter: opts.Filter}
	b.rebuildSnapshot()

	return func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		delete(b.subscribers, id)
		b.rebuildSnapshot()
	}
}

// Close signals the dispatch loop to drain remaining events and stop.
func (b *inProcessBus) Close() error {
	if b.closed.Swap(true) {
		return nil // already closed
	}
	close(b.done)
	b.wg.Wait()
	return nil
}

// dispatch runs in a goroutine and delivers events to all subscribers.
func (b *inProcessBus) dispatch() {
	defer b.wg.Done()
	for {
		select {
		case event := <-b.eventCh:
			b.deliverToAll(event)
		case <-b.done:
			// Drain remaining events.
			for {
				select {
				case event := <-b.eventCh:
					b.deliverToAll(event)
				default:
					return
				}
			}
		}
	}
}

// deliverToAll calls every registered handler with the event. Errors
// are logged but do not remove the handler. Uses the copy-on-write
// snapshot so no lock is held during delivery.
func (b *inProcessBus) deliverToAll(event IngestTriggerEvent) {
	b.mu.Lock()
	subs := b.snapshot
	b.mu.Unlock()

	for _, s := range subs {
		if s.filter != nil && !s.filter(event) {
			continue
		}
		if err := s.handler(event); err != nil {
			if b.logger != nil {
				b.logger.Error("trigger: handler error", map[string]string{
					"error":   err.Error(),
					"eventId": event.ID,
				})
			}
		}
	}
}

// rebuildSnapshot creates a fresh subscriber slice from the map.
// Must be called while b.mu is held.
func (b *inProcessBus) rebuildSnapshot() {
	snap := make([]subscriber, 0, len(b.subscribers))
	for _, s := range b.subscribers {
		snap = append(snap, s)
	}
	b.snapshot = snap
}

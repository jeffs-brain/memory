// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// PgListener abstracts the PostgreSQL LISTEN/NOTIFY interface so callers
// can inject any PostgreSQL driver (pgx, lib/pq, etc.) without a hard
// dependency.
type PgListener interface {
	// Listen blocks until the context is cancelled, calling onNotify for
	// each notification received on the channel.
	Listen(ctx context.Context, channel string, onNotify func(payload string)) error
}

// PostgresBridgeOptions configures the PostgreSQL LISTEN/NOTIFY bridge.
type PostgresBridgeOptions struct {
	// Listener is the PostgreSQL notification client.
	Listener PgListener
	// Channel is the PostgreSQL channel to LISTEN on. Defaults to "ingest_trigger".
	Channel string
	// Bus is the local event bus to forward parsed events to.
	Bus Bus
	// Logger receives diagnostic messages.
	Logger Logger
	// ReconnectDelay is the delay between reconnection attempts. Defaults to 5s.
	ReconnectDelay time.Duration
}

// PostgresBridge forwards events from PostgreSQL LISTEN/NOTIFY to a local Bus.
type PostgresBridge struct {
	opts   PostgresBridgeOptions
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewPostgresBridge creates and starts a PostgreSQL LISTEN/NOTIFY bridge.
// The bridge listens on the configured channel and publishes parsed events
// to the local bus. Call Close to stop.
func NewPostgresBridge(opts PostgresBridgeOptions) *PostgresBridge {
	if opts.Channel == "" {
		opts.Channel = "ingest_trigger"
	}
	if opts.ReconnectDelay == 0 {
		opts.ReconnectDelay = 5 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())
	b := &PostgresBridge{opts: opts, cancel: cancel}
	b.wg.Add(1)
	go b.listen(ctx)
	return b
}

// Close stops the bridge and waits for the listener to exit.
func (b *PostgresBridge) Close() error {
	b.cancel()
	b.wg.Wait()
	return nil
}

// listen runs the LISTEN loop with automatic reconnection on failure.
func (b *PostgresBridge) listen(ctx context.Context) {
	defer b.wg.Done()
	for {
		err := b.opts.Listener.Listen(ctx, b.opts.Channel, func(payload string) {
			var event IngestTriggerEvent
			if err := json.Unmarshal([]byte(payload), &event); err != nil {
				if b.opts.Logger != nil {
					b.opts.Logger.Warn("trigger/postgres: invalid JSON, discarding", map[string]string{
						"error":   err.Error(),
						"channel": b.opts.Channel,
					})
				}
				return
			}
			if event.Source == "" {
				event.Source = SourcePostgres
			}
			if err := b.opts.Bus.Publish(event); err != nil {
				if b.opts.Logger != nil {
					b.opts.Logger.Error("trigger/postgres: publish to bus failed", map[string]string{
						"error":   err.Error(),
						"eventId": event.ID,
					})
				}
			}
		})
		if ctx.Err() != nil {
			return
		}
		if err != nil && b.opts.Logger != nil {
			b.opts.Logger.Warn("trigger/postgres: listen error, reconnecting", map[string]string{
				"error": err.Error(),
				"delay": b.opts.ReconnectDelay.String(),
			})
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(b.opts.ReconnectDelay):
		}
	}
}

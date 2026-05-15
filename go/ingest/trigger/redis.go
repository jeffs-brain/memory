// SPDX-License-Identifier: Apache-2.0
package trigger

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"sync"
	"time"
)

// RedisSubscriber abstracts the Redis pub/sub interface so that callers
// can inject any Redis client (go-redis, redigo, etc.) without creating
// a hard dependency.
type RedisSubscriber interface {
	// Subscribe blocks until the context is cancelled, calling onMessage
	// for each message received on the channel.
	Subscribe(ctx context.Context, channel string, onMessage func(payload string)) error
}

// RedisBridgeOptions configures the Redis pub/sub bridge.
type RedisBridgeOptions struct {
	// Subscriber is the Redis pub/sub client.
	Subscriber RedisSubscriber
	// Channel is the Redis channel to subscribe to. Defaults to "ingest:trigger".
	Channel string
	// Bus is the local event bus to forward parsed events to.
	Bus Bus
	// Logger receives diagnostic messages.
	Logger Logger
	// ReconnectDelay is the base delay between reconnection attempts. Defaults to 5s.
	ReconnectDelay time.Duration
	// MaxReconnectDelay is the maximum backoff delay. Defaults to 30s.
	MaxReconnectDelay time.Duration
}

// RedisBridge forwards events from a Redis pub/sub channel to a local Bus.
type RedisBridge struct {
	opts   RedisBridgeOptions
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewRedisBridge creates and starts a Redis pub/sub bridge. The bridge
// listens on the configured channel and publishes parsed events to the
// local bus. Call Close to stop.
func NewRedisBridge(opts RedisBridgeOptions) *RedisBridge {
	if opts.Channel == "" {
		opts.Channel = "ingest:trigger"
	}
	if opts.ReconnectDelay == 0 {
		opts.ReconnectDelay = 5 * time.Second
	}
	if opts.MaxReconnectDelay == 0 {
		opts.MaxReconnectDelay = 30 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())
	b := &RedisBridge{opts: opts, cancel: cancel}
	b.wg.Add(1)
	go b.listen(ctx)
	return b
}

// Close stops the bridge and waits for the listener to exit.
func (b *RedisBridge) Close() error {
	b.cancel()
	b.wg.Wait()
	return nil
}

// listen runs the subscribe loop with automatic reconnection on failure.
// Uses exponential backoff with jitter to avoid thundering herd.
func (b *RedisBridge) listen(ctx context.Context) {
	defer b.wg.Done()
	attempt := 0
	for {
		err := b.opts.Subscriber.Subscribe(ctx, b.opts.Channel, func(payload string) {
			var event IngestTriggerEvent
			if err := json.Unmarshal([]byte(payload), &event); err != nil {
				if b.opts.Logger != nil {
					b.opts.Logger.Warn("trigger/redis: invalid JSON, discarding", map[string]string{
						"error":   err.Error(),
						"channel": b.opts.Channel,
					})
				}
				return
			}
			if event.Source == "" {
				event.Source = SourceRedis
			}
			if err := b.opts.Bus.Publish(event); err != nil {
				if b.opts.Logger != nil {
					b.opts.Logger.Error("trigger/redis: publish to bus failed", map[string]string{
						"error":   err.Error(),
						"eventId": event.ID,
					})
				}
			}
		})
		if ctx.Err() != nil {
			return
		}
		backoff := time.Duration(math.Min(
			float64(b.opts.ReconnectDelay)*math.Pow(2, float64(attempt)),
			float64(b.opts.MaxReconnectDelay),
		))
		jitteredDelay := time.Duration(float64(backoff) * (0.5 + rand.Float64()*0.5))
		if err != nil && b.opts.Logger != nil {
			b.opts.Logger.Warn("trigger/redis: subscription error, reconnecting", map[string]string{
				"error": err.Error(),
				"delay": jitteredDelay.String(),
			})
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(jitteredDelay):
		}
		attempt++
	}
}

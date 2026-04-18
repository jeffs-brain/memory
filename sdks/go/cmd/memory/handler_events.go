// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/internal/httpd"
)

// handleEvents subscribes to a brain's change events and streams
// them as SSE frames. Emits a `ready` event on attach, periodic
// `ping` keep-alives, and `change` events for every store mutation.
func (d *Daemon) handleEvents(w http.ResponseWriter, r *http.Request) {
	br := d.resolveBrain(w, r)
	if br == nil {
		return
	}
	stream := httpd.SSEStart(w)
	if stream == nil {
		return
	}
	if err := stream.SendRaw("ready", "ok"); err != nil {
		return
	}

	ctx := r.Context()
	events := make(chan brain.ChangeEvent, 64)
	unsubscribe := br.Store.Subscribe(brain.EventSinkFunc(func(evt brain.ChangeEvent) {
		select {
		case events <- evt:
		default:
			// Drop on slow consumer; the next refresh fetches fresh
			// state anyway.
		}
	}))
	defer unsubscribe()

	pingTicker := time.NewTicker(25 * time.Second)
	defer pingTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pingTicker.C:
			if err := stream.SendRaw("ping", "keepalive"); err != nil {
				return
			}
		case evt := <-events:
			payload := map[string]any{
				"kind": evt.Kind.String(),
				"path": string(evt.Path),
				"when": evt.When.UTC().Format(time.RFC3339Nano),
			}
			if evt.OldPath != "" {
				payload["old_path"] = string(evt.OldPath)
			}
			if evt.Reason != "" {
				payload["reason"] = evt.Reason
			}
			if err := stream.SendJSON("change", payload); err != nil {
				return
			}
		}
	}
}

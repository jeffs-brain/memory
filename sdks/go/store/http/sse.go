// SPDX-License-Identifier: Apache-2.0

package http

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// rawChangeEvent mirrors the wire shape documented in spec/PROTOCOL.md
// under the "change" SSE frame. Fields not present on the wire decode to
// their zero values; the caller coerces into [brain.ChangeEvent].
type rawChangeEvent struct {
	Kind    string `json:"kind"`
	Path    string `json:"path"`
	OldPath string `json:"old_path,omitempty"`
	Reason  string `json:"reason,omitempty"`
	When    string `json:"when"`
}

// toChangeEvent coerces the wire shape into the typed brain event. Unknown
// kinds map to zero so subscribers can branch on ChangeKind.String()
// without importing this package.
func (r rawChangeEvent) toChangeEvent() brain.ChangeEvent {
	var kind brain.ChangeKind
	switch r.Kind {
	case "created":
		kind = brain.ChangeCreated
	case "updated":
		kind = brain.ChangeUpdated
	case "deleted":
		kind = brain.ChangeDeleted
	case "renamed":
		kind = brain.ChangeRenamed
	}
	when, _ := time.Parse(time.RFC3339Nano, r.When)
	return brain.ChangeEvent{
		Kind:    kind,
		Path:    brain.Path(r.Path),
		OldPath: brain.Path(r.OldPath),
		Reason:  r.Reason,
		When:    when,
	}
}

// sseReader parses a text/event-stream body frame by frame. It stops when
// the reader hits EOF, the context is cancelled, or the caller closes the
// response. Malformed frames are dropped to keep long-lived streams
// resilient, matching the TS reference client.
type sseReader struct {
	body   io.ReadCloser
	scan   *bufio.Scanner
	closed atomic.Bool
}

func newSSEReader(body io.ReadCloser) *sseReader {
	s := bufio.NewScanner(body)
	// Default bufio.Scanner buffer is 64 KiB. Change events are tiny
	// (under 1 KiB) but we budget generously so a future reason field or
	// path cannot truncate a frame.
	s.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &sseReader{body: body, scan: s}
}

// readFrame returns the next (event, data) tuple. Returns io.EOF when the
// underlying stream closes cleanly. Comment lines (":...") and unknown
// field names are ignored per the SSE spec.
func (r *sseReader) readFrame() (event string, data string, err error) {
	var dataLines []string
	for r.scan.Scan() {
		line := r.scan.Text()
		if line == "" {
			if event == "" && len(dataLines) == 0 {
				continue
			}
			return event, strings.Join(dataLines, "\n"), nil
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		colon := strings.Index(line, ":")
		field := line
		value := ""
		if colon >= 0 {
			field = line[:colon]
			value = line[colon+1:]
			if strings.HasPrefix(value, " ") {
				value = value[1:]
			}
		}
		switch field {
		case "event":
			event = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}
	if err := r.scan.Err(); err != nil {
		return "", "", err
	}
	return "", "", io.EOF
}

// close drops the underlying connection so the server-side goroutine can
// exit. Safe to call concurrently and multiple times; only the first caller
// actually closes the body.
func (r *sseReader) close() error {
	if !r.closed.CompareAndSwap(false, true) {
		return nil
	}
	return r.body.Close()
}

// runSubscription holds the goroutine that feeds events into a fan-out
// dispatcher. It exits on context cancellation or when the stream closes.
// All errors are surfaced via the onError hook; callers that want to swallow
// transient errors pass a no-op.
func runSubscription(
	ctx context.Context,
	client *http.Client,
	url string,
	authHeader string,
	userAgent string,
	onEvent func(brain.ChangeEvent),
	onError func(error),
) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		onError(err)
		return
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("User-Agent", userAgent)
	if authHeader != "" {
		req.Header.Set("Authorization", authHeader)
	}
	resp, err := client.Do(req)
	if err != nil {
		onError(err)
		return
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		onError(responseError(resp))
		return
	}
	reader := newSSEReader(resp.Body)
	defer func() {
		_ = reader.close()
	}()
	// Close the body as soon as the context cancels so any in-flight
	// Scanner.Scan unblocks. The request context alone should achieve this,
	// but some transports buffer up the next read and miss the cancellation
	// signal closing the body is a belt-and-braces fallback.
	watchDone := make(chan struct{})
	defer close(watchDone)
	go func() {
		select {
		case <-ctx.Done():
			_ = reader.close()
		case <-watchDone:
		}
	}()
	for {
		event, data, err := reader.readFrame()
		if err != nil {
			if err != io.EOF && ctx.Err() == nil {
				onError(err)
			}
			return
		}
		if event != "change" {
			continue
		}
		var raw rawChangeEvent
		if err := json.Unmarshal([]byte(data), &raw); err != nil {
			// Malformed change payload drop and continue so the stream
			// stays resilient.
			continue
		}
		onEvent(raw.toChangeEvent())
	}
}

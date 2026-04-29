// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// SSEWriter is a small wrapper that emits text/event-stream frames
// and flushes after every write. Callers use [SSEStart] to set the
// canonical headers and obtain a writer.
type SSEWriter struct {
	w  http.ResponseWriter
	fl http.Flusher
}

// SSEStart sets the SSE response headers and returns a writer ready
// for [SSEWriter.SendJSON] / [SSEWriter.SendRaw]. Returns nil and
// writes a 500 Problem+JSON when the response writer cannot stream.
func SSEStart(w http.ResponseWriter) *SSEWriter {
	flusher, ok := w.(http.Flusher)
	if !ok {
		InternalError(w, "streaming not supported")
		return nil
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return &SSEWriter{w: w, fl: flusher}
}

// SendRaw emits an SSE frame with the given event and pre-encoded
// data. Returns the underlying error so callers can detect client
// disconnect.
func (s *SSEWriter) SendRaw(event, data string) error {
	if s == nil {
		return io.ErrClosedPipe
	}
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data); err != nil {
		return err
	}
	s.fl.Flush()
	return nil
}

// SendJSON marshals payload and emits an SSE frame.
func (s *SSEWriter) SendJSON(event string, payload any) error {
	if s == nil {
		return io.ErrClosedPipe
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.SendRaw(event, string(data))
}

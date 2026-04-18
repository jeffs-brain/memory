// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/jeffs-brain/memory/go/internal/httpd"
)

// errBodyTooLarge is returned by readLimitedBody when the request
// body exceeds the supplied cap.
var errBodyTooLarge = errors.New("request body too large")

// readLimitedBody reads up to limit bytes from r.Body and returns
// [errBodyTooLarge] if the limit is exceeded. The actual body is
// always drained.
func readLimitedBody(r *http.Request, limit int64) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	defer func() { _ = r.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, errBodyTooLarge
	}
	return body, nil
}

// decodeJSONBody decodes the request body into target with a size
// cap. Returns a wrapped error when the body is too large or the
// JSON is malformed.
func decodeJSONBody(r *http.Request, target any, limit int64) error {
	body, err := readLimitedBody(r, limit)
	if err != nil {
		if errors.Is(err, errBodyTooLarge) {
			return fmt.Errorf("body exceeds %d bytes", limit)
		}
		return err
	}
	if len(body) == 0 {
		return fmt.Errorf("empty body")
	}
	if err := json.Unmarshal(body, target); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

// resolveBrain pulls the brainId path value and looks it up via the
// daemon's BrainManager. Writes the appropriate Problem+JSON and
// returns nil when the brain is missing.
func (d *Daemon) resolveBrain(w http.ResponseWriter, r *http.Request) *BrainResources {
	id := r.PathValue("brainId")
	if id == "" {
		httpd.ValidationError(w, "missing brainId")
		return nil
	}
	br, err := d.Brains.Get(r.Context(), id)
	if err != nil {
		if errors.Is(err, ErrBrainNotFound) {
			httpd.NotFound(w, "brain not found: "+id)
			return nil
		}
		httpd.InternalError(w, err.Error())
		return nil
	}
	return br
}

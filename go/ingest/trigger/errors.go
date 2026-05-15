// SPDX-License-Identifier: Apache-2.0
package trigger

import "errors"

var (
	ErrMissingID             = errors.New("trigger: event ID is required")
	ErrMissingBrainID        = errors.New("trigger: event brainId is required")
	ErrMissingTimestamp       = errors.New("trigger: event timestamp is required")
	ErrMissingPayloadPath    = errors.New("trigger: file payload requires a path")
	ErrMissingPayloadURL     = errors.New("trigger: url payload requires a URL")
	ErrMissingPayloadContent = errors.New("trigger: raw payload requires content")
	ErrInvalidPayloadKind    = errors.New("trigger: payload kind must be file, url, or raw")
	ErrBusClosed             = errors.New("trigger: bus is closed")
	ErrQueueFull             = errors.New("trigger: queue depth exceeded, event dropped")
)

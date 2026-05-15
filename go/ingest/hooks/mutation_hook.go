// SPDX-License-Identifier: Apache-2.0
package hooks

import (
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

const defaultDebounceInterval = 1 * time.Second

// NewMutationHook creates a MutationHook and returns it along with an
// EventSink that should be subscribed to the store. Call Close when done
// to release timers and goroutines.
func NewMutationHook(opts MutationHookOptions) *MutationHook {
	if opts.DebounceInterval == 0 {
		opts.DebounceInterval = defaultDebounceInterval
	}
	if len(opts.PathMatchers) == 0 {
		opts.PathMatchers = []PathMatcher{DefaultPathMatcher}
	}
	if opts.OptOutReasons == nil {
		opts.OptOutReasons = map[string]bool{}
	}

	return &MutationHook{
		opts:   opts,
		timers: make(map[string]*time.Timer),
		closed: make(chan struct{}),
	}
}

// Sink returns an EventSink adapter that feeds change events into the
// mutation hook. This is the value you pass to Store.Subscribe.
func (h *MutationHook) Sink() brain.EventSink {
	return brain.EventSinkFunc(func(evt brain.ChangeEvent) {
		h.handleEvent(evt)
	})
}

// Close stops all pending debounce timers.
func (h *MutationHook) Close() {
	select {
	case <-h.closed:
		return
	default:
		close(h.closed)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for path, timer := range h.timers {
		timer.Stop()
		delete(h.timers, path)
	}
}

func (h *MutationHook) handleEvent(evt brain.ChangeEvent) {
	select {
	case <-h.closed:
		return
	default:
	}

	// Only react to created and updated events.
	if evt.Kind != brain.ChangeCreated && evt.Kind != brain.ChangeUpdated {
		return
	}

	path := string(evt.Path)

	// Reject paths containing ".." to prevent path traversal.
	if strings.Contains(path, "..") {
		if h.opts.Logger != nil {
			h.opts.Logger.Warn("hooks: rejecting path with traversal segment", map[string]string{
				"path": path,
			})
		}
		return
	}

	// Opt-out by batch reason.
	if evt.Reason != "" && h.opts.OptOutReasons[evt.Reason] {
		if h.opts.Logger != nil {
			h.opts.Logger.Debug("hooks: skipping event due to opt-out reason", map[string]string{
				"path":   path,
				"reason": evt.Reason,
			})
		}
		return
	}

	// Check path matchers.
	if !h.matchesAny(path) {
		return
	}

	h.debounceDispatch(path)
}

func (h *MutationHook) matchesAny(path string) bool {
	for _, matcher := range h.opts.PathMatchers {
		if matcher(path) {
			return true
		}
	}
	return false
}

func (h *MutationHook) debounceDispatch(path string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Reset existing timer for this path if it has not already fired.
	if entryI, ok := h.timers[path]; ok {
		entryI.Stop()
	}

	h.timers[path] = time.AfterFunc(h.opts.DebounceInterval, func() {
		h.mu.Lock()
		delete(h.timers, path)
		h.mu.Unlock()

		// Guard against dispatch after Close.
		select {
		case <-h.closed:
			return
		default:
		}

		if err := h.opts.Dispatch(h.opts.BrainID, path); err != nil {
			if h.opts.Logger != nil {
				h.opts.Logger.Error("hooks: dispatch failed", map[string]string{
					"path":  path,
					"error": err.Error(),
				})
			}
		}
	})
}

// PrefixPathMatcher returns a PathMatcher that matches paths starting
// with the given prefix. The bare prefix itself is not matched; the
// path must contain at least one additional character after the prefix.
func PrefixPathMatcher(prefix string) PathMatcher {
	return func(path string) bool {
		return len(path) > len(prefix) && strings.HasPrefix(path, prefix)
	}
}

// GlobPathMatcher returns a PathMatcher that matches paths against a
// simple glob pattern where * matches any non-slash characters and **
// matches any characters including slashes.
func GlobPathMatcher(pattern string) PathMatcher {
	return func(path string) bool {
		return globMatch(pattern, path)
	}
}

// globMatch implements basic glob matching supporting * and **.
func globMatch(pattern, name string) bool {
	return matchAt(pattern, 0, name, 0)
}

func matchAt(pattern string, pi int, name string, ni int) bool {
	for pi < len(pattern) && ni < len(name) {
		if pi+1 < len(pattern) && pattern[pi] == '*' && pattern[pi+1] == '*' {
			// ** matches any characters including slashes.
			pi += 2
			if pi < len(pattern) && pattern[pi] == '/' {
				pi++
			}
			for k := ni; k <= len(name); k++ {
				if matchAt(pattern, pi, name, k) {
					return true
				}
			}
			return false
		}
		if pattern[pi] == '*' {
			pi++
			for k := ni; k <= len(name); k++ {
				if k > ni && name[k-1] == '/' {
					break
				}
				if matchAt(pattern, pi, name, k) {
					return true
				}
			}
			return false
		}
		if pattern[pi] == '?' {
			if name[ni] == '/' {
				return false
			}
			pi++
			ni++
			continue
		}
		if pattern[pi] != name[ni] {
			return false
		}
		pi++
		ni++
	}
	for pi < len(pattern) && pattern[pi] == '*' {
		pi++
	}
	return pi == len(pattern) && ni == len(name)
}

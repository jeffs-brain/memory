// SPDX-License-Identifier: Apache-2.0

package brain

import "errors"

// Sentinel errors returned by [Store] implementations. All backends must
// wrap these (never replace) so callers can use errors.Is. Backends
// typically wrap with path context using
// fmt.Errorf("brain: read %s: %w", p, brain.ErrNotFound).
var (
	// ErrNotFound is returned when the requested [Path] does not exist.
	ErrNotFound = errors.New("brain: not found")

	// ErrConflict is returned by transactional writes when concurrent
	// activity (another process, another machine via git) prevents a safe
	// commit.
	ErrConflict = errors.New("brain: concurrent modification")

	// ErrReadOnly is returned when the store is in a read-only state, for
	// example after Close has been called.
	ErrReadOnly = errors.New("brain: read-only")

	// ErrInvalidPath is returned when a [Path] violates conventions
	// (leading slash, ".." component, backslashes, outside known roots).
	ErrInvalidPath = errors.New("brain: invalid path")

	// ErrSchemaVersion is returned when the brain's schema version is
	// newer than the running binary understands. The store must not be
	// modified until the binary is upgraded.
	ErrSchemaVersion = errors.New("brain: unsupported schema version")

	// ErrUnauthorized maps to HTTP 401. The caller supplied no credentials
	// or an unparseable Authorization header.
	ErrUnauthorized = errors.New("brain: unauthorized")

	// ErrForbidden maps to HTTP 403. The authenticated principal lacks the
	// scope or RBAC right for the requested operation.
	ErrForbidden = errors.New("brain: forbidden")

	// ErrValidation maps to HTTP 400 and covers schema violations that are
	// not path-shape issues. Backends should prefer [ErrInvalidPath] when
	// the failure is solely about path validation.
	ErrValidation = errors.New("brain: validation error")

	// ErrPayloadTooLarge maps to HTTP 413 and is also returned by client
	// backends when a caller exceeds the on-wire body size limit before
	// the request leaves the process.
	ErrPayloadTooLarge = errors.New("brain: payload too large")

	// ErrUnsupportedMedia maps to HTTP 415.
	ErrUnsupportedMedia = errors.New("brain: unsupported media type")

	// ErrRateLimited maps to HTTP 429. Backends that honour a Retry-After
	// header do so transparently; callers see the wrapped sentinel only
	// when retry budget is exhausted.
	ErrRateLimited = errors.New("brain: rate limited")

	// ErrInternal maps to HTTP 500. Unhandled server-side failure.
	ErrInternal = errors.New("brain: internal error")

	// ErrBadGateway maps to HTTP 502. Upstream store returned an unusable
	// response.
	ErrBadGateway = errors.New("brain: bad gateway")

	// ErrTimeout maps to HTTP 504. Server-side deadline exceeded.
	ErrTimeout = errors.New("brain: timeout")
)

// SPDX-License-Identifier: Apache-2.0

package git

import (
	"context"
	"errors"

	gogit "github.com/go-git/go-git/v5"
)

// push pushes the current branch to origin. Errors are returned
// verbatim so callers can distinguish network failures from merge
// conflicts.
func (s *Store) push(ctx context.Context) error {
	return s.repo.PushContext(ctx, &gogit.PushOptions{
		RemoteName: defaultRemoteName,
		Force:      false,
	})
}

// Push performs an explicit push. Exposed for callers that want to
// force a sync after a batch of offline work.
func (s *Store) Push(ctx context.Context) error {
	if s.opts.RemoteURL == "" {
		return errors.New("gitstore: no remote configured")
	}
	return s.push(ctx)
}

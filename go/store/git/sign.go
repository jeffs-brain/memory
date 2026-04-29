// SPDX-License-Identifier: Apache-2.0

package git

// GitSignPayload is handed to a [GitSignFn] for every commit the gitstore
// produces. Payload holds the UTF-8 serialised commit object that git
// would normally hash to produce the commit SHA.
//
// Callers own key material: the SDK never sees the private key. The
// callback returns the detached signature as an ASCII-armored string
// (PGP or SSH block) which is embedded verbatim in the commit header.
type GitSignPayload struct {
	Payload string
}

// GitSignFn signs a commit payload and returns the detached armored
// signature. Invoked on every batch commit and on the init commit
// produced when the gitstore bootstraps a brand-new repository. Not
// invoked on tag creation or on [Store.Push] in v1.0.
type GitSignFn func(payload GitSignPayload) (string, error)

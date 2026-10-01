// SPDX-License-Identifier: Apache-2.0

// Package version holds the Go module's release version. The memory CLI
// and the memory-mcp server both report it, so the two cannot drift, and
// the release workflow refuses a tag that does not match it.
package version

// Version is the released version of the Go module.
const Version = "1.2.0"

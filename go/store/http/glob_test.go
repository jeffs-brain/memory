// SPDX-License-Identifier: Apache-2.0

package http

import (
	"fmt"
	stdpath "path"
)

// stdlibPathMatch is a tiny wrapper over path.Match, kept in its own file
// so the main test file can keep the net/http import clean of shadowing.
func stdlibPathMatch(pattern, name string) (bool, error) {
	return stdpath.Match(pattern, name)
}

// stdlibSprintf is a pinhole into fmt.Sprintf used by the conformance
// harness; declared here so the conformance file can remain free of an
// additional fmt import on top of its existing encoding imports.
func stdlibSprintf(format string, args ...any) string {
	return fmt.Sprintf(format, args...)
}

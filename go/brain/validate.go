// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"fmt"
	"regexp"
	"strings"
)

var validBrainIDPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$`)

// ValidateBrainID returns an error if id is not a safe brain identifier.
// A valid brain ID starts with an alphanumeric character, contains only
// [a-zA-Z0-9._-], is between 1 and 128 characters long, and does not
// contain path traversal sequences.
func ValidateBrainID(id string) error {
	if id == "" {
		return fmt.Errorf("brain: ID must not be empty")
	}
	if strings.Contains(id, "..") {
		return fmt.Errorf("brain: ID must not contain '..'")
	}
	if !validBrainIDPattern.MatchString(id) {
		return fmt.Errorf("brain: ID must start with alphanumeric, contain only [a-zA-Z0-9._-], max 128 chars")
	}
	return nil
}

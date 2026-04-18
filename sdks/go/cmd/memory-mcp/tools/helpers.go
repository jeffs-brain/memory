// SPDX-License-Identifier: Apache-2.0

package tools

// ptrInt returns a pointer to an int literal for JSON schema fields
// that expect *int (MinLength, MaxLength, MinItems, MaxItems, etc).
func ptrInt(v int) *int { return &v }

// ptrFloat returns a pointer to a float64 literal for JSON schema
// fields that expect *float64 (Minimum, Maximum, ...).
func ptrFloat(v float64) *float64 { return &v }

// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"strings"
	"time"
	"unicode"
)

// StateKey identifies a mutable attribute, such as a user's city,
// preferred editor, current project status, or recurring schedule.
type StateKey string

// ClaimKey identifies an asserted proposition whose truth may change or be
// disputed independently from other claims.
type ClaimKey string

// ValidityWindow describes when a state or claim is true. A zero endpoint is
// open, so an all-zero window means "currently or generally valid".
type ValidityWindow struct {
	ValidFrom  time.Time
	ValidUntil time.Time
}

// StateMetadata is a lightweight, storage-agnostic description of a mutable
// state value.
type StateMetadata struct {
	Key      StateKey
	Value    string
	Validity ValidityWindow
}

// ClaimMetadata is a lightweight, storage-agnostic description of a claim.
// Value is intentionally generic so callers can encode domain-specific
// polarity, status, or asserted value without this package understanding it.
type ClaimMetadata struct {
	Key      ClaimKey
	Value    string
	Validity ValidityWindow
}

// SupersessionDecision explains whether a newer metadata record should retire
// an older one.
type SupersessionDecision struct {
	Supersedes bool
	Reason     string
}

// NormaliseStateKey turns a free-form state key into a stable lower-case token.
func NormaliseStateKey(raw string) StateKey {
	return StateKey(normaliseMetadataKey(raw))
}

// NormaliseClaimKey turns a free-form claim key into a stable lower-case token.
func NormaliseClaimKey(raw string) ClaimKey {
	return ClaimKey(normaliseMetadataKey(raw))
}

// Normalised returns the key after applying state-key normalisation again.
func (k StateKey) Normalised() StateKey {
	return NormaliseStateKey(string(k))
}

// Normalised returns the key after applying claim-key normalisation again.
func (k ClaimKey) Normalised() ClaimKey {
	return NormaliseClaimKey(string(k))
}

// IsZero reports whether the window has no explicit bounds.
func (w ValidityWindow) IsZero() bool {
	return w.ValidFrom.IsZero() && w.ValidUntil.IsZero()
}

// IsValid reports whether the window bounds are coherent.
func (w ValidityWindow) IsValid() bool {
	return w.ValidUntil.IsZero() || w.ValidFrom.IsZero() || !w.ValidUntil.Before(w.ValidFrom)
}

// Contains reports whether t falls inside the window. Open endpoints are
// inclusive.
func (w ValidityWindow) Contains(t time.Time) bool {
	if t.IsZero() || !w.IsValid() {
		return false
	}
	if !w.ValidFrom.IsZero() && t.Before(w.ValidFrom) {
		return false
	}
	if !w.ValidUntil.IsZero() && t.After(w.ValidUntil) {
		return false
	}
	return true
}

// Overlaps reports whether two validity windows could refer to the same
// interval. Open-ended windows are treated as unbounded.
func (w ValidityWindow) Overlaps(other ValidityWindow) bool {
	if !w.IsValid() || !other.IsValid() {
		return false
	}

	if !w.ValidUntil.IsZero() && !other.ValidFrom.IsZero() && w.ValidUntil.Before(other.ValidFrom) {
		return false
	}
	if !other.ValidUntil.IsZero() && !w.ValidFrom.IsZero() && other.ValidUntil.Before(w.ValidFrom) {
		return false
	}

	return true
}

// StateSupersedes returns true only when newer clearly replaces older for the
// same normalised state key.
func StateSupersedes(newer, older StateMetadata) SupersessionDecision {
	newKey := newer.Key.Normalised()
	oldKey := older.Key.Normalised()
	if newKey == "" || oldKey == "" {
		return SupersessionDecision{Reason: "empty-key"}
	}
	if newKey != oldKey {
		return SupersessionDecision{Reason: "different-key"}
	}

	return decideSameKeySupersession(newer.Value, newer.Validity, older.Value, older.Validity)
}

// ClaimSupersedes returns true only when newer clearly replaces older for the
// same normalised claim key.
func ClaimSupersedes(newer, older ClaimMetadata) SupersessionDecision {
	newKey := newer.Key.Normalised()
	oldKey := older.Key.Normalised()
	if newKey == "" || oldKey == "" {
		return SupersessionDecision{Reason: "empty-key"}
	}
	if newKey != oldKey {
		return SupersessionDecision{Reason: "different-key"}
	}

	return decideSameKeySupersession(newer.Value, newer.Validity, older.Value, older.Validity)
}

func decideSameKeySupersession(newValue string, newWindow ValidityWindow, oldValue string, oldWindow ValidityWindow) SupersessionDecision {
	if !newWindow.IsValid() || !oldWindow.IsValid() {
		return SupersessionDecision{Reason: "invalid-window"}
	}
	if !newWindow.Overlaps(oldWindow) {
		return SupersessionDecision{Reason: "disjoint-validity"}
	}
	if normaliseMetadataValue(newValue) == "" || normaliseMetadataValue(oldValue) == "" {
		return SupersessionDecision{Reason: "empty-value"}
	}
	if normaliseMetadataValue(newValue) == normaliseMetadataValue(oldValue) {
		return SupersessionDecision{Reason: "same-value"}
	}
	return SupersessionDecision{Supersedes: true, Reason: "same-key-different-value"}
}

func normaliseMetadataKey(raw string) string {
	raw = strings.TrimSpace(strings.ToLower(raw))
	if raw == "" {
		return ""
	}

	var b strings.Builder
	lastWasSeparator := false
	for _, r := range raw {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
			lastWasSeparator = false
		case r == '.' || r == '/' || r == ':' || r == '-' || r == '_' || unicode.IsSpace(r):
			if b.Len() > 0 && !lastWasSeparator {
				b.WriteByte('.')
				lastWasSeparator = true
			}
		default:
			if b.Len() > 0 && !lastWasSeparator {
				b.WriteByte('.')
				lastWasSeparator = true
			}
		}
	}

	return strings.Trim(b.String(), ".")
}

func normaliseMetadataValue(raw string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(raw))), " ")
}

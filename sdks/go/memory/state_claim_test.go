// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"testing"
	"time"
)

func TestNormaliseStateKey(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want StateKey
	}{
		{
			name: "lowercases and collapses separators",
			raw:  " User / Profile: Current_City ",
			want: "user.profile.current.city",
		},
		{
			name: "removes leading and trailing separators",
			raw:  " -- schedule.weekly-gym! ",
			want: "schedule.weekly.gym",
		},
		{
			name: "keeps digits",
			raw:  "Device 2 Status",
			want: "device.2.status",
		},
		{
			name: "empty stays empty",
			raw:  " \t\n ",
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NormaliseStateKey(tt.raw); got != tt.want {
				t.Fatalf("NormaliseStateKey(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestNormaliseClaimKey(t *testing.T) {
	got := NormaliseClaimKey(" Immigration Claim / Status ")
	want := ClaimKey("immigration.claim.status")
	if got != want {
		t.Fatalf("NormaliseClaimKey = %q, want %q", got, want)
	}

	if got.Normalised() != want {
		t.Fatalf("Normalised = %q, want %q", got.Normalised(), want)
	}
}

func TestValidityWindow(t *testing.T) {
	start := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2025, 2, 1, 0, 0, 0, 0, time.UTC)
	mid := time.Date(2025, 1, 15, 0, 0, 0, 0, time.UTC)

	window := ValidityWindow{ValidFrom: start, ValidUntil: end}
	if !window.IsValid() {
		t.Fatal("expected coherent window to be valid")
	}
	if !window.Contains(mid) {
		t.Fatal("expected midpoint to be inside window")
	}
	if window.Contains(start.Add(-time.Nanosecond)) {
		t.Fatal("expected time before start to be outside window")
	}
	if window.Contains(end.Add(time.Nanosecond)) {
		t.Fatal("expected time after end to be outside window")
	}

	openEnded := ValidityWindow{ValidFrom: start}
	if !openEnded.Contains(end.Add(24 * time.Hour)) {
		t.Fatal("expected open-ended window to contain later time")
	}

	invalid := ValidityWindow{ValidFrom: end, ValidUntil: start}
	if invalid.IsValid() {
		t.Fatal("expected reversed window to be invalid")
	}
	if invalid.Contains(mid) {
		t.Fatal("invalid window must not contain times")
	}
}

func TestValidityWindowOverlaps(t *testing.T) {
	jan := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	feb := time.Date(2025, 2, 1, 0, 0, 0, 0, time.UTC)
	mar := time.Date(2025, 3, 1, 0, 0, 0, 0, time.UTC)

	if !(ValidityWindow{ValidFrom: jan, ValidUntil: feb}).Overlaps(ValidityWindow{ValidFrom: feb, ValidUntil: mar}) {
		t.Fatal("touching inclusive windows should overlap")
	}
	if (ValidityWindow{ValidFrom: jan, ValidUntil: feb}).Overlaps(ValidityWindow{ValidFrom: feb.Add(time.Nanosecond), ValidUntil: mar}) {
		t.Fatal("separated windows must not overlap")
	}
	if !(ValidityWindow{ValidFrom: feb}).Overlaps(ValidityWindow{ValidUntil: mar}) {
		t.Fatal("open-ended windows with shared range should overlap")
	}
}

func TestStateSupersedes(t *testing.T) {
	old := StateMetadata{
		Key:   "profile.current.city",
		Value: "Amsterdam",
	}
	newer := StateMetadata{
		Key:   " Profile / Current City ",
		Value: "Amersfoort",
	}

	decision := StateSupersedes(newer, old)
	if !decision.Supersedes {
		t.Fatalf("expected supersession, got reason %q", decision.Reason)
	}
	if decision.Reason != "same-key-different-value" {
		t.Fatalf("reason = %q, want same-key-different-value", decision.Reason)
	}
}

func TestStateSupersedesIsConservative(t *testing.T) {
	jan := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	feb := time.Date(2025, 2, 1, 0, 0, 0, 0, time.UTC)
	mar := time.Date(2025, 3, 1, 0, 0, 0, 0, time.UTC)
	apr := time.Date(2025, 4, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name  string
		newer StateMetadata
		older StateMetadata
		want  string
	}{
		{
			name:  "different keys",
			newer: StateMetadata{Key: "profile.city", Value: "Amersfoort"},
			older: StateMetadata{Key: "profile.country", Value: "Netherlands"},
			want:  "different-key",
		},
		{
			name:  "same comparable value",
			newer: StateMetadata{Key: "profile.city", Value: "  AMERSFOORT "},
			older: StateMetadata{Key: "profile.city", Value: "Amersfoort"},
			want:  "same-value",
		},
		{
			name: "disjoint validity",
			newer: StateMetadata{
				Key:      "schedule.gym",
				Value:    "Wednesdays",
				Validity: ValidityWindow{ValidFrom: mar, ValidUntil: apr},
			},
			older: StateMetadata{
				Key:      "schedule.gym",
				Value:    "Tuesdays",
				Validity: ValidityWindow{ValidFrom: jan, ValidUntil: feb},
			},
			want: "disjoint-validity",
		},
		{
			name: "empty value",
			newer: StateMetadata{
				Key:   "profile.city",
				Value: "",
			},
			older: StateMetadata{
				Key:   "profile.city",
				Value: "Amersfoort",
			},
			want: "empty-value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := StateSupersedes(tt.newer, tt.older)
			if decision.Supersedes {
				t.Fatalf("expected no supersession")
			}
			if decision.Reason != tt.want {
				t.Fatalf("reason = %q, want %q", decision.Reason, tt.want)
			}
		})
	}
}

func TestClaimSupersedes(t *testing.T) {
	old := ClaimMetadata{
		Key:   "application.status",
		Value: "pending",
	}
	newer := ClaimMetadata{
		Key:   "Application Status",
		Value: "approved",
	}

	decision := ClaimSupersedes(newer, old)
	if !decision.Supersedes {
		t.Fatalf("expected claim supersession, got reason %q", decision.Reason)
	}
}

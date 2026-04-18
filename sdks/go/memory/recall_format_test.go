// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

var fixedNow = time.Date(2026, 4, 15, 12, 0, 0, 0, time.UTC)

func rfc3339Days(daysAgo int) string {
	return fixedNow.Add(-time.Duration(daysAgo) * 24 * time.Hour).Format(time.RFC3339)
}

func TestRelativeTimeString_Buckets(t *testing.T) {
	cases := []struct {
		name string
		then time.Time
		want string
	}{
		{"same day", fixedNow.Add(-2 * time.Hour), "today"},
		{"yesterday", fixedNow.Add(-24 * time.Hour), "yesterday"},
		{"three days", fixedNow.Add(-3 * 24 * time.Hour), "3 days ago"},
		{"six days", fixedNow.Add(-6 * 24 * time.Hour), "6 days ago"},
		{"one week", fixedNow.Add(-7 * 24 * time.Hour), "1 week ago"},
		{"three weeks", fixedNow.Add(-21 * 24 * time.Hour), "3 weeks ago"},
		{"one month", fixedNow.Add(-30 * 24 * time.Hour), "1 month ago"},
		{"six months", fixedNow.Add(-180 * 24 * time.Hour), "6 months ago"},
		{"eleven months", fixedNow.Add(-330 * 24 * time.Hour), "11 months ago"},
		{"one year", fixedNow.Add(-400 * 24 * time.Hour), "1 year ago"},
		{"two years", fixedNow.Add(-2 * 365 * 24 * time.Hour), "2 years ago"},
		{"future", fixedNow.Add(48 * time.Hour), ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := relativeTimeString(tc.then, fixedNow)
			if got != tc.want {
				t.Errorf("relativeTimeString(%s) = %q, want %q", tc.name, got, tc.want)
			}
		})
	}
}

func TestSortMemoriesChronologically_Empty(t *testing.T) {
	out := SortMemoriesChronologically(nil)
	if out != nil {
		t.Errorf("expected nil, got %#v", out)
	}
}

func TestSortMemoriesChronologically_OldestFirst(t *testing.T) {
	recent := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("recent"),
		Topic: TopicFile{Modified: rfc3339Days(1)},
	}
	ancient := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("ancient"),
		Topic: TopicFile{Modified: rfc3339Days(400)},
	}
	mid := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("mid"),
		Topic: TopicFile{Modified: rfc3339Days(30)},
	}

	in := []SurfacedMemory{recent, ancient, mid}
	out := SortMemoriesChronologically(in)

	if len(out) != 3 {
		t.Fatalf("expected 3 memories, got %d", len(out))
	}
	if string(out[0].Path) != string(ancient.Path) {
		t.Errorf("expected oldest first, got %s", out[0].Path)
	}
	if string(out[1].Path) != string(mid.Path) {
		t.Errorf("expected mid second, got %s", out[1].Path)
	}
	if string(out[2].Path) != string(recent.Path) {
		t.Errorf("expected recent last, got %s", out[2].Path)
	}

	if string(in[0].Path) != string(recent.Path) {
		t.Errorf("input was mutated: in[0] = %s", in[0].Path)
	}
}

func TestSortMemoriesChronologically_UndatedPreservedAtEnd(t *testing.T) {
	dated1 := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("dated1"),
		Topic: TopicFile{Modified: rfc3339Days(10)},
	}
	dated2 := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("dated2"),
		Topic: TopicFile{Modified: rfc3339Days(2)},
	}
	undatedA := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("undatedA"),
		Topic: TopicFile{},
	}
	undatedB := SurfacedMemory{
		Path:  brain.MemoryGlobalTopic("undatedB"),
		Topic: TopicFile{Modified: "not a date"},
	}

	in := []SurfacedMemory{dated2, undatedA, dated1, undatedB}
	out := SortMemoriesChronologically(in)

	if string(out[0].Path) != string(dated1.Path) {
		t.Errorf("out[0] = %s, want dated1", out[0].Path)
	}
	if string(out[1].Path) != string(dated2.Path) {
		t.Errorf("out[1] = %s, want dated2", out[1].Path)
	}
	if string(out[2].Path) != string(undatedA.Path) {
		t.Errorf("out[2] = %s, want undatedA (original order)", out[2].Path)
	}
	if string(out[3].Path) != string(undatedB.Path) {
		t.Errorf("out[3] = %s, want undatedB (original order)", out[3].Path)
	}
}

func TestSortMemoriesChronologically_StableOnEqualTimestamps(t *testing.T) {
	ts := rfc3339Days(5)
	a := SurfacedMemory{Path: brain.MemoryGlobalTopic("a"), Topic: TopicFile{Modified: ts}}
	b := SurfacedMemory{Path: brain.MemoryGlobalTopic("b"), Topic: TopicFile{Modified: ts}}
	c := SurfacedMemory{Path: brain.MemoryGlobalTopic("c"), Topic: TopicFile{Modified: ts}}

	out := SortMemoriesChronologically([]SurfacedMemory{a, b, c})
	if string(out[0].Path) != string(a.Path) ||
		string(out[1].Path) != string(b.Path) ||
		string(out[2].Path) != string(c.Path) {
		t.Errorf("expected stable a,b,c; got %s,%s,%s", out[0].Path, out[1].Path, out[2].Path)
	}
}

func TestFormatRecalledMemoriesWithContext_InjectsDateHeader(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryGlobalTopic("dated"),
			Content: "dated body",
			Topic: TopicFile{
				Name:     "Dated",
				Scope:    "global",
				Modified: rfc3339Days(21),
			},
		},
	}

	result := FormatRecalledMemoriesWithContext(memories, fixedNow)

	expectedDate := fixedNow.Add(-21 * 24 * time.Hour).UTC().Format("2006-01-02")
	if !strings.Contains(result, "=== "+expectedDate+" (3 weeks ago) ===") {
		t.Errorf("expected ISO date + relative header, got:\n%s", result)
	}
	if !strings.Contains(result, "dated body") {
		t.Error("expected memory content in output")
	}
	if !strings.Contains(result, "Global memory (saved") {
		t.Error("expected label preserved")
	}
}

func TestFormatRecalledMemoriesWithContext_RelativeBuckets(t *testing.T) {
	cases := []struct {
		name        string
		daysAgo     int
		wantRelFrag string
	}{
		{"today", 0, "today"},
		{"yesterday", 1, "yesterday"},
		{"three weeks", 21, "3 weeks ago"},
		{"two years", 2 * 365, "2 years ago"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			memories := []SurfacedMemory{
				{
					Path:    brain.MemoryGlobalTopic(tc.name),
					Content: "body",
					Topic:   TopicFile{Modified: rfc3339Days(tc.daysAgo)},
				},
			}

			out := FormatRecalledMemoriesWithContext(memories, fixedNow)
			if !strings.Contains(out, tc.wantRelFrag) {
				t.Errorf("expected relative fragment %q in output:\n%s", tc.wantRelFrag, out)
			}
			if !strings.Contains(out, "=== ") {
				t.Errorf("expected date header, got:\n%s", out)
			}
		})
	}
}

func TestFormatRecalledMemoriesWithContext_NoHeaderWithoutTimestamp(t *testing.T) {
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryGlobalTopic("nots"),
			Content: "content",
			Topic:   TopicFile{Name: "Nots", Scope: "project"},
		},
	}

	result := FormatRecalledMemoriesWithContext(memories, fixedNow)

	if strings.Contains(result, "=== ") {
		t.Errorf("did not expect date header for timestamp-less memory, got:\n%s", result)
	}
	if !strings.Contains(result, "content") {
		t.Error("expected content to still render")
	}
	if !strings.Contains(result, "Memory (saved unknown time ago)") {
		t.Errorf("expected unknown age label, got:\n%s", result)
	}
}

func TestFormatRecalledMemoriesWithContext_FutureTimestampOmitsRelative(t *testing.T) {
	future := fixedNow.Add(48 * time.Hour).Format(time.RFC3339)
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryGlobalTopic("future"),
			Content: "forward note",
			Topic:   TopicFile{Modified: future, Scope: "project"},
		},
	}

	result := FormatRecalledMemoriesWithContext(memories, fixedNow)

	if !strings.Contains(result, "=== ") {
		t.Errorf("expected ISO date header even for future memory, got:\n%s", result)
	}
	if strings.Contains(result, "(in ") || strings.Contains(result, "ago)") {
		t.Errorf("unexpected relative annotation for future memory:\n%s", result)
	}
}

func TestFormatRecalledMemoriesWithContext_Empty(t *testing.T) {
	if got := FormatRecalledMemoriesWithContext(nil, fixedNow); got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestFormatRecalledMemories_DelegatesToContextual(t *testing.T) {
	now := time.Now().UTC()
	memories := []SurfacedMemory{
		{
			Path:    brain.MemoryGlobalTopic("now"),
			Content: "fresh",
			Topic:   TopicFile{Modified: now.Format(time.RFC3339), Scope: "global"},
		},
	}

	result := FormatRecalledMemories(memories)

	isoToday := now.Format("2006-01-02")
	if !strings.Contains(result, "=== "+isoToday) {
		t.Errorf("expected ISO header with today's date (%s), got:\n%s", isoToday, result)
	}
	if !strings.Contains(result, "today") {
		t.Error("expected 'today' relative marker")
	}
}

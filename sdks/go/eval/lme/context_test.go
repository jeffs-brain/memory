// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"strings"
	"testing"
)

func TestProcessSessionContextForQuestion_PrioritisesRelevantSessionsAndKeepsTime(t *testing.T) {
	raw := strings.Join([]string{
		"---\nsession_id: older\nsession_date: 2023/05/25 (Thu) 03:59\n---\n[user]: I saw Dr. Smith every two weeks before the change.\n",
		"---\nsession_id: latest\nsession_date: 2023/05/25 (Thu) 13:14\n---\n[user]: I see Dr. Smith every week now.\n",
		"---\nsession_id: unrelated\nsession_date: 2023/05/26 (Fri) 09:00\n---\n[user]: I bought new trainers.\n",
	}, "\n\n---\n\n")

	got := processSessionContextForQuestion(raw, "How often do I see my therapist, Dr. Smith?")

	latestAt := strings.Index(got, "2023/05/25 (Thu) 13:14")
	olderAt := strings.Index(got, "2023/05/25 (Thu) 03:59")
	unrelatedAt := strings.Index(got, "2023/05/26 (Fri) 09:00")
	if latestAt == -1 || olderAt == -1 || unrelatedAt == -1 {
		t.Fatalf("expected every rendered session date to survive:\n%s", got)
	}
	if latestAt > olderAt {
		t.Fatalf("latest relevant session should sort ahead of the older one:\n%s", got)
	}
	if olderAt > unrelatedAt {
		t.Fatalf("unrelated session should sort after relevant ones:\n%s", got)
	}
}

func TestRenderRetrievedPassages_IncludesSourceRoleLabel(t *testing.T) {
	t.Parallel()

	got := RenderRetrievedPassages([]RetrievedPassage{
		{
			Path:       "memory/global/user-bike-count.md",
			Date:       "2024-03-01",
			SessionID:  "s1",
			SourceRole: "user",
			EventDate:  "2024-02-29",
			Body:       "The user owns two bikes.",
		},
	}, "How many bikes do I own?", "2024/03/13 (Wed) 10:00")

	if !strings.Contains(got, "[source_role=user]") {
		t.Fatalf("rendered evidence missing source role label:\n%s", got)
	}
	if !strings.Contains(got, "[event_date=2024-02-29]") {
		t.Fatalf("rendered evidence missing event date label:\n%s", got)
	}
}

func TestRenderRetrievedPassages_IncludesDaysAgoDelta(t *testing.T) {
	t.Parallel()

	got := RenderRetrievedPassages([]RetrievedPassage{
		{
			Path: "memory/global/networking-event.md",
			Date: "2022-03-09",
			Body: "The user attended a networking event.",
		},
	}, "How many days ago did I attend a networking event?", "2022/04/04 (Mon) 09:00")

	if !strings.Contains(got, "[days_before_question=26]") {
		t.Fatalf("rendered evidence missing date delta label:\n%s", got)
	}
}

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

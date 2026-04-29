// SPDX-License-Identifier: Apache-2.0

package retrieval

import "testing"

func TestShouldExpandSameSessionNeighboursUsesConversationCues(t *testing.T) {
	if !shouldExpandSameSessionNeighbours("Looking back at our previous conversation, what did we decide?") {
		t.Fatal("expected previous conversation cue to enable same-session expansion")
	}
}

func TestShouldExpandSameSessionNeighboursSkipsExactListRecall(t *testing.T) {
	query := "Can you remind me of the specific programming languages you recommended in our previous conversation?"
	if shouldExpandSameSessionNeighbours(query) {
		t.Fatal("exact list recall should avoid neighbour expansion")
	}
}

func TestShouldExpandSameSessionNeighboursUsesContextualTypeQueries(t *testing.T) {
	query := "How many different types of citrus fruits have I used in my cocktail recipes?"
	if !shouldExpandSameSessionNeighbours(query) {
		t.Fatal("type-context count query should enable same-session expansion")
	}
}

func TestShouldExpandSameSessionNeighboursUsesWhenDidActionDateQueries(t *testing.T) {
	query := "When did I submit my research paper on sentiment analysis?"
	if !shouldExpandSameSessionNeighbours(query) {
		t.Fatal("when-did action date query should enable same-session expansion")
	}
}

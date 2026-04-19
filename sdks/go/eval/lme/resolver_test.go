// SPDX-License-Identifier: Apache-2.0

package lme

import "testing"

func TestResolveDeterministicAnswer_AnchoredActionDate(t *testing.T) {
	content := `Retrieved facts (2):

 1. [2023-05-22] [paper]
[Date: 2023-05-22 Monday May 2023]

I worked on a research paper on sentiment analysis, which I submitted to ACL.

 2. [2023-05-30] [submission]
[Date: 2023-02-01 Wednesday February 2023]

I'm reviewing for ACL, and their submission date was February 1st.`

	got, ok := ResolveDeterministicAnswer(
		"When did I submit my research paper on sentiment analysis?",
		content,
	)
	if !ok {
		t.Fatal("expected deterministic action-date answer")
	}
	if got != "February 1st" {
		t.Fatalf("got %q, want February 1st", got)
	}
}

func TestResolveDeterministicAnswer_NamedSpendTotal(t *testing.T) {
	content := `Retrieved facts (2):

 1. [2023-05-28] [brother]
[Date: 2023-05-28 Sunday May 2023]

The user spent a total of $500 on gifts recently. They bought their brother a graduation gift in May: a $100 gift card to his favourite electronics store.

 2. [2023-05-28] [coworker]
[Date: 2023-05-28 Sunday May 2023]

The user thinks they bought a set of baby clothes and toys for their coworker's baby shower, and it cost around $100.`

	got, ok := ResolveDeterministicAnswer(
		"What is the total amount I spent on gifts for my coworker and brother?",
		content,
	)
	if !ok {
		t.Fatal("expected deterministic total-spend answer")
	}
	if got != "$200" {
		t.Fatalf("got %q, want $200", got)
	}
}

func TestResolveDeterministicAnswer_PartialNamedSpendAbstains(t *testing.T) {
	content := `Retrieved facts (1):

 1. [2023-05-28] [coworker]
[Date: 2023-05-28 Sunday May 2023]

The user bought a baby shower gift for their coworker, and it cost $100.`

	if got, ok := ResolveDeterministicAnswer(
		"What is the total amount I spent on gifts for my coworker and sister?",
		content,
	); ok {
		t.Fatalf("unexpected answer %q for partial named total", got)
	}
}

func TestResolveDeterministicAnswer_BackendRecommendation(t *testing.T) {
	content := `Retrieved facts (2):

 1. [2023-05-26] [full-stack-tips]
[Date: 2023-05-26 Friday May 2023]

Learn a back-end programming language, such as Ruby, Python, or PHP.

 2. [2023-05-26] [recommended-resources]
[Date: 2023-05-26 Friday May 2023]

Recommended resources and courses include NodeSchool, Python, SQL, Flask, and Django.`

	got, ok := ResolveDeterministicAnswer(
		"Can you remind me of the specific back-end programming languages you recommended I learn?",
		content,
	)
	if !ok {
		t.Fatal("expected deterministic recommendation answer")
	}
	if got != "I recommended learning Ruby, Python, or PHP as a back-end programming language." {
		t.Fatalf("got %q", got)
	}
}

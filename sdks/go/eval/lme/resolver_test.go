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

func TestResolveDeterministicAnswer_CashbackAmount(t *testing.T) {
	content := `Retrieved facts (4):

 1. [2023-05-25] [loyalty-tracker]
[Date: 2023-05-25 Thursday May 2023]

Maintain an up-to-date record of the user's loyalty programs, balances, redemptions, and savings so future assistance can reference exact amounts and history.

 2. [2023-05-18] [savemart]
[Date: 2023-05-18 Thursday May 2023]

I spent $75 on groceries at SaveMart last Thursday.

 3. [2023-05-22] [membership]
[Date: 2023-05-22 Monday May 2023]

I have a membership there and can earn 1% cashback on all purchases.

 4. [2023-05-26] [walmart-plus]
[Date: 2023-05-26 Friday May 2023]

Do you think it's worth it to pay the monthly fee for Walmart+ just for the 2% cashback on online grocery purchases?`

	got, ok := ResolveDeterministicAnswer(
		"How much cashback did I earn at SaveMart last Thursday?",
		content,
	)
	if !ok {
		t.Fatal("expected deterministic cashback answer")
	}
	if got != "$0.75" {
		t.Fatalf("got %q, want $0.75", got)
	}
}

func TestResolveDeterministicAnswer_FirstComparison(t *testing.T) {
	content := `Retrieved facts (3):

 1. [2023-05-25] [smart-thermostat]
[Date: 2023-04-25 Tuesday April 2023]

Also, since I set up my smart thermostat a month ago, I've noticed that it's been learning my schedule and preferences.

 2. [2023-05-25] [mesh-network]
[Date: 2023-05-25 Thursday May 2023]

Since I recently upgraded my home Wi-Fi router to a new mesh network system, which has significantly improved my internet connection, I'm thinking maybe it's time to upgrade my computer too.

 3. [2023-05-25] [mesh-network-follow-up]
[Date: 2023-05-25 Thursday May 2023]

I'm not really sure about the specific components yet, but I do know that I want to make sure it can handle my internet connection well, since I just upgraded to a mesh network system.`

	got, ok := ResolveDeterministicAnswer(
		"Which device did I set up first, the smart thermostat or the mesh network system?",
		content,
	)
	if !ok {
		t.Fatal("expected deterministic comparison answer")
	}
	if got != "smart thermostat" {
		t.Fatalf("got %q, want smart thermostat", got)
	}
}

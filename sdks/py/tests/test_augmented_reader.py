# SPDX-License-Identifier: Apache-2.0
"""Deterministic augmented-reader resolver tests."""

from jeffs_brain_memory.augmented_reader import resolve_deterministic_augmented_answer


def test_resolver_combines_action_fact_with_anchored_submission_date() -> None:
    rendered = """
Retrieved facts (2):

 1. [2023-05-22] [session=s1] [paper]
I submitted my research paper on sentiment analysis to ACL.

 2. [2023-02-01] [session=s2] [acl]
I'm reviewing for ACL, and their submission date was February 1st.
""".strip()

    answer = resolve_deterministic_augmented_answer(
        "When did I submit my research paper on sentiment analysis?",
        rendered,
    )

    assert answer is not None
    assert "February 1st" in answer


def test_resolver_requires_specific_booking_target_match() -> None:
    rendered = """
Retrieved facts (2):

 1. [2023-06-10] [session=s1] [airbnb]
I booked an Airbnb in San Francisco for the wedding trip.

 2. [2023-06-01] [session=s2] [airbnb-date]
The San Francisco Airbnb booking date was June 1st.
""".strip()

    assert (
        resolve_deterministic_augmented_answer(
            "When did I book the Airbnb in Sacramento?",
            rendered,
        )
        is None
    )


def test_resolver_sums_named_recipient_spend_without_double_counting_rollups() -> None:
    rendered = """
Retrieved facts (4):

 1. [2023-05-28] [session=s1] [brother]
I bought a $100 gift card for my brother's graduation in May.

 2. [2023-05-28] [session=s1] [coworker]
I purchased a set of baby clothes and toys from Buy Buy Baby for my coworker's baby shower, costing around $100.

 3. [2023-05-29] [session=s2] [coworker-recap]
It was recalled that the user got a set of baby clothes and toys from Buy Buy Baby for a coworker's baby shower costing around $100.

 4. [2023-05-30] [session=s3] [rollup]
Summary: in total I spent $200 on gifts for my coworker and brother.
""".strip()

    answer = resolve_deterministic_augmented_answer(
        "What is the total amount I spent on gifts for my coworker and brother?",
        rendered,
    )

    assert answer == "$200"


def test_resolver_prefers_direct_back_end_language_clause_over_resource_lists() -> None:
    rendered = """
Retrieved facts (2):

 1. [2023-05-26] [session=s1] [resources]
Recommended back-end resources include NodeSchool, Udacity, Coursera, Flask, Django, Spring, Hibernate, SQL.

 2. [2023-05-26] [session=s1] [languages]
Learn a back-end programming language, such as Ruby, Python, or PHP.
""".strip()

    answer = resolve_deterministic_augmented_answer(
        "Can you remind me of the specific back-end programming languages you recommended I learn?",
        rendered,
    )

    assert answer == "I recommended learning Ruby, Python, or PHP as a back-end programming language."

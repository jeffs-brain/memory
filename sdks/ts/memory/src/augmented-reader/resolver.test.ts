// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import {
  parseRenderedRetrievedFacts,
  resolveDeterministicAugmentedAnswer,
} from './resolver.js'

const renderFacts = (bodies: readonly string[]): string =>
  [
    `Retrieved facts (${String(bodies.length)}):`,
    '',
    ...bodies.flatMap((body, index) => [
      ` ${String(index + 1)}. [2024-01-${String(index + 1).padStart(2, '0')}] [fact-${String(index + 1)}]`,
      body,
      '',
    ]),
  ].join('\n')

const renderDatedFacts = (
  facts: readonly {
    readonly date: string
    readonly source: string
    readonly body: string
  }[],
): string =>
  [
    `Retrieved facts (${String(facts.length)}):`,
    '',
    ...facts.flatMap((fact, index) => [
      ` ${String(index + 1)}. [${fact.date}] [${fact.source}]`,
      fact.body,
      '',
    ]),
  ].join('\n')

describe('parseRenderedRetrievedFacts', () => {
  it('parses numbered retrieved-facts blocks', () => {
    const facts = parseRenderedRetrievedFacts(
      [
        '[Resolved temporal references: "last Friday" -> 2024/03/08 (Friday)]',
        '',
        'Retrieved facts (2):',
        '',
        ' 1. [2024-01-10] [session=s1] [note-a]',
        'First note body.',
        '',
        ' 2. [2024-01-11] [note-b]',
        'Second note body.',
      ].join('\n'),
    )

    expect(facts).toEqual([
      {
        index: 1,
        date: '2024-01-10',
        sessionId: 's1',
        source: 'note-a',
        body: 'First note body.',
      },
      {
        index: 2,
        date: '2024-01-11',
        source: 'note-b',
        body: 'Second note body.',
      },
    ])
  })
})

describe('resolveDeterministicAugmentedAnswer', () => {
  it('combines anchored action facts with matching submission-date facts', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question: 'When did I submit my research paper on sentiment analysis?',
      rendered: renderFacts([
        'I submitted my research paper on sentiment analysis to ACL.',
        "I'm reviewing for ACL, and their submission date was February 1st.",
      ]),
    })

    expect(resolved).toEqual({
      kind: 'action-date',
      answer: 'February 1st',
    })
  })

  it('abstains when the booking target does not match specifically', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question: 'When did I book the Airbnb in Sacramento?',
      rendered: renderDatedFacts([
        {
          date: '2023-06-10',
          source: 'airbnb',
          body: 'I booked an Airbnb in San Francisco for the wedding trip.',
        },
        {
          date: '2023-06-01',
          source: 'airbnb-date',
          body: 'The San Francisco Airbnb booking date was June 1st.',
        },
      ]),
    })

    expect(resolved).toBeUndefined()
  })

  it('resolves relative action dates from the fact anchor date', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question: 'What date did I join the running club?',
      rendered: renderDatedFacts([
        {
          date: '2024-03-08',
          source: 'running-club',
          body: 'I joined the running club yesterday.',
        },
      ]),
    })

    expect(resolved).toEqual({
      kind: 'action-date',
      answer: 'March 7th',
    })
  })

  it('sums direct recipient spend without counting roll-up summaries', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question: 'What is the total amount I spent on gifts for my coworker and brother?',
      rendered: renderFacts([
        'I bought my coworker a coffee gift set for $24.',
        'I spent $36 on a scarf for my brother.',
        'Gift summary: in total I spent $60 on gifts for my coworker and brother.',
      ]),
    })

    expect(resolved).toEqual({
      kind: 'recipient-total-spend',
      answer: '$60',
    })
  })

  it('sums third-person recipient spend notes without counting summary roll-ups', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question: 'What is the total amount I spent on gifts for my coworker and brother?',
      rendered: renderDatedFacts([
        {
          date: '2023-05-28',
          source: 'brother',
          body: 'The user spent a total of $500 on gifts recently. They bought their brother a graduation gift in May: a $100 gift card to his favourite electronics store.',
        },
        {
          date: '2023-05-28',
          source: 'coworker',
          body: "The user thinks they bought a set of baby clothes and toys for their coworker's baby shower, and it cost around $100.",
        },
      ]),
    })

    expect(resolved).toEqual({
      kind: 'recipient-total-spend',
      answer: '$200',
    })
  })

  it('extracts direct back-end language recommendations and ignores resource lists', () => {
    const resolved = resolveDeterministicAugmentedAnswer({
      question:
        'Can you remind me of the specific back-end programming languages you recommended I learn?',
      rendered: renderFacts([
        'Recommended back-end resources include NodeSchool, Udacity, Coursera, Flask, Django, Spring, Hibernate and SQL.',
        'Learn a back-end programming language, such as Ruby, Python or PHP.',
      ]),
    })

    expect(resolved).toEqual({
      kind: 'backend-language-recommendation',
      answer: 'I recommended learning Ruby, Python, or PHP as a back-end programming language.',
    })
  })
})

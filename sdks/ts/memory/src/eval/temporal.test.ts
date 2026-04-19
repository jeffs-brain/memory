// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  augmentQueryWithTemporal,
  dateSearchTokens,
  readerTodayAnchor,
  resolvedTemporalHintLine,
} from './temporal.js'

describe('temporal helpers', () => {
  it('normalises the reader date anchor', () => {
    expect(readerTodayAnchor('2023/09/30 (Sat) 18:36')).toBe(
      '2023-09-30 (Saturday)',
    )
  })

  it('augments relative temporal queries with concrete dates', () => {
    expect(
      augmentQueryWithTemporal(
        'What did I do 3 weeks ago?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toContain('2023/09/09')
  })

  it('augments number-word relative temporal queries with concrete dates', () => {
    expect(
      augmentQueryWithTemporal(
        'What did I do two weeks ago?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toContain('2023/09/16')
  })

  it('augments yesterday with a concrete date', () => {
    expect(
      augmentQueryWithTemporal(
        'What did I do yesterday?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toContain('2023/09/29')
  })

  it('augments last week with every day in the prior seven-day window', () => {
    const augmented = augmentQueryWithTemporal(
      'Where did I volunteer last week?',
      '2023/09/30 (Sat) 18:36',
    )
    expect(augmented).toContain('2023/09/23')
    expect(augmented).toContain('2023/09/29')
  })

  it('renders resolved temporal hints for the reader context', () => {
    expect(
      resolvedTemporalHintLine(
        'What did I do 3 weeks ago?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toBe('[Resolved temporal references: 2023/09/09]')
  })

  it('renders the full last-week hint range for the reader context', () => {
    expect(
      resolvedTemporalHintLine(
        'Where did I volunteer last week?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toBe(
      '[Resolved temporal references: 2023/09/23, 2023/09/24, 2023/09/25, 2023/09/26, 2023/09/27, 2023/09/28, 2023/09/29]',
    )
  })

  it('derives retrieval tags from stored dates', () => {
    expect(dateSearchTokens('2023/09/30 (Sat) 18:36')).toEqual([
      '2023-09-30',
      '2023/09/30',
      '2023',
      'Saturday',
      'September',
    ])
  })
})

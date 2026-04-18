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

  it('renders resolved temporal hints for the reader context', () => {
    expect(
      resolvedTemporalHintLine(
        'What did I do 3 weeks ago?',
        '2023/09/30 (Sat) 18:36',
      ),
    ).toBe('[Resolved temporal references: 2023/09/09]')
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

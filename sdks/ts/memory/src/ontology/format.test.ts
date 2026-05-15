// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { formatNodeTypeLabel, formatEdgeTypeLabel } from './format.js'

describe('formatNodeTypeLabel', () => {
  it('formats a dotted entity type into "Name (Prefix)" form', () => {
    expect(formatNodeTypeLabel('entity.customer')).toBe('Customer (Entity)')
  })

  it('converts underscored names to title case with spaces', () => {
    expect(formatNodeTypeLabel('process.approval_chain')).toBe('Approval Chain (Process)')
  })

  it('returns the raw string when there is no dot separator', () => {
    expect(formatNodeTypeLabel('standalone')).toBe('standalone')
  })

  it('handles single-character prefix and name', () => {
    expect(formatNodeTypeLabel('a.b')).toBe('B (A)')
  })

  it('handles multiple underscores in the name segment', () => {
    expect(formatNodeTypeLabel('rule.multi_word_name_here')).toBe('Multi Word Name Here (Rule)')
  })

  it('handles a type with only a dot and no name', () => {
    expect(formatNodeTypeLabel('prefix.')).toBe(' (Prefix)')
  })

  it('handles a type starting with a dot', () => {
    expect(formatNodeTypeLabel('.suffix')).toBe('Suffix ()')
  })
})

describe('formatEdgeTypeLabel', () => {
  it('formats a single-word edge type to title case', () => {
    expect(formatEdgeTypeLabel('triggers')).toBe('Triggers')
  })

  it('converts underscored edge types to title-cased words', () => {
    expect(formatEdgeTypeLabel('requires_approval_from')).toBe('Requires Approval From')
  })

  it('handles a single character', () => {
    expect(formatEdgeTypeLabel('x')).toBe('X')
  })

  it('handles multiple consecutive underscores gracefully', () => {
    const result = formatEdgeTypeLabel('a__b')
    expect(result).toBe('A  B')
  })
})

// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import type { CompletionRequest, CompletionResponse, Provider, StreamEvent } from '../llm/types.js'
import type { ResolvedOntology, ResolvedType } from './store.js'

import {
  CONFIDENCE_CAP,
  CONFIDENCE_FLOOR,
  Extractor,
  isTabularContent,
  noisyOr,
  buildOntologyExtractionPrompt,
  splitContent,
  SINGLE_SECTION_THRESHOLD,
} from './extract.js'

function makeFakeProvider(responses: string[]): Provider {
  let idx = 0
  return {
    name: () => 'fake',
    modelName: () => 'fake-model',
    supportsStructuredDecoding: () => false,
    structured: async () => '',
    stream: (_req: CompletionRequest, _signal?: AbortSignal): AsyncIterable<StreamEvent> => {
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true as const, value: undefined }),
        }),
      }
    },
    complete: async (): Promise<CompletionResponse> => {
      const text = responses[idx % responses.length] ?? ''
      idx++
      return {
        content: text,
        toolCalls: [],
        usage: { inputTokens: 10, outputTokens: 10 },
        stopReason: 'end_turn',
      }
    },
  }
}

function makeLLMResponse(
  domain: string,
  confidence: number,
  nodeTypes: Array<{ type: string; label: string; description: string }>,
  edgeTypes: Array<{ type: string; label: string; description: string }>,
  categories: string[],
): string {
  return JSON.stringify({ domain, confidence, nodeTypes, edgeTypes, businessCategories: categories })
}

function generateLongContent(minBytes: number): string {
  const section =
    '# Section\n\nThis is a paragraph about server hardware and configuration management. ' +
    'It discusses various components, compatibility rules, and operational procedures.\n\n'
  let content = ''
  while (content.length < minBytes) {
    content += section
  }
  return content
}

describe('Extractor', () => {
  it('returns empty result when provider is undefined', async () => {
    const ext = new Extractor({ provider: undefined })
    const result = await ext.extract({
      content: 'Some document about servers.',
      fileName: 'servers.md',
    })
    expect(result.nodeTypes).toHaveLength(0)
    expect(result.edgeTypes).toHaveLength(0)
    expect(result.businessCategories).toHaveLength(0)
  })

  it('returns empty result for empty content', async () => {
    const ext = new Extractor({
      provider: makeFakeProvider(['{}']),
    })
    const result = await ext.extract({ content: '', fileName: '' })
    expect(result.nodeTypes).toHaveLength(0)
  })

  it('extracts from single section when content is under threshold', async () => {
    const response = makeLLMResponse(
      'healthcare',
      0.85,
      [
        { type: 'entity.patient', label: 'Patient', description: 'A person receiving care' },
        {
          type: 'rule.clinical_protocol',
          label: 'Clinical Protocol',
          description: 'A clinical treatment rule',
        },
      ],
      [{ type: 'treats', label: 'Treats', description: 'Treatment relationship' }],
      ['patient_care'],
    )

    const ext = new Extractor({ provider: makeFakeProvider([response]) })
    const result = await ext.extract({
      content: 'Short healthcare document about patients.',
      fileName: 'healthcare.md',
    })

    expect(result.domain).toBe('healthcare')
    expect(result.confidence).toBe(0.85)
    expect(result.nodeTypes).toHaveLength(2)
    expect(result.edgeTypes).toHaveLength(1)
    expect(result.businessCategories).toHaveLength(1)
  })

  it('handles multi-section extraction for long content', async () => {
    const longContent = generateLongContent(SINGLE_SECTION_THRESHOLD + 1000)
    const response1 = makeLLMResponse(
      'finance',
      0.7,
      [{ type: 'entity.account', label: 'Account', description: 'A financial account' }],
      [{ type: 'holds', label: 'Holds', description: 'An account holds assets' }],
      ['retail_banking'],
    )
    const response2 = makeLLMResponse(
      'finance',
      0.8,
      [{ type: 'entity.payment', label: 'Payment', description: 'A transfer of funds' }],
      [{ type: 'settles', label: 'Settles', description: 'A transaction settles' }],
      ['payments'],
    )

    const ext = new Extractor({ provider: makeFakeProvider([response1, response2]) })
    const result = await ext.extract({
      content: longContent,
      fileName: 'finance.md',
    })

    expect(result.domain).toBe('finance')
    expect(result.nodeTypes.length).toBeGreaterThanOrEqual(1)
  })

  it('filters existing types from result', async () => {
    const response = makeLLMResponse(
      'healthcare',
      0.85,
      [
        { type: 'entity.patient', label: 'Patient', description: 'A person receiving care' },
        { type: 'entity.medication', label: 'Medication', description: 'A pharmaceutical product' },
      ],
      [{ type: 'treats', label: 'Treats', description: 'Treatment relationship' }],
      ['patient_care'],
    )

    const existing: ResolvedOntology = {
      nodeTypes: [
        {
          type: 'entity.patient',
          label: 'Patient',
          description: 'A person',
          createdAt: '2026-01-01',
          status: 'active',
          scope: 'built-in',
        },
      ],
      edgeTypes: [],
      businessCategories: [],
    }

    const ext = new Extractor({ provider: makeFakeProvider([response]) })
    const result = await ext.extract({
      content: 'Healthcare document about patients and medications.',
      fileName: 'healthcare.md',
      existingTypes: existing,
    })

    const patientType = result.nodeTypes.find((nt) => nt.type === 'entity.patient')
    expect(patientType).toBeUndefined()
  })

  it('retries on malformed JSON and succeeds', async () => {
    const goodResponse = makeLLMResponse(
      'finance',
      0.9,
      [{ type: 'entity.account', label: 'Account', description: 'A financial account' }],
      [],
      ['banking'],
    )

    const ext = new Extractor({
      provider: makeFakeProvider(['Here is analysis: {invalid...', goodResponse]),
    })
    const result = await ext.extract({
      content: 'Short finance document.',
      fileName: 'finance.md',
    })

    expect(result.domain).toBe('finance')
  })

  it('throws after exhausting retries', async () => {
    const ext = new Extractor({
      provider: makeFakeProvider(['not json', 'still not', 'nope']),
    })
    await expect(
      ext.extract({ content: 'Some document.', fileName: 'doc.md' }),
    ).rejects.toThrow(/extraction failed/)
  })

  it('handles JSON wrapped in prose', async () => {
    const goodJSON = makeLLMResponse(
      'tech',
      0.8,
      [{ type: 'entity.server', label: 'Server', description: 'A server instance' }],
      [],
      ['infrastructure'],
    )
    const wrapped = `Here is my analysis:\n\n${goodJSON}\n\nLet me know if you need more.`

    const ext = new Extractor({ provider: makeFakeProvider([wrapped]) })
    const result = await ext.extract({
      content: 'Server documentation.',
      fileName: 'servers.md',
    })

    expect(result.domain).toBe('tech')
  })
})

describe('noisyOr', () => {
  it('returns 0 for empty input', () => {
    expect(noisyOr([])).toBe(0)
  })

  it('returns single value as-is when above floor', () => {
    expect(noisyOr([0.7])).toBe(0.7)
  })

  it('caps single value at 0.99', () => {
    expect(noisyOr([1.0])).toBe(CONFIDENCE_CAP)
  })

  it('computes noisy-OR for multiple values', () => {
    // 1 - (0.3 * 0.2) = 0.94
    const result = noisyOr([0.7, 0.8])
    expect(result).toBeCloseTo(0.94, 2)
  })

  it('returns 0 for single value below floor', () => {
    expect(noisyOr([0.2])).toBe(0)
  })

  it('filters values below floor', () => {
    expect(noisyOr([0.1, 0.2, 0.15])).toBe(0)
  })

  it('caps at 0.99 for very high confidences', () => {
    const result = noisyOr([0.95, 0.95, 0.95])
    expect(result).toBe(CONFIDENCE_CAP)
  })

  it('filters below-floor values from mixed input', () => {
    // 0.1 is below floor; 0.7 and 0.8 are above
    const result = noisyOr([0.1, 0.7, 0.8])
    expect(result).toBeCloseTo(0.94, 2)
  })
})

describe('splitContent', () => {
  it('does not split content under threshold', () => {
    const content = 'Short document.'
    const sections = splitContent(content, SINGLE_SECTION_THRESHOLD)
    expect(sections).toHaveLength(1)
    expect(sections[0]).toBe(content)
  })

  it('splits tabular content with header in each section', () => {
    let content = 'col1,col2,col3\nval1,val2,val3\nval4,val5,val6\nval7,val8,val9'
    while (content.length <= SINGLE_SECTION_THRESHOLD) {
      content += '\nextra1,extra2,extra3'
    }
    const sections = splitContent(content, SINGLE_SECTION_THRESHOLD)
    expect(sections.length).toBeGreaterThanOrEqual(2)
    for (const s of sections) {
      expect(s.startsWith('col1,col2,col3')).toBe(true)
    }
  })

  it('splits headed content at markdown headings', () => {
    let content = '# Section 1\n\nContent for section 1.\n\n# Section 2\n\nContent for section 2.'
    while (content.length <= SINGLE_SECTION_THRESHOLD) {
      content += '\n\nMore content to pad the document.'
    }
    const sections = splitContent(content, SINGLE_SECTION_THRESHOLD)
    expect(sections.length).toBeGreaterThanOrEqual(2)
  })
})

describe('isTabularContent', () => {
  it('detects CSV content', () => {
    const content =
      'col1,col2,col3\nval1,val2,val3\nval4,val5,val6\nval7,val8,val9\nval10,val11,val12'
    expect(isTabularContent(content)).toBe(true)
  })

  it('detects pipe-delimited content', () => {
    const content =
      'col1|col2|col3\nval1|val2|val3\nval4|val5|val6\nval7|val8|val9\nval10|val11|val12'
    expect(isTabularContent(content)).toBe(true)
  })

  it('rejects prose content', () => {
    const content =
      'This is a paragraph.\nIt has sentences.\nBut no delimiters.\nJust text.\nNothing tabular.'
    expect(isTabularContent(content)).toBe(false)
  })
})

describe('buildOntologyExtractionPrompt', () => {
  it('includes existing types when provided', () => {
    const existing: ResolvedOntology = {
      nodeTypes: [
        {
          type: 'entity.customer',
          label: 'Customer',
          description: 'A customer',
          createdAt: '2026-01-01',
          status: 'active',
          scope: 'built-in',
        },
      ],
      edgeTypes: [
        {
          type: 'triggers',
          label: 'Triggers',
          description: 'Triggers',
          createdAt: '2026-01-01',
          status: 'active',
          scope: 'built-in',
        },
      ],
      businessCategories: [],
    }
    const prompt = buildOntologyExtractionPrompt(existing)
    expect(prompt).toContain('entity.customer')
    expect(prompt).toContain('triggers')
    expect(prompt).toContain('Do NOT include these')
  })

  it('does not include existing types section when none provided', () => {
    const prompt = buildOntologyExtractionPrompt(undefined)
    expect(prompt).not.toContain('Do NOT include these')
  })
})

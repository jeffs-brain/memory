// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { extractCSV, extractJSON, extractJSONL } from './structured.js'

// -------------------------------------------------------------------
// CSV Extractor
// -------------------------------------------------------------------

describe('extractCSV', () => {
  it('extracts basic CSV with headers', () => {
    const input = Buffer.from('Name,Age,City\nAlice,30,London\nBob,25,Paris\n')
    const result = extractCSV(input)
    expect(result.content).toContain('- Name: Alice')
    expect(result.content).toContain('- Age: 30')
    expect(result.content).toContain('- City: London')
    expect(result.metadata.column_count).toBe('3')
    expect(result.metadata.row_count).toBe('2')
  })

  it('auto-detects semicolon delimiter', () => {
    const input = Buffer.from('Name;Age;City\nAlice;30;London\nBob;25;Paris\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe(';')
    expect(result.content).toContain('- Name: Alice')
  })

  it('auto-detects tab delimiter', () => {
    const input = Buffer.from('Name\tAge\tCity\nAlice\t30\tLondon\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe('\t')
    expect(result.mime).toBe('text/tab-separated-values')
  })

  it('auto-detects pipe delimiter', () => {
    const input = Buffer.from('Name|Age|City\nAlice|30|London\nBob|25|Paris\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe('|')
  })

  it('uses synthetic headers when first row is numeric', () => {
    const input = Buffer.from('1,2,3\n4,5,6\n7,8,9\n')
    const result = extractCSV(input)
    expect(result.content).toContain('Column_1')
    expect(result.content).toContain('Column_2')
    expect(result.content).toContain('Column_3')
    expect(result.metadata.row_count).toBe('3')
  })

  it('throws for empty input', () => {
    expect(() => extractCSV(Buffer.alloc(0))).toThrow('empty csv')
  })

  it('strips UTF-8 BOM', () => {
    const bom = Buffer.from([0xef, 0xbb, 0xbf])
    const content = Buffer.from('Name,Age\nAlice,30\n')
    const input = Buffer.concat([bom, content])
    const result = extractCSV(input)
    expect(result.metadata.encoding).toBe('utf-8-bom')
    expect(result.content).toContain('- Name: Alice')
  })

  it('decodes Latin-1 encoding', () => {
    // Latin-1 bytes: ï = 0xEF, é = 0xE9, ü = 0xFC
    const input = Buffer.from([
      0x4e, 0x61, 0xef, 0x76, 0x65, 0x2c, // "Naïve,"
      0x43, 0x61, 0x66, 0xe9, 0x0a, // "Café\n"
      0x41, 0x6c, 0x69, 0x63, 0x65, 0x2c, // "Alice,"
      0x5a, 0xfc, 0x72, 0x69, 0x63, 0x68, 0x0a, // "Zürich\n"
    ])
    const result = extractCSV(input)
    expect(result.metadata.encoding).toBe('latin-1')
  })

  it('chunks rows at configured boundaries', () => {
    const lines = ['Name,Value']
    for (let i = 0; i < 120; i++) {
      lines.push(`item,${i}`)
    }
    const input = Buffer.from(lines.join('\n'))
    const result = extractCSV(input, { rowsPerChunk: 50 })
    const separators = (result.content.match(/---/g) ?? []).length
    expect(separators).toBe(2) // 3 chunks = 2 separators
    expect(result.metadata.row_count).toBe('120')
  })

  it('respects forced delimiter', () => {
    const input = Buffer.from('Name;Age\nAlice;30\n')
    const result = extractCSV(input, { forceDelimiter: ',' })
    // With comma forced, the entire "Name;Age" is one field.
    expect(result.metadata.delimiter).toBe(',')
  })

  it('uses synthetic headers for duplicate first row values', () => {
    const input = Buffer.from('Name,Name,Age\nAlice,Bob,30\n')
    const result = extractCSV(input)
    expect(result.content).toContain('Column_1')
  })

  it('preserves row numbering across chunks', () => {
    const lines = ['Name,Value']
    for (let i = 0; i < 60; i++) {
      lines.push(`item_${i},${i}`)
    }
    const input = Buffer.from(lines.join('\n'))
    const result = extractCSV(input, { rowsPerChunk: 50 })
    expect(result.content).toContain('Row 1:')
    expect(result.content).toContain('Row 51:')
  })
})

// -------------------------------------------------------------------
// JSON Extractor
// -------------------------------------------------------------------

describe('extractJSON', () => {
  it('extracts simple object with structural context', () => {
    const input = Buffer.from('{"name":"Alice","age":30,"city":"London"}')
    const result = extractJSON(input)
    expect(result.metadata.structure_type).toBe('object')
    expect(result.content).toContain('Object with 3 keys')
    expect(result.content).toContain('age:')
  })

  it('renders uniform array of objects as markdown table', () => {
    const input = Buffer.from(
      '[{"name":"Alice","age":30},{"name":"Bob","age":25},{"name":"Carol","age":35}]',
    )
    const result = extractJSON(input, { tableThreshold: 3 })
    expect(result.metadata.structure_type).toBe('uniform_object_array')
    expect(result.content).toContain('| age |')
    expect(result.content).toContain('| name |')
    expect(result.content).toContain('| --- |')
  })

  it('renders heterogeneous array of objects individually', () => {
    const input = Buffer.from('[{"a":1,"b":2},{"c":3,"d":4}]')
    const result = extractJSON(input)
    expect(result.metadata.structure_type).toBe('heterogeneous_object_array')
    expect(result.content).toContain('Item 1:')
  })

  it('flattens deeply nested objects to dot-notation', () => {
    const input = Buffer.from('{"a":{"b":{"c":{"d":{"e":"deep"}}}}}')
    const result = extractJSON(input)
    expect(result.content).toContain('a.b.c.d.e: deep')
  })

  it('handles empty object', () => {
    const input = Buffer.from('{}')
    const result = extractJSON(input)
    expect(result.content).toBe('{}')
    expect(result.metadata.structure_type).toBe('empty_object')
  })

  it('handles empty array', () => {
    const input = Buffer.from('[]')
    const result = extractJSON(input)
    expect(result.content).toBe('[]')
    expect(result.metadata.structure_type).toBe('empty_array')
  })

  it('throws for invalid JSON', () => {
    const input = Buffer.from('{invalid')
    expect(() => extractJSON(input)).toThrow('invalid json')
  })

  it('joins primitive array with newlines', () => {
    const input = Buffer.from('[1,2,3,4,5]')
    const result = extractJSON(input)
    expect(result.metadata.structure_type).toBe('primitive_array')
    const lines = result.content.split('\n')
    expect(lines).toHaveLength(5)
  })

  it('throws for empty input', () => {
    expect(() => extractJSON(Buffer.alloc(0))).toThrow('empty json')
  })

  it('chunks objects at configured boundaries', () => {
    const objects = Array.from({ length: 120 }, () => '{"id":1,"name":"test"}')
    const input = Buffer.from(`[${objects.join(',')}]`)
    const result = extractJSON(input, { objectsPerChunk: 50, tableThreshold: 3 })
    const separators = (result.content.match(/\n---\n/g) ?? []).length
    expect(separators).toBeGreaterThanOrEqual(2)
  })

  it('detects schema from sampled objects', () => {
    const input = Buffer.from(
      '[{"name":"Alice","age":30,"city":"London"},{"name":"Bob","age":25,"city":"Paris"},{"name":"Carol","age":35,"city":"Berlin"}]',
    )
    const result = extractJSON(input, { tableThreshold: 3 })
    expect(result.metadata.detected_schema).toBeDefined()
    expect(result.metadata.detected_schema).toContain('name')
  })

  it('handles null values', () => {
    const input = Buffer.from('{"key":null}')
    const result = extractJSON(input)
    expect(result.content).toContain('null')
  })

  it('handles boolean values', () => {
    const input = Buffer.from('{"active":true,"deleted":false}')
    const result = extractJSON(input)
    expect(result.content).toContain('true')
    expect(result.content).toContain('false')
  })

  it('handles string values', () => {
    const input = Buffer.from('{"greeting":"hello world"}')
    const result = extractJSON(input)
    expect(result.content).toContain('hello world')
  })

  it('renders below-threshold uniform arrays individually', () => {
    const input = Buffer.from('[{"a":1},{"a":2}]')
    const result = extractJSON(input, { tableThreshold: 5 })
    // Only 2 objects but threshold is 5, so should render individually.
    expect(result.metadata.structure_type).toBe('heterogeneous_object_array')
  })
})

// -------------------------------------------------------------------
// JSONL Extractor
// -------------------------------------------------------------------

describe('extractJSONL', () => {
  it('extracts basic JSONL', () => {
    const input = Buffer.from('{"a":1}\n{"a":2}\n{"a":3}\n')
    const result = extractJSONL(input, { tableThreshold: 3 })
    expect(result.mime).toBe('application/jsonl')
  })

  it('throws for empty input', () => {
    expect(() => extractJSONL(Buffer.alloc(0))).toThrow('empty jsonl')
  })

  it('throws for invalid JSONL', () => {
    expect(() => extractJSONL(Buffer.from('{bad}\n'))).toThrow('invalid jsonl')
  })

  it('handles mixed object types in JSONL', () => {
    const input = Buffer.from('{"name":"Alice"}\n{"age":30}\n')
    const result = extractJSONL(input)
    expect(result.mime).toBe('application/jsonl')
  })
})

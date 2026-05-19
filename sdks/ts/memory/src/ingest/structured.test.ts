// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { extractCSV, extractJSON, extractJSONL } from './structured.js'
import { extractXML } from './xml.js'
import {
  createCSVExtractor,
  createJSONExtractor,
  createJSONLExtractor,
  createXMLExtractor,
} from './extractor.js'

// -------------------------------------------------------------------
// CSV Extractor
// -------------------------------------------------------------------

describe('extractCSV', () => {
  it('extracts basic CSV with headers', () => {
    const input = Buffer.from('Name,Age,City\nAlice,30,London\nBob,25,Paris\n')
    const result = extractCSV(input)
    expect(result.text).toContain('- Name: Alice')
    expect(result.text).toContain('- Age: 30')
    expect(result.text).toContain('- City: London')
    expect(result.metadata.column_count).toBe('3')
    expect(result.metadata.row_count).toBe('2')
  })

  it('auto-detects semicolon delimiter', () => {
    const input = Buffer.from('Name;Age;City\nAlice;30;London\nBob;25;Paris\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe(';')
    expect(result.text).toContain('- Name: Alice')
  })

  it('auto-detects tab delimiter', () => {
    const input = Buffer.from('Name\tAge\tCity\nAlice\t30\tLondon\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe('\t')
    expect(result.contentType).toBe('text/tab-separated-values')
  })

  it('auto-detects pipe delimiter', () => {
    const input = Buffer.from('Name|Age|City\nAlice|30|London\nBob|25|Paris\n')
    const result = extractCSV(input)
    expect(result.metadata.delimiter).toBe('|')
  })

  it('uses synthetic headers when first row is numeric', () => {
    const input = Buffer.from('1,2,3\n4,5,6\n7,8,9\n')
    const result = extractCSV(input)
    expect(result.text).toContain('Column_1')
    expect(result.text).toContain('Column_2')
    expect(result.text).toContain('Column_3')
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
    expect(result.text).toContain('- Name: Alice')
  })

  it('decodes Latin-1 encoding', () => {
    const input = Buffer.from([
      0x4e, 0x61, 0xef, 0x76, 0x65, 0x2c,
      0x43, 0x61, 0x66, 0xe9, 0x0a,
      0x41, 0x6c, 0x69, 0x63, 0x65, 0x2c,
      0x5a, 0xfc, 0x72, 0x69, 0x63, 0x68, 0x0a,
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
    const separators = (result.text.match(/---/g) ?? []).length
    expect(separators).toBe(2)
    expect(result.metadata.row_count).toBe('120')
  })

  it('respects forced delimiter', () => {
    const input = Buffer.from('Name;Age\nAlice;30\n')
    const result = extractCSV(input, { forceDelimiter: ',' })
    expect(result.metadata.delimiter).toBe(',')
  })

  it('uses synthetic headers for duplicate first row values', () => {
    const input = Buffer.from('Name,Name,Age\nAlice,Bob,30\n')
    const result = extractCSV(input)
    expect(result.text).toContain('Column_1')
  })

  it('preserves row numbering across chunks', () => {
    const lines = ['Name,Value']
    for (let i = 0; i < 60; i++) {
      lines.push(`item_${i},${i}`)
    }
    const input = Buffer.from(lines.join('\n'))
    const result = extractCSV(input, { rowsPerChunk: 50 })
    expect(result.text).toContain('Row 1:')
    expect(result.text).toContain('Row 51:')
  })

  it('sanitises formula injection values', () => {
    const input = Buffer.from('Name,Formula\nAlice,=SUM(A1)\nBob,+CMD\nCarol,-1+1\nDave,@INDIRECT(A1)\n')
    const result = extractCSV(input)
    expect(result.text).toContain("'=SUM(A1)")
    expect(result.text).toContain("'+CMD")
    expect(result.text).toContain("'-1+1")
    expect(result.text).toContain("'@INDIRECT(A1)")
  })

  it('does not sanitise safe values', () => {
    const input = Buffer.from('Name,Value\nAlice,hello\nBob,42\n')
    const result = extractCSV(input)
    expect(result.text).not.toContain("'hello")
  })

  it('rejects input exceeding maxInputSize', () => {
    const input = Buffer.from('Name,Age\nAlice,30\n')
    expect(() => extractCSV(input, { maxInputSize: 5 })).toThrow('exceeds')
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
    expect(result.text).toContain('Object with 3 keys')
    expect(result.text).toContain('age:')
  })

  it('renders uniform array of objects as markdown table', () => {
    const input = Buffer.from(
      '[{"name":"Alice","age":30},{"name":"Bob","age":25},{"name":"Carol","age":35}]',
    )
    const result = extractJSON(input, { tableThreshold: 3 })
    expect(result.metadata.structure_type).toBe('uniform_object_array')
    expect(result.text).toContain('| age |')
    expect(result.text).toContain('| name |')
    expect(result.text).toContain('| --- |')
  })

  it('renders heterogeneous array of objects individually', () => {
    const input = Buffer.from('[{"a":1,"b":2},{"c":3,"d":4}]')
    const result = extractJSON(input)
    expect(result.metadata.structure_type).toBe('heterogeneous_object_array')
    expect(result.text).toContain('Item 1:')
  })

  it('flattens deeply nested objects to dot-notation', () => {
    const input = Buffer.from('{"a":{"b":{"c":{"d":{"e":"deep"}}}}}')
    const result = extractJSON(input)
    expect(result.text).toContain('a.b.c.d.e: deep')
  })

  it('handles empty object', () => {
    const input = Buffer.from('{}')
    const result = extractJSON(input)
    expect(result.text).toBe('{}')
    expect(result.metadata.structure_type).toBe('empty_object')
  })

  it('handles empty array', () => {
    const input = Buffer.from('[]')
    const result = extractJSON(input)
    expect(result.text).toBe('[]')
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
    const lines = result.text.split('\n')
    expect(lines).toHaveLength(5)
  })

  it('throws for empty input', () => {
    expect(() => extractJSON(Buffer.alloc(0))).toThrow('empty json')
  })

  it('chunks objects at configured boundaries', () => {
    const objects = Array.from({ length: 120 }, () => '{"id":1,"name":"test"}')
    const input = Buffer.from(`[${objects.join(',')}]`)
    const result = extractJSON(input, { objectsPerChunk: 50, tableThreshold: 3 })
    const separators = (result.text.match(/\n---\n/g) ?? []).length
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
    expect(result.text).toContain('null')
  })

  it('handles boolean values', () => {
    const input = Buffer.from('{"active":true,"deleted":false}')
    const result = extractJSON(input)
    expect(result.text).toContain('true')
    expect(result.text).toContain('false')
  })

  it('handles string values', () => {
    const input = Buffer.from('{"greeting":"hello world"}')
    const result = extractJSON(input)
    expect(result.text).toContain('hello world')
  })

  it('renders below-threshold uniform arrays individually', () => {
    const input = Buffer.from('[{"a":1},{"a":2}]')
    const result = extractJSON(input, { tableThreshold: 5 })
    expect(result.metadata.structure_type).toBe('heterogeneous_object_array')
  })

  it('rejects input exceeding maxInputSize', () => {
    const input = Buffer.from('{"key":"value"}')
    expect(() => extractJSON(input, { maxInputSize: 5 })).toThrow('exceeds')
  })
})

// -------------------------------------------------------------------
// JSONL Extractor
// -------------------------------------------------------------------

describe('extractJSONL', () => {
  it('extracts basic JSONL', () => {
    const input = Buffer.from('{"a":1}\n{"a":2}\n{"a":3}\n')
    const result = extractJSONL(input, { tableThreshold: 3 })
    expect(result.contentType).toBe('application/jsonl')
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
    expect(result.contentType).toBe('application/jsonl')
  })
})

// -------------------------------------------------------------------
// XML Extractor
// -------------------------------------------------------------------

describe('extractXML', () => {
  it('extracts basic XML with element path context', () => {
    const input = Buffer.from('<root><item>hello</item><item>world</item></root>')
    const result = extractXML(input)
    expect(result.contentType).toBe('application/xml')
    expect(result.metadata.root_element).toBe('root')
    expect(result.text).toContain('root/item: hello')
    expect(result.text).toContain('root/item: world')
  })

  it('renders attributes with path context', () => {
    const input = Buffer.from('<root><user id="1" name="Alice">text content</user></root>')
    const result = extractXML(input)
    expect(result.text).toContain('root/user@id: 1')
    expect(result.text).toContain('root/user@name: Alice')
    expect(result.text).toContain('root/user: text content')
  })

  it('strips namespace prefixes', () => {
    const input = Buffer.from('<ns:root xmlns:ns="http://example.com"><ns:item>value</ns:item></ns:root>')
    const result = extractXML(input)
    expect(result.text).toContain('root/item: value')
  })

  it('handles CDATA sections as text', () => {
    const input = Buffer.from('<root><data><![CDATA[some <special> content]]></data></root>')
    const result = extractXML(input)
    expect(result.text).toContain('some <special> content')
  })

  it('ignores processing instructions', () => {
    const input = Buffer.from('<?xml version="1.0"?><root><item>value</item></root>')
    const result = extractXML(input)
    expect(result.text).toContain('root/item: value')
  })

  it('throws for empty input', () => {
    expect(() => extractXML(Buffer.alloc(0))).toThrow('empty xml')
  })

  it('renders nested elements with full path', () => {
    const input = Buffer.from('<root><parent><child><grandchild>deep</grandchild></child></parent></root>')
    const result = extractXML(input)
    expect(result.text).toContain('root/parent/child/grandchild: deep')
  })

  it('chunks top-level children', () => {
    const items = Array.from({ length: 60 }, (_, i) => `<item>value_${i}</item>`).join('')
    const input = Buffer.from(`<root>${items}</root>`)
    const result = extractXML(input, { elementsPerChunk: 50 })
    const separators = (result.text.match(/---/g) ?? []).length
    expect(separators).toBeGreaterThanOrEqual(1)
  })

  it('rejects input exceeding maxInputSize', () => {
    const input = Buffer.from('<root><item>value</item></root>')
    expect(() => extractXML(input, { maxInputSize: 5 })).toThrow('exceeds')
  })
})

// -------------------------------------------------------------------
// Extractor Interface
// -------------------------------------------------------------------

describe('Extractor', () => {
  it('CSV extractor implements the interface', async () => {
    const e = createCSVExtractor()
    expect(e.name).toBe('csv')
    expect(await e.available()).toBe(true)
    expect(e.capability().mimeTypes.length).toBeGreaterThan(0)
    const result = await e.extract(Buffer.from('Name,Age\nAlice,30\n'), {})
    expect(result.text).toContain('- Name: Alice')
  })

  it('JSON extractor implements the interface', async () => {
    const e = createJSONExtractor()
    expect(e.name).toBe('json')
    expect(await e.available()).toBe(true)
    const result = await e.extract(Buffer.from('{"key":"value"}'), {})
    expect(result.text).toContain('key:')
  })

  it('JSONL extractor implements the interface', async () => {
    const e = createJSONLExtractor()
    expect(e.name).toBe('jsonl')
    expect(await e.available()).toBe(true)
    const result = await e.extract(Buffer.from('{"a":1}\n{"a":2}\n'), {})
    expect(result.contentType).toBe('application/jsonl')
  })

  it('XML extractor implements the interface', async () => {
    const e = createXMLExtractor()
    expect(e.name).toBe('xml')
    expect(await e.available()).toBe(true)
    expect(e.capability().mimeTypes.length).toBeGreaterThan(0)
    const result = await e.extract(Buffer.from('<root><item>hello</item></root>'), {})
    expect(result.text).toContain('root/item: hello')
  })
})

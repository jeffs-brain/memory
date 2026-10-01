// SPDX-License-Identifier: Apache-2.0

/**
 * Every tool's advertised input schema must match `spec/MCP-TOOLS.md`:
 * the same top-level fields, the same required set, and matching types
 * and enum values. The Go and Python servers run the same check.
 */

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { toJsonSchema } from './server.js'
import { tools } from './tools/index.js'

type SpecField = {
  readonly name: string
  readonly required: boolean
  readonly type: 'string' | 'integer' | 'number' | 'boolean' | 'array'
  readonly enumValues?: readonly string[]
}

const specTypeOf = (raw: string): Pick<SpecField, 'type' | 'enumValues'> => {
  const decl = raw.split('#')[0]?.trim() ?? ''
  if (decl.startsWith("'")) {
    return { type: 'string', enumValues: [...decl.matchAll(/'([^']+)'/g)].map((m) => m[1] ?? '') }
  }
  for (const type of ['string', 'integer', 'number', 'boolean'] as const) {
    if (decl.startsWith(type)) return { type }
  }
  if (decl.startsWith('Array<')) return { type: 'array' }
  throw new Error(`unrecognised spec type: ${raw}`)
}

/** Top-level input fields per tool, parsed from the spec's shorthand. */
const parseSpec = (markdown: string): Map<string, SpecField[]> => {
  const specs = new Map<string, SpecField[]>()
  for (const section of markdown.split(/^## /m)) {
    const name = section.match(/^`(memory_[a-z_]+)`/)?.[1]
    if (name === undefined) continue
    const block = section.match(/\*\*Input schema\*\*(?::\s*`\{\}`|\s*```[a-z]*\n([\s\S]*?)```)/)
    if (block === null) throw new Error(`${name}: no input schema in the spec`)
    const fields = [...(block[1] ?? '').matchAll(/^ {2}([A-Za-z_]+)(\?)?:\s*(.+)$/gm)].map(
      ([, field, optional, rest]) => ({
        name: field ?? '',
        required: optional === undefined,
        ...specTypeOf(rest ?? ''),
      }),
    )
    specs.set(name, fields)
  }
  return specs
}

type JsonProperty = { readonly type?: string; readonly enum?: readonly string[] }
type JsonObjectSchema = {
  readonly type: string
  readonly properties: Readonly<Record<string, JsonProperty>>
  readonly required?: readonly string[]
}

const spec = parseSpec(readFileSync(new URL('../../../spec/MCP-TOOLS.md', import.meta.url), 'utf8'))

describe('tool schemas match spec/MCP-TOOLS.md', () => {
  it('advertises exactly the spec tools', () => {
    expect(tools.map((tool) => tool.name).sort()).toEqual([...spec.keys()].sort())
  })

  it.each(tools.map((tool) => [tool.name, tool] as const))('%s', (name, tool) => {
    const fields = spec.get(name) ?? []
    const schema = toJsonSchema(tool.inputSchema) as JsonObjectSchema
    expect(schema.type).toBe('object')
    expect(Object.keys(schema.properties).sort()).toEqual(fields.map((f) => f.name).sort())
    expect([...(schema.required ?? [])].sort()).toEqual(
      fields
        .filter((f) => f.required)
        .map((f) => f.name)
        .sort(),
    )
    for (const field of fields) {
      const property = schema.properties[field.name]
      // An integer satisfies a spec `number`.
      const accepted = field.type === 'number' ? ['number', 'integer'] : [field.type]
      expect(accepted, `${name}.${field.name}`).toContain(property?.type)
      if (field.enumValues !== undefined) {
        expect([...(property?.enum ?? [])].sort(), `${name}.${field.name}`).toEqual(
          [...field.enumValues].sort(),
        )
      }
    }
  })
})

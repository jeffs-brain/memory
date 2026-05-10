// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import {
  INDUSTRY_TEMPLATES,
  getTemplate,
  isValidBusinessCategory,
  isValidEdgeType,
  isValidNodeType,
  listTemplates,
} from './templates.js'

describe('listTemplates', () => {
  it('returns exactly 6 keys in sorted order', () => {
    const keys = listTemplates()
    expect(keys).toEqual([
      'finance',
      'healthcare',
      'insurance',
      'logistics',
      'order_processing',
      'server_hardware',
    ])
  })
})

describe('getTemplate', () => {
  it('returns correct template for each key', () => {
    const keys = listTemplates()
    for (const key of keys) {
      const tmpl = getTemplate(key)
      expect(tmpl).toBeDefined()
      expect(tmpl!.label).toBeTruthy()
      expect(tmpl!.description).toBeTruthy()
    }
  })

  it('returns undefined for unknown key', () => {
    const tmpl = getTemplate('nonexistent_template')
    expect(tmpl).toBeUndefined()
  })
})

describe('template counts', () => {
  const expectedCounts: Record<string, { nodes: number; edges: number; categories: number }> = {
    server_hardware: { nodes: 9, edges: 6, categories: 1 },
    insurance: { nodes: 20, edges: 10, categories: 7 },
    logistics: { nodes: 27, edges: 11, categories: 8 },
    finance: { nodes: 22, edges: 12, categories: 9 },
    healthcare: { nodes: 30, edges: 15, categories: 9 },
    order_processing: { nodes: 8, edges: 3, categories: 1 },
  }

  for (const [key, counts] of Object.entries(expectedCounts)) {
    it(`${key} has correct type counts`, () => {
      const tmpl = getTemplate(key)
      expect(tmpl).toBeDefined()
      expect(tmpl!.nodeTypes.length).toBe(counts.nodes)
      expect(tmpl!.edgeTypes.length).toBe(counts.edges)
      expect(tmpl!.businessCategories.length).toBe(counts.categories)
    })
  }
})

describe('template node types valid', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} node types all pass validation`, () => {
      for (const nt of tmpl.nodeTypes) {
        expect(isValidNodeType(nt.type)).toBe(true)
        expect(nt.label).toBeTruthy()
        expect(nt.description).toBeTruthy()
      }
    })
  }
})

describe('template edge types valid', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} edge types all pass validation`, () => {
      for (const et of tmpl.edgeTypes) {
        expect(isValidEdgeType(et.type)).toBe(true)
        expect(et.label).toBeTruthy()
        expect(et.description).toBeTruthy()
      }
    })
  }
})

describe('template business categories valid', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} business categories all pass validation`, () => {
      for (const cat of tmpl.businessCategories) {
        expect(isValidBusinessCategory(cat)).toBe(true)
      }
    })
  }
})

describe('no duplicate node types within a template', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} has no duplicate node types`, () => {
      const types = tmpl.nodeTypes.map((nt) => nt.type)
      const unique = new Set(types)
      expect(unique.size).toBe(types.length)
    })
  }
})

describe('no duplicate edge types within a template', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} has no duplicate edge types`, () => {
      const types = tmpl.edgeTypes.map((et) => et.type)
      const unique = new Set(types)
      expect(unique.size).toBe(types.length)
    })
  }
})

describe('no duplicate business categories within a template', () => {
  for (const [key, tmpl] of Object.entries(INDUSTRY_TEMPLATES)) {
    it(`${key} has no duplicate business categories`, () => {
      const unique = new Set(tmpl.businessCategories)
      expect(unique.size).toBe(tmpl.businessCategories.length)
    })
  }
})

describe('cross-SDK parity', () => {
  it('has exactly 6 templates', () => {
    expect(Object.keys(INDUSTRY_TEMPLATES).length).toBe(6)
  })

  it('template names match Go SDK', () => {
    const goTemplateNames = [
      'finance',
      'healthcare',
      'insurance',
      'logistics',
      'order_processing',
      'server_hardware',
    ]
    expect(listTemplates()).toEqual(goTemplateNames)
  })
})

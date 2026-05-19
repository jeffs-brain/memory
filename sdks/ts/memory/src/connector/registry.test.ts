// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  ConnectorRegistry,
  ConnectorNotFoundError,
  ConnectorExistsError,
} from './registry.js'
import type { Connector, ConnectorConfig, ConnectorDocument, SyncCursor } from './types.js'

const stubConnector = (name: string): Connector => ({
  name,
  configure: async () => {},
  fetchAll: async function* () {},
  fetchSince: async function* () {},
  start: async () => {},
  stop: async () => {},
})

const stubFactory = (name: string) => (_cfg: ConnectorConfig): Connector => stubConnector(name)

describe('ConnectorRegistry', () => {
  it('registers and looks up a connector', () => {
    const reg = new ConnectorRegistry()
    reg.register('slack', stubFactory('slack'))

    const conn = reg.lookup('slack', { name: 'slack', brainId: 'b1' } as ConnectorConfig)
    expect(conn.name).toBe('slack')
  })

  it('throws on duplicate registration', () => {
    const reg = new ConnectorRegistry()
    reg.register('slack', stubFactory('slack'))
    expect(() => reg.register('slack', stubFactory('slack'))).toThrow(ConnectorExistsError)
  })

  it('throws on lookup of unknown connector', () => {
    const reg = new ConnectorRegistry()
    expect(() => reg.lookup('nonexistent', {} as ConnectorConfig)).toThrow(ConnectorNotFoundError)
  })

  it('lists registered names', () => {
    const reg = new ConnectorRegistry()
    reg.register('slack', stubFactory('slack'))
    reg.register('gdrive', stubFactory('gdrive'))

    const names = [...reg.names()].sort()
    expect(names).toEqual(['gdrive', 'slack'])
  })

  it('reports has correctly', () => {
    const reg = new ConnectorRegistry()
    reg.register('slack', stubFactory('slack'))

    expect(reg.has('slack')).toBe(true)
    expect(reg.has('notion')).toBe(false)
  })
})

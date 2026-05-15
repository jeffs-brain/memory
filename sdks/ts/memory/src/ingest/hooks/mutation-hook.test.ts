// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type { ChangeEvent } from '../../store/index.js'
import type { Path } from '../../store/index.js'
import {
  createMutationHook,
  defaultPathMatcher,
  globPathMatcher,
  prefixPathMatcher,
} from './mutation-hook.js'

const makeEvent = (
  kind: ChangeEvent['kind'],
  path: string,
  reason?: string,
): ChangeEvent => ({
  kind,
  path: path as Path,
  reason,
  when: new Date(),
})

describe('mutation-hook', () => {
  it('dispatches on created event in raw/documents/', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', 'raw/documents/readme.md'))
    await delay(50)

    expect(dispatch).toHaveBeenCalledWith('brain-1', 'raw/documents/readme.md')
    hook.close()
  })

  it('dispatches on updated event in raw/documents/', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('updated', 'raw/documents/notes.md'))
    await delay(50)

    expect(dispatch).toHaveBeenCalledTimes(1)
    hook.close()
  })

  it('ignores deleted events', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('deleted', 'raw/documents/readme.md'))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()
    hook.close()
  })

  it('ignores paths outside default pattern', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', 'memory/global/fact.md'))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()
    hook.close()
  })

  it('supports custom path matchers', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      pathMatchers: [prefixPathMatcher('custom/')],
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', 'custom/data.json'))
    await delay(50)

    expect(dispatch).toHaveBeenCalledWith('brain-1', 'custom/data.json')
    hook.close()
  })

  it('debounces rapid writes to the same path', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 100,
    })

    for (let i = 0; i < 5; i++) {
      hook.sink(makeEvent('updated', 'raw/documents/rapid.md'))
      await delay(10)
    }

    await delay(200)

    expect(dispatch).toHaveBeenCalledTimes(1)
    hook.close()
  })

  it('dispatches independently for different paths', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', 'raw/documents/a.md'))
    hook.sink(makeEvent('created', 'raw/documents/b.md'))
    await delay(50)

    expect(dispatch).toHaveBeenCalledTimes(2)
    hook.close()
  })

  it('opts out by batch reason', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      optOutReasons: new Set(['pipeline', 'ingest']),
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', 'raw/documents/pipeline-output.md', 'pipeline'))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()

    // Non-opt-out reason should dispatch.
    hook.sink(makeEvent('created', 'raw/documents/user-upload.md', 'user-write'))
    await delay(50)

    expect(dispatch).toHaveBeenCalledTimes(1)
    hook.close()
  })

  it('close() stops pending timers', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 200,
    })

    hook.sink(makeEvent('created', 'raw/documents/closing.md'))
    hook.close()

    await delay(300)
    expect(dispatch).not.toHaveBeenCalled()
  })

  it('events after close are ignored', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.close()
    hook.sink(makeEvent('created', 'raw/documents/late.md'))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()
  })

  it('rejects paths containing ".."', async () => {
    const dispatch = vi.fn()
    const warnFn = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
      logger: { debug: vi.fn(), info: vi.fn(), warn: warnFn, error: vi.fn() },
    })

    hook.sink(makeEvent('created', 'raw/documents/../../../etc/passwd'))
    hook.sink(makeEvent('created', 'raw/documents/..secret.md'))
    hook.sink(makeEvent('created', '../raw/documents/readme.md'))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()
    expect(warnFn).toHaveBeenCalledTimes(3)
    hook.close()
  })

  it('ignores empty path', async () => {
    const dispatch = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch,
      debounceIntervalMs: 10,
    })

    hook.sink(makeEvent('created', ''))
    await delay(50)

    expect(dispatch).not.toHaveBeenCalled()
    hook.close()
  })

  it('dispatch errors are logged', async () => {
    const errorFn = vi.fn()
    const hook = createMutationHook({
      brainId: 'brain-1',
      dispatch: () => { throw new Error('dispatch fail') },
      debounceIntervalMs: 10,
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: errorFn },
    })

    hook.sink(makeEvent('created', 'raw/documents/failing.md'))
    await delay(50)

    expect(errorFn).toHaveBeenCalled()
    hook.close()
  })
})

describe('defaultPathMatcher', () => {
  it('matches raw/documents/ paths', () => {
    expect(defaultPathMatcher('raw/documents/readme.md')).toBe(true)
    expect(defaultPathMatcher('raw/documents/sub/deep.md')).toBe(true)
  })

  it('rejects non-matching paths', () => {
    expect(defaultPathMatcher('memory/global/fact.md')).toBe(false)
    expect(defaultPathMatcher('raw/documents/')).toBe(false)
  })

  it('rejects empty path', () => {
    expect(defaultPathMatcher('')).toBe(false)
  })
})

describe('globPathMatcher', () => {
  const matcher = globPathMatcher('raw/documents/**/*.md')

  it('matches markdown files under raw/documents/', () => {
    expect(matcher('raw/documents/readme.md')).toBe(true)
    expect(matcher('raw/documents/sub/notes.md')).toBe(true)
    expect(matcher('raw/documents/sub/deep/nested.md')).toBe(true)
  })

  it('rejects non-matching files', () => {
    expect(matcher('raw/documents/readme.txt')).toBe(false)
    expect(matcher('memory/global/fact.md')).toBe(false)
  })

  it('matches ** at end of glob', () => {
    const allMatcher = globPathMatcher('raw/documents/**')
    expect(allMatcher('raw/documents/readme.md')).toBe(true)
    expect(allMatcher('raw/documents/sub/notes.md')).toBe(true)
    expect(allMatcher('memory/global/fact.md')).toBe(false)
  })
})

describe('prefixPathMatcher boundary', () => {
  const matcher = prefixPathMatcher('custom/')

  it('matches paths longer than the prefix', () => {
    expect(matcher('custom/data.json')).toBe(true)
  })

  it('rejects bare prefix', () => {
    expect(matcher('custom/')).toBe(false)
  })

  it('rejects shorter-than-prefix', () => {
    expect(matcher('custom')).toBe(false)
  })

  it('rejects empty path', () => {
    expect(matcher('')).toBe(false)
  })
})

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

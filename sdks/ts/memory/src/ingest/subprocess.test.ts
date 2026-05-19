// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from 'vitest'
import { checkBinaryAvailable, resetBinaryCache, runSubprocess } from './subprocess.js'

afterEach(() => {
  resetBinaryCache()
})

describe('runSubprocess', () => {
  it('runs a command and captures stdout', async () => {
    const result = await runSubprocess('/bin/echo', ['hello'])
    expect(result.exitCode).toBe(0)
    expect(result.stdout.toString('utf-8').trim()).toBe('hello')
  })

  it('captures non-zero exit codes', async () => {
    const result = await runSubprocess('sh', ['-c', 'exit 42'])
    expect(result.exitCode).toBe(42)
  })

  it('passes stdin to process', async () => {
    const result = await runSubprocess('cat', [], Buffer.from('stdin data'))
    expect(result.exitCode).toBe(0)
    expect(result.stdout.toString('utf-8')).toBe('stdin data')
  })

  it('enforces timeout', async () => {
    await expect(runSubprocess('sleep', ['30'], undefined, { timeout: 100 })).rejects.toThrow(
      'timed out',
    )
  })

  it('throws for nonexistent binary', async () => {
    await expect(runSubprocess('nonexistent-binary-xyz-abc', [])).rejects.toThrow()
  })

  it('captures stderr', async () => {
    const result = await runSubprocess('sh', ['-c', 'echo error >&2; exit 1'])
    expect(result.exitCode).toBe(1)
    expect(result.stderr).toContain('error')
  })
})

describe('checkBinaryAvailable', () => {
  // Use /bin/ls which is a real binary on PATH (not a shell builtin).
  it('returns true for ls', async () => {
    const result = await checkBinaryAvailable('ls')
    expect(result).toBe(true)
  })

  it('returns false for nonexistent binary', async () => {
    const result = await checkBinaryAvailable('nonexistent-binary-xyz-abc')
    expect(result).toBe(false)
  })

  it('caches results', async () => {
    const result1 = await checkBinaryAvailable('ls')
    const result2 = await checkBinaryAvailable('ls')
    expect(result1).toBe(result2)
  })
})

describe('resetBinaryCache', () => {
  it('clears cached results without error', () => {
    expect(() => resetBinaryCache()).not.toThrow()
  })

  it('allows re-probing after reset', async () => {
    await checkBinaryAvailable('ls')
    resetBinaryCache()
    const result = await checkBinaryAvailable('ls')
    expect(result).toBe(true)
  })
})

// SPDX-License-Identifier: Apache-2.0

/**
 * Subprocess runner utilities shared by all extraction backends that
 * invoke external binaries (FFmpeg, faster-whisper, Tesseract, etc.).
 */

import { execFile, type ExecFileException } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

export type SubprocessOptions = {
  readonly timeout?: number
  readonly cwd?: string
  readonly env?: Readonly<Record<string, string>>
  readonly signal?: AbortSignal
}

export type SubprocessResult = {
  readonly stdout: Buffer
  readonly stderr: string
  readonly exitCode: number
}

/**
 * Run a subprocess and capture its output. Returns the stdout as a
 * Buffer, stderr as a string, and the exit code.
 */
export const runSubprocess = async (
  binary: string,
  args: readonly string[],
  input?: Buffer,
  opts?: SubprocessOptions,
): Promise<SubprocessResult> => {
  const timeout = opts?.timeout ?? 120_000
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  // If the caller provided a signal, forward its abort.
  const callerSignal = opts?.signal
  const onCallerAbort = (): void => controller.abort()
  callerSignal?.addEventListener('abort', onCallerAbort, { once: true })

  try {
    const result = await execFileAsync(binary, [...args], {
      encoding: 'buffer',
      maxBuffer: 512 * 1024 * 1024,
      timeout,
      signal: controller.signal,
      ...(opts?.cwd !== undefined ? { cwd: opts.cwd } : {}),
      ...(opts?.env !== undefined ? { env: { ...process.env, ...opts.env } } : {}),
      ...(input !== undefined ? { input } : {}),
    })
    return {
      stdout: result.stdout,
      stderr: result.stderr.toString('utf8'),
      exitCode: 0,
    }
  } catch (err: unknown) {
    const execErr = err as ExecFileException & {
      stdout?: Buffer
      stderr?: Buffer
    }
    if (execErr.code === 'ENOENT') {
      throw new Error(`extract: binary not found: ${binary}`)
    }
    return {
      stdout: execErr.stdout ?? Buffer.alloc(0),
      stderr: execErr.stderr?.toString('utf8') ?? execErr.message,
      exitCode: execErr.code !== undefined && typeof execErr.code === 'number' ? execErr.code : 1,
    }
  } finally {
    clearTimeout(timeoutId)
    callerSignal?.removeEventListener('abort', onCallerAbort)
  }
}

const binaryCache = new Map<string, { available: boolean; checkedAt: number }>()
const BINARY_CACHE_TTL_MS = 5 * 60 * 1000

/**
 * Check whether a binary is available on PATH. Results are cached for
 * five minutes to avoid repeated filesystem lookups.
 */
export const checkBinaryAvailable = async (name: string): Promise<boolean> => {
  const cached = binaryCache.get(name)
  if (cached !== undefined && Date.now() - cached.checkedAt < BINARY_CACHE_TTL_MS) {
    return cached.available
  }

  try {
    await execFileAsync('which', [name], { timeout: 5_000 })
    binaryCache.set(name, { available: true, checkedAt: Date.now() })
    return true
  } catch {
    binaryCache.set(name, { available: false, checkedAt: Date.now() })
    return false
  }
}

/**
 * Clear the binary availability cache. Intended for tests only.
 */
export const resetBinaryCache = (): void => {
  binaryCache.clear()
}

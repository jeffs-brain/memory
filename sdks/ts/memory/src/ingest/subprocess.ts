// SPDX-License-Identifier: Apache-2.0

/**
 * Shared subprocess runner utility for Phase 4 extractors. Provides
 * timeout-enforced subprocess execution and binary availability
 * checking with caching. Reused by image OCR (P4-1), audio
 * transcription (P4-2), and video processing (P4-3).
 */

import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

/** Result of a subprocess invocation. */
export type SubprocessResult = {
  readonly stdout: Buffer
  readonly stderr: string
  readonly exitCode: number
}

/** Configuration for subprocess invocation. */
export type SubprocessOptions = {
  readonly timeout?: number
  readonly cwd?: string
  readonly env?: Readonly<Record<string, string>>
  readonly signal?: AbortSignal
  readonly maxBuffer?: number
}

/** Default subprocess timeout in milliseconds (60 seconds). */
export const DEFAULT_SUBPROCESS_TIMEOUT_MS = 60_000

/** Default maximum buffer size for subprocess output (50 MB). */
const DEFAULT_MAX_BUFFER = 50 * 1024 * 1024

/**
 * Maximum number of simultaneous subprocess invocations. Prevents
 * resource exhaustion when many extractions run concurrently.
 */
export const MAX_CONCURRENT_SUBPROCESSES = 8

/**
 * Simple counting semaphore that limits concurrency. Callers await
 * acquire() before spawning a subprocess and call release() when done.
 */
const createSemaphore = (maxConcurrency: number) => {
  let current = 0
  const waiting: Array<() => void> = []

  const acquire = (): Promise<void> => {
    if (current < maxConcurrency) {
      current++
      return Promise.resolve()
    }
    return new Promise<void>((resolve) => {
      waiting.push(resolve)
    })
  }

  const release = (): void => {
    const next = waiting.shift()
    if (next !== undefined) {
      next()
    } else {
      current--
    }
  }

  return { acquire, release }
}

const subprocessSemaphore = createSemaphore(MAX_CONCURRENT_SUBPROCESSES)

/**
 * Executes a binary with the given arguments and optional stdin input.
 * Arguments are passed as an array — no shell expansion occurs.
 * Timeout is enforced via setTimeout + process.kill().
 *
 * Uses spawn() for stdin support since execFile's `input` option can
 * hang when combined with promisify.
 */
export const runSubprocess = async (
  binary: string,
  args: readonly string[],
  stdin?: Buffer,
  opts?: SubprocessOptions,
): Promise<SubprocessResult> => {
  await subprocessSemaphore.acquire()

  const timeout = opts?.timeout ?? DEFAULT_SUBPROCESS_TIMEOUT_MS
  const maxBuffer = opts?.maxBuffer ?? DEFAULT_MAX_BUFFER

  return new Promise<SubprocessResult>((resolve, reject) => {
    let child: ChildProcess
    let timedOut = false

    try {
      child = spawn(binary, [...args], {
        cwd: opts?.cwd,
        env: opts?.env ? { ...process.env, ...opts.env } : undefined,
        stdio: ['pipe', 'pipe', 'pipe'],
        signal: opts?.signal,
      })
    } catch (err: unknown) {
      subprocessSemaphore.release()
      reject(
        new Error(
          `ingest: subprocess "${binary}": ${err instanceof Error ? err.message : String(err)}`,
        ),
      )
      return
    }

    const timer = setTimeout(() => {
      timedOut = true
      child.kill('SIGKILL')
    }, timeout)

    const stdoutChunks: Buffer[] = []
    const stderrChunks: Buffer[] = []
    let stdoutBytes = 0
    let stderrBytes = 0

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBytes += chunk.length
      if (stdoutBytes <= maxBuffer) {
        stdoutChunks.push(chunk)
      }
    })

    child.stderr?.on('data', (chunk: Buffer) => {
      stderrBytes += chunk.length
      if (stderrBytes <= maxBuffer) {
        stderrChunks.push(chunk)
      }
    })

    child.on('error', (err: Error) => {
      clearTimeout(timer)
      subprocessSemaphore.release()
      reject(new Error(`ingest: subprocess "${binary}": ${err.message}`))
    })

    child.on('close', (code: number | null) => {
      clearTimeout(timer)
      subprocessSemaphore.release()

      if (timedOut) {
        reject(new Error(`ingest: subprocess "${binary}" timed out after ${timeout}ms`))
        return
      }

      resolve({
        stdout: Buffer.concat(stdoutChunks),
        stderr: Buffer.concat(stderrChunks).toString('utf-8'),
        exitCode: code ?? 1,
      })
    })

    // Write stdin if provided, then close the stream.
    if (stdin !== undefined && child.stdin !== null) {
      child.stdin.end(stdin)
    } else if (child.stdin !== null) {
      child.stdin.end()
    }
  })
}

/** Cache entry for binary availability. */
type BinaryCacheEntry = {
  readonly available: boolean
  readonly expiresAt: number
}

/** Cache TTL in milliseconds (5 minutes). */
const BINARY_CACHE_TTL_MS = 5 * 60 * 1000

const binaryCache = new Map<string, BinaryCacheEntry>()

/**
 * Reports whether the named binary is present on the system PATH.
 * Results are cached for 5 minutes.
 */
export const checkBinaryAvailable = async (name: string): Promise<boolean> => {
  const cached = binaryCache.get(name)
  if (cached !== undefined && Date.now() < cached.expiresAt) {
    return cached.available
  }

  const available = await probeBinary(name)

  binaryCache.set(name, {
    available,
    expiresAt: Date.now() + BINARY_CACHE_TTL_MS,
  })

  return available
}

/**
 * Clears the binary availability cache. Intended for testing only.
 */
export const resetBinaryCache = (): void => {
  binaryCache.clear()
}

/**
 * Probes for a binary by attempting to run `which` (Unix) or
 * `where` (Windows).
 */
const probeBinary = async (name: string): Promise<boolean> => {
  const whichBinary = process.platform === 'win32' ? 'where' : 'which'
  try {
    await execFileAsync(whichBinary, [name], { timeout: 5000 })
    return true
  } catch {
    return false
  }
}

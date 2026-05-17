// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { describe, expect, it, vi } from 'vitest'
import {
  createAudioExtractor,
  formatTranscription,
  type AudioExtractorConfig,
  type TranscriptionSegment,
} from './audio.js'
import { createExtractorRegistry } from './extractor.js'

/**
 * Creates a temporary shell script that simulates a faster-whisper
 * subprocess. The script ignores all arguments and outputs according
 * to the specified mode. Returns the script path and a cleanup function.
 */
const createMockScript = async (
  mode: 'success' | 'language_detect' | 'corrupt' | 'timeout' | 'availability_ok' | 'availability_fail',
): Promise<{ scriptPath: string; cleanup: () => Promise<void> }> => {
  const dir = await mkdtemp(join(tmpdir(), 'audio-test-'))

  const scripts: Record<string, string> = {
    success: `#!/bin/sh
echo '{"language":"en","language_probability":0.98,"duration":15.5,"segments":[{"start":0,"end":5.0,"text":"Hello world from the test.","words":[{"start":0,"end":0.5,"word":"Hello","probability":0.99},{"start":0.6,"end":1.0,"word":"world","probability":0.97}]},{"start":5.0,"end":15.5,"text":"This is a longer segment for testing."}]}'
`,
    language_detect: `#!/bin/sh
echo '{"language":"fr","language_probability":0.92,"duration":10.0,"segments":[{"start":0,"end":10.0,"text":"Bonjour le monde."}]}'
`,
    corrupt: `#!/bin/sh
echo "Error: could not decode audio file" >&2
exit 1
`,
    timeout: `#!/bin/sh
sleep 60
`,
    availability_ok: `#!/bin/sh
echo "ok"
`,
    availability_fail: `#!/bin/sh
echo "ModuleNotFoundError: No module named 'faster_whisper'" >&2
exit 1
`,
  }

  const scriptPath = join(dir, 'mock-python.sh')
  await writeFile(scriptPath, scripts[mode], { mode: 0o755 })

  return {
    scriptPath,
    cleanup: () => rm(dir, { recursive: true, force: true }),
  }
}

describe('createAudioExtractor', () => {
  const ext = createAudioExtractor()

  it('has the correct name', () => {
    expect(ext.name).toBe('audio-whisper')
  })

  it('declares audio content types', () => {
    expect(ext.contentTypes).toContain('audio/mpeg')
    expect(ext.contentTypes).toContain('audio/wav')
    expect(ext.contentTypes).toContain('audio/x-wav')
    expect(ext.contentTypes).toContain('audio/flac')
    expect(ext.contentTypes).toContain('audio/mp4')
    expect(ext.contentTypes).toContain('audio/ogg')
    expect(ext.contentTypes).toContain('audio/aac')
    expect(ext.contentTypes).toContain('audio/x-ms-wma')
    expect(ext.contentTypes).toContain('audio/opus')
  })

  it('reports capability with extensions and MIME types', () => {
    const cap = ext.capability()
    expect(cap.extensions).toContain('.mp3')
    expect(cap.extensions).toContain('.wav')
    expect(cap.extensions).toContain('.flac')
    expect(cap.extensions).toContain('.m4a')
    expect(cap.extensions).toContain('.ogg')
    expect(cap.extensions).toContain('.aac')
    expect(cap.requiresBinary).toBe(true)
    expect(cap.magicBytes.length).toBeGreaterThan(0)
  })

  it('reports MIME types in capability matching contentTypes', () => {
    const cap = ext.capability()
    for (const ct of ext.contentTypes) {
      expect(cap.mimeTypes).toContain(ct)
    }
  })
})

describe('createAudioExtractor - availability', () => {
  it('returns false when python binary does not exist', async () => {
    const ext = createAudioExtractor({ pythonBinary: '/nonexistent/python3' })
    const available = await ext.available()
    expect(available).toBe(false)
  })

  it('caches availability result on repeated calls', async () => {
    const ext = createAudioExtractor({ pythonBinary: '/nonexistent/python3' })
    const first = await ext.available()
    const second = await ext.available()
    expect(first).toBe(second)
    expect(first).toBe(false)
  })
})

describe('createAudioExtractor - extract', () => {
  it('returns skipped for empty input', async () => {
    const ext = createAudioExtractor()
    const result = await ext.extract(Buffer.alloc(0), { contentType: 'audio/wav' })
    expect(result.skipped).toBe(true)
    expect(result.reason).toBe('empty audio input')
    expect(result.text).toBe('')
  })

  it('throws for file exceeding max size', async () => {
    const ext = createAudioExtractor({ maxFileSizeBytes: 100 })
    const largeBuffer = Buffer.alloc(200)
    await expect(
      ext.extract(largeBuffer, { contentType: 'audio/wav' }),
    ).rejects.toThrow('exceeds maximum')
  })

  it('throws for stream exceeding max size', async () => {
    const ext = createAudioExtractor({ maxFileSizeBytes: 50 })
    const source = Readable.from([Buffer.alloc(100)])
    await expect(
      ext.extractStream(source, { contentType: 'audio/wav' }),
    ).rejects.toThrow('exceeds maximum')
  })
})

describe('createAudioExtractor - custom config', () => {
  it('accepts custom python binary path', () => {
    const ext = createAudioExtractor({ pythonBinary: '/usr/local/bin/python3.11' })
    expect(ext.name).toBe('audio-whisper')
  })

  it('accepts custom model size', () => {
    const ext = createAudioExtractor({ modelSize: 'large-v3' })
    expect(ext.name).toBe('audio-whisper')
  })

  it('accepts custom language', () => {
    const ext = createAudioExtractor({ defaultLanguage: 'fr' })
    expect(ext.name).toBe('audio-whisper')
  })

  it('accepts custom timeouts', () => {
    const ext = createAudioExtractor({
      timeoutPerMinuteMs: 60_000,
      minTimeoutMs: 60_000,
    })
    expect(ext.name).toBe('audio-whisper')
  })
})

describe('formatTranscription', () => {
  it('returns empty string for empty segments', () => {
    expect(formatTranscription([])).toBe('')
  })

  it('formats a single segment with timestamp', () => {
    const segments: TranscriptionSegment[] = [
      { start: 0, end: 5, text: 'Hello world' },
    ]
    const result = formatTranscription(segments)
    expect(result).toContain('[00:00')
    expect(result).toContain('Hello world')
  })

  it('inserts paragraph breaks every 30 seconds', () => {
    const segments: TranscriptionSegment[] = [
      { start: 0, end: 10, text: 'First segment.' },
      { start: 10, end: 20, text: 'Second segment.' },
      { start: 35, end: 45, text: 'Third segment after break.' },
      { start: 45, end: 55, text: 'Fourth segment.' },
    ]
    const result = formatTranscription(segments)
    const parts = result.split('\n\n')
    expect(parts.length).toBeGreaterThanOrEqual(2)
    expect(parts[0]).toContain('First segment.')
    expect(result).toContain('Third segment after break.')
  })

  it('formats timestamps correctly for times over a minute', () => {
    const segments: TranscriptionSegment[] = [
      { start: 65, end: 90, text: 'One minute five.' },
    ]
    const result = formatTranscription(segments)
    expect(result).toContain('[01:05')
  })

  it('formats timestamps for hour-long content', () => {
    const segments: TranscriptionSegment[] = [
      { start: 3661, end: 3690, text: 'Over an hour in.' },
    ]
    const result = formatTranscription(segments)
    expect(result).toContain('[61:01')
  })

  it('handles multiple segments within same paragraph', () => {
    const segments: TranscriptionSegment[] = [
      { start: 0, end: 5, text: 'Hello' },
      { start: 5, end: 10, text: 'world' },
      { start: 10, end: 15, text: 'test' },
    ]
    const result = formatTranscription(segments)
    // All within 30s window, should be one paragraph.
    expect(result.split('\n\n')).toHaveLength(1)
    expect(result).toContain('Hello')
    expect(result).toContain('world')
    expect(result).toContain('test')
  })
})

describe('whisper output parsing', () => {
  it('parses well-formed whisper JSON', () => {
    const jsonStr = JSON.stringify({
      language: 'fr',
      language_probability: 0.95,
      duration: 65.3,
      segments: [
        { start: 0.0, end: 30.0, text: 'Bonjour le monde' },
        { start: 30.0, end: 65.3, text: 'Comment allez-vous' },
      ],
    })

    const output = JSON.parse(jsonStr) as {
      language: string
      language_probability: number
      duration: number
      segments: TranscriptionSegment[]
    }
    expect(output.language).toBe('fr')
    expect(output.duration).toBe(65.3)
    expect(output.segments).toHaveLength(2)

    const text = formatTranscription(output.segments)
    expect(text).toContain('Bonjour le monde')
    expect(text).toContain('Comment allez-vous')
  })

  it('parses word-level timestamps', () => {
    const output = {
      language: 'en',
      language_probability: 0.99,
      duration: 5.0,
      segments: [
        {
          start: 0.0,
          end: 5.0,
          text: 'Hello world',
          words: [
            { start: 0.0, end: 0.5, word: 'Hello', probability: 0.99 },
            { start: 0.6, end: 1.0, word: 'world', probability: 0.97 },
          ],
        },
      ],
    }

    expect(output.segments[0]!.words).toHaveLength(2)
    expect(output.segments[0]!.words![0]!.word).toBe('Hello')
    expect(output.segments[0]!.words![1]!.probability).toBe(0.97)
  })
})

describe('createAudioExtractor - registry integration', () => {
  it('registers with extractor registry and routes audio types', async () => {
    const registry = createExtractorRegistry()
    const ext = createAudioExtractor()
    registry.register(ext)

    // Empty input should be handled (skipped), not unsupported.
    const result = await registry.extract(Buffer.alloc(0), {
      contentType: 'audio/mpeg',
    })
    expect(result.reason).toBe('empty audio input')
  })

  it('routes audio/wav to audio extractor', async () => {
    const registry = createExtractorRegistry()
    const ext = createAudioExtractor()
    registry.register(ext)

    const result = await registry.extract(Buffer.alloc(0), {
      contentType: 'audio/wav',
    })
    expect(result.reason).toBe('empty audio input')
  })

  it('routes audio/ogg to audio extractor', async () => {
    const registry = createExtractorRegistry()
    const ext = createAudioExtractor()
    registry.register(ext)

    const result = await registry.extract(Buffer.alloc(0), {
      contentType: 'audio/ogg',
    })
    expect(result.reason).toBe('empty audio input')
  })
})

describe('createAudioExtractor - subprocess mocking', () => {
  it('transcribes successfully with mocked subprocess', async () => {
    const { scriptPath, cleanup } = await createMockScript('success')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 30_000,
      })

      const input = Buffer.from('fake audio data for testing')
      const result = await ext.extract(input, { contentType: 'audio/wav' })

      expect(result.skipped).toBe(false)
      expect(result.text).toContain('Hello world from the test.')
      expect(result.language).toBe('en')
      expect(result.confidence).toBeGreaterThanOrEqual(0.9)
      expect(result.metadata.detected_language).toBe('en')
      expect(result.metadata.transcription_engine).toBe('faster-whisper')
      expect(result.metadata.segment_count).toBe('2')
      expect(result.metadata.segments).toBeTruthy()
    } finally {
      await cleanup()
    }
  })

  it('auto-detects language when no default is set', async () => {
    const { scriptPath, cleanup } = await createMockScript('language_detect')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 30_000,
      })

      const input = Buffer.from('fake french audio data')
      const result = await ext.extract(input, { contentType: 'audio/mp4' })

      expect(result.language).toBe('fr')
      expect(result.text).toContain('Bonjour le monde.')
    } finally {
      await cleanup()
    }
  })

  it('returns error for corrupt audio (non-zero exit)', async () => {
    const { scriptPath, cleanup } = await createMockScript('corrupt')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 30_000,
      })

      const input = Buffer.from('corrupt audio bytes')
      await expect(
        ext.extract(input, { contentType: 'audio/wav' }),
      ).rejects.toThrow('transcription failed')
    } finally {
      await cleanup()
    }
  })

  it('times out for a hanging subprocess', async () => {
    const { scriptPath, cleanup } = await createMockScript('timeout')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 1_500,
        timeoutPerMinuteMs: 500,
      })

      const input = Buffer.from('audio data that will time out')
      await expect(
        ext.extract(input, { contentType: 'audio/wav' }),
      ).rejects.toThrow('timed out')
    } finally {
      await cleanup()
    }
  })

  it('cleans up temp files after successful extraction', async () => {
    const { scriptPath, cleanup } = await createMockScript('success')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 30_000,
      })

      const input = Buffer.from('fake audio data for cleanup test')
      await ext.extract(input, { contentType: 'audio/wav' })

      // Temp files are cleaned up in a finally block; verify by checking
      // that no memory-audio-* directories remain in tmpdir that were
      // created in the last 5 seconds.
      const { readdirSync, statSync } = await import('node:fs')
      const tempBase = tmpdir()
      const entries = readdirSync(tempBase).filter(e => e.startsWith('memory-audio-'))
      const recent = entries.filter(e => {
        try {
          const info = statSync(join(tempBase, e))
          return Date.now() - info.mtimeMs < 5000
        } catch {
          return false
        }
      })
      expect(recent).toHaveLength(0)
    } finally {
      await cleanup()
    }
  })
})

describe('createAudioExtractor - abort signal / cancellation', () => {
  it('aborts extraction when signal is triggered', async () => {
    const { scriptPath, cleanup } = await createMockScript('timeout')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 60_000, // Long timeout so abort wins.
      })

      const controller = new AbortController()
      // Abort after a short delay.
      setTimeout(() => controller.abort(), 500)

      const input = Buffer.from('audio data to be aborted')
      await expect(
        ext.extract(input, { contentType: 'audio/wav' }, controller.signal),
      ).rejects.toThrow()
    } finally {
      await cleanup()
    }
  })

  it('aborts stream extraction when signal is triggered', async () => {
    const { scriptPath, cleanup } = await createMockScript('timeout')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        minTimeoutMs: 60_000,
      })

      const controller = new AbortController()
      setTimeout(() => controller.abort(), 500)

      const source = Readable.from([Buffer.from('audio stream data to abort')])
      await expect(
        ext.extractStream(source, { contentType: 'audio/wav' }, controller.signal),
      ).rejects.toThrow()
    } finally {
      await cleanup()
    }
  })
})

describe('createAudioExtractor - logging', () => {
  it('logs warning when faster-whisper is unavailable', async () => {
    const warn = vi.fn()
    const ext = createAudioExtractor({
      pythonBinary: '/nonexistent/python3',
      logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
    })

    await ext.available()

    expect(warn).toHaveBeenCalledWith(
      'audio-whisper: faster-whisper not available',
      expect.objectContaining({ python: '/nonexistent/python3' }),
    )
  })

  it('does not log when faster-whisper is available', async () => {
    const { scriptPath, cleanup } = await createMockScript('availability_ok')
    try {
      const warn = vi.fn()
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
        logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
      })

      const avail = await ext.available()

      expect(avail).toBe(true)
      expect(warn).not.toHaveBeenCalled()
    } finally {
      await cleanup()
    }
  })
})

describe('createAudioExtractor - availability race condition', () => {
  it('deduplicates concurrent availability checks', async () => {
    const { scriptPath, cleanup } = await createMockScript('availability_ok')
    try {
      const ext = createAudioExtractor({
        pythonBinary: scriptPath,
      })

      // Fire multiple concurrent calls. If the pending-promise pattern
      // works, they should all resolve to the same value without spawning
      // separate subprocesses.
      const results = await Promise.all([
        ext.available(),
        ext.available(),
        ext.available(),
      ])

      expect(results).toEqual([true, true, true])
    } finally {
      await cleanup()
    }
  })
})

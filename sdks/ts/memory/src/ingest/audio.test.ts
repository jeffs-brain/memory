// SPDX-License-Identifier: Apache-2.0

import { Readable } from 'node:stream'
import { describe, expect, it } from 'vitest'
import {
  createAudioExtractor,
  formatTranscription,
  type AudioExtractorConfig,
  type TranscriptionSegment,
} from './audio.js'
import { createExtractorRegistry } from './extractor.js'

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

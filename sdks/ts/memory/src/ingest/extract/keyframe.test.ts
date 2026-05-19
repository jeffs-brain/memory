// SPDX-License-Identifier: Apache-2.0

import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import type { ExtractOptions, ExtractResult, Extractor, ExtractorCapability } from './types.js'
import { resetBinaryCache } from './subprocess.js'
import {
  extractKeyframes,
  formatKeyframeText,
  formatKeyframeTimestamp,
  mergeKeyframeMetadata,
  resolveKeyframeConfig,
  type KeyframeConfig,
  type KeyframeResult,
} from './keyframe.js'

// ---- Test helpers ----

const writeMockScript = (dir: string, name: string, content: string): string => {
  const path = join(dir, name)
  writeFileSync(path, content, { mode: 0o755 })
  chmodSync(path, 0o755)
  return path
}

const createMockOCRExtractor = (text: string, confidence: number): Extractor => ({
  name: 'mock-ocr',
  contentTypes: ['image/png'],
  async extract(_raw: Buffer, _opts: ExtractOptions): Promise<ExtractResult> {
    return {
      text,
      contentType: 'image/png',
      encoding: 'UTF-8',
      metadata: { ocr_engine: 'mock' },
      pages: 0,
      language: '',
      confidence,
      skipped: false,
    }
  },
  async extractStream() {
    throw new Error('not implemented')
  },
  async available(): Promise<boolean> {
    return true
  },
  capability(): ExtractorCapability {
    return {
      extensions: ['.png'],
      mimeTypes: ['image/png'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

const createUnavailableOCRExtractor = (): Extractor => ({
  name: 'mock-ocr-unavailable',
  contentTypes: ['image/png'],
  async extract(): Promise<ExtractResult> {
    throw new Error('not available')
  },
  async extractStream() {
    throw new Error('not available')
  },
  async available(): Promise<boolean> {
    return false
  },
  capability(): ExtractorCapability {
    return {
      extensions: ['.png'],
      mimeTypes: ['image/png'],
      magicBytes: [],
      requiresBinary: true,
    }
  },
})

// ---- resolveKeyframeConfig ----

describe('resolveKeyframeConfig', () => {
  it('applies defaults when no config provided', () => {
    const cfg = resolveKeyframeConfig({})
    expect(cfg.ffmpegBinary).toBe('ffmpeg')
    expect(cfg.intervalSecs).toBe(30)
    expect(cfg.maxKeyframes).toBe(20)
    expect(cfg.timeout).toBe(60_000)
    expect(cfg.ocrExtractor).toBeUndefined()
  })

  it('accepts custom values', () => {
    const cfg = resolveKeyframeConfig({
      ffmpegBinary: '/custom/ffmpeg',
      intervalSecs: 15,
      maxKeyframes: 10,
      timeout: 30_000,
    })
    expect(cfg.ffmpegBinary).toBe('/custom/ffmpeg')
    expect(cfg.intervalSecs).toBe(15)
    expect(cfg.maxKeyframes).toBe(10)
    expect(cfg.timeout).toBe(30_000)
  })
})

// ---- extractKeyframes ----

describe('extractKeyframes', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('returns empty array when ocrExtractor is undefined', async () => {
    const cfg: KeyframeConfig = {
      ffmpegBinary: 'ffmpeg',
      intervalSecs: 30,
      maxKeyframes: 20,
      timeout: 60_000,
    }
    const results = await extractKeyframes('/fake/video.mp4', '/fake/tmp', cfg)
    expect(results).toHaveLength(0)
  })

  it('returns empty array when ocrExtractor is unavailable', async () => {
    const cfg: KeyframeConfig = {
      ffmpegBinary: 'ffmpeg',
      intervalSecs: 30,
      maxKeyframes: 20,
      ocrExtractor: createUnavailableOCRExtractor(),
      timeout: 60_000,
    }
    const results = await extractKeyframes('/fake/video.mp4', '/fake/tmp', cfg)
    expect(results).toHaveLength(0)
  })
})

// ---- formatKeyframeText ----

describe('formatKeyframeText', () => {
  it('returns empty string for empty results', () => {
    expect(formatKeyframeText([])).toBe('')
  })

  it('formats results with timestamps and text', () => {
    const results: readonly KeyframeResult[] = [
      { timestampSecs: 30, text: 'First frame text', confidence: 0.9 },
      { timestampSecs: 60, text: 'Second frame text', confidence: 0.85 },
      { timestampSecs: 90, text: 'Third frame text', confidence: 0.92 },
    ]

    const text = formatKeyframeText(results)

    expect(text).toContain('--- Keyframe Text ---')
    expect(text).toContain('[00:30]')
    expect(text).toContain('[01:00]')
    expect(text).toContain('[01:30]')
    expect(text).toContain('First frame text')
    expect(text).toContain('Second frame text')
    expect(text).toContain('Third frame text')
  })

  it('handles single result', () => {
    const results: readonly KeyframeResult[] = [
      { timestampSecs: 30, text: 'Only frame', confidence: 0.95 },
    ]

    const text = formatKeyframeText(results)
    expect(text).toContain('[00:30]')
    expect(text).toContain('Only frame')
  })
})

// ---- formatKeyframeTimestamp ----

describe('formatKeyframeTimestamp', () => {
  it('formats zero seconds', () => {
    expect(formatKeyframeTimestamp(0)).toBe('00:00')
  })

  it('formats 30 seconds', () => {
    expect(formatKeyframeTimestamp(30)).toBe('00:30')
  })

  it('formats 60 seconds', () => {
    expect(formatKeyframeTimestamp(60)).toBe('01:00')
  })

  it('formats 90 seconds', () => {
    expect(formatKeyframeTimestamp(90)).toBe('01:30')
  })

  it('formats 3600 seconds', () => {
    expect(formatKeyframeTimestamp(3600)).toBe('60:00')
  })

  it('formats 125 seconds', () => {
    expect(formatKeyframeTimestamp(125)).toBe('02:05')
  })
})

// ---- mergeKeyframeMetadata ----

describe('mergeKeyframeMetadata', () => {
  it('sets count to 0 for empty results', () => {
    const metadata: Record<string, string> = { existing: 'value' }
    mergeKeyframeMetadata(metadata, [])
    expect(metadata['keyframe_count']).toBe('0')
    expect(metadata['keyframe_avg_confidence']).toBeUndefined()
    expect(metadata['existing']).toBe('value')
  })

  it('computes average confidence for results', () => {
    const metadata: Record<string, string> = {}
    const results: readonly KeyframeResult[] = [
      { timestampSecs: 30, text: 'a', confidence: 0.9 },
      { timestampSecs: 60, text: 'b', confidence: 0.8 },
    ]
    mergeKeyframeMetadata(metadata, results)
    expect(metadata['keyframe_count']).toBe('2')
    expect(metadata['keyframe_avg_confidence']).toBe('0.8500')
  })

  it('handles single result', () => {
    const metadata: Record<string, string> = {}
    const results: readonly KeyframeResult[] = [
      { timestampSecs: 30, text: 'a', confidence: 0.95 },
    ]
    mergeKeyframeMetadata(metadata, results)
    expect(metadata['keyframe_count']).toBe('1')
    expect(metadata['keyframe_avg_confidence']).toBe('0.9500')
  })
})

// ---- Mock-based extraction test ----

describe('extractKeyframes with mocked FFmpeg', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('extracts keyframes and runs OCR on each', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'kf-test-'))

    const mockFFmpegScript = `#!/bin/sh
# Creates fake PNG frame files.
output_pattern=""
for arg; do
  output_pattern="$arg"
done
dir=$(dirname "$output_pattern")
mkdir -p "$dir"
printf '\\x89PNG' > "\${dir}/frame-0001.png"
printf '\\x89PNG' > "\${dir}/frame-0002.png"
`
    const ffmpegPath = writeMockScript(tmpDir, 'ffmpeg', mockFFmpegScript)

    const ocrExtractor = createMockOCRExtractor('OCR text from keyframe', 0.88)

    const videoDir = mkdtempSync(join(tmpdir(), 'kf-video-'))
    const videoPath = join(videoDir, 'test.mp4')
    writeFileSync(videoPath, 'fake video data')

    const workDir = mkdtempSync(join(tmpdir(), 'kf-work-'))

    const cfg: KeyframeConfig = {
      ffmpegBinary: ffmpegPath,
      intervalSecs: 30,
      maxKeyframes: 5,
      ocrExtractor,
      timeout: 10_000,
    }

    const results = await extractKeyframes(videoPath, workDir, cfg)
    expect(results).toHaveLength(2)
    expect(results[0]?.text).toBe('OCR text from keyframe')
    expect(results[0]?.timestampSecs).toBe(30)
    expect(results[1]?.timestampSecs).toBe(60)
    expect(results[0]?.confidence).toBe(0.88)
  })
})

// ---- VideoExtractor config integration ----

describe('VideoExtractor keyframe config', () => {
  it('resolves keyframe config with video extractor settings', () => {
    const cfg = resolveKeyframeConfig({
      ffmpegBinary: '/usr/bin/ffmpeg',
      maxKeyframes: 10,
      timeout: 120_000,
    })
    expect(cfg.maxKeyframes).toBe(10)
    expect(cfg.ffmpegBinary).toBe('/usr/bin/ffmpeg')
  })
})

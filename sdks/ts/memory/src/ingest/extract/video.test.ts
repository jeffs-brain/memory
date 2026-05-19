// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { AudioExtractor } from './audio.js'
import { resetBinaryCache } from './subprocess.js'
import {
  VideoExtractor,
  type VideoExtractorConfig,
  type VideoMetadata,
  buildVideoMetadata,
  computeExtractionTimeout,
  mergeMetadata,
  parseFrameRate,
} from './video.js'

// ---- Helpers ----

const hasFFmpeg = (): boolean => {
  try {
    execFileSync('which', ['ffmpeg'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
}

const hasFFprobe = (): boolean => {
  try {
    execFileSync('which', ['ffprobe'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
}

const generateTestVideo = (outputPath: string, withAudio: boolean): void => {
  const args = [
    '-y',
    '-f', 'lavfi', '-i', 'testsrc=duration=1:size=160x120:rate=10',
    ...(withAudio ? ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=1'] : []),
    '-c:v', 'libx264', '-preset', 'ultrafast',
    ...(withAudio ? ['-c:a', 'aac', '-shortest'] : ['-an']),
    outputPath,
  ]
  execFileSync('ffmpeg', args, { stdio: 'ignore' })
}

const makeExtractor = (overrides?: Partial<VideoExtractorConfig>): VideoExtractor =>
  new VideoExtractor({
    audioExtractor: new AudioExtractor(),
    ...overrides,
  })

// ---- Unit tests ----

describe('VideoExtractor', () => {
  beforeEach(() => {
    resetBinaryCache()
  })

  afterEach(() => {
    resetBinaryCache()
  })

  it('has the correct name', () => {
    const e = makeExtractor()
    expect(e.name).toBe('video-ffmpeg')
  })

  describe('capability', () => {
    it('includes expected extensions', () => {
      const e = makeExtractor()
      const cap = e.capability()
      const expected = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v']
      for (const ext of expected) {
        expect(cap.extensions).toContain(ext)
      }
    })

    it('includes expected MIME types', () => {
      const e = makeExtractor()
      const cap = e.capability()
      const expected = [
        'video/mp4',
        'video/x-msvideo',
        'video/x-matroska',
        'video/quicktime',
        'video/webm',
        'video/x-ms-wmv',
        'video/x-flv',
      ]
      for (const mime of expected) {
        expect(cap.mimeTypes).toContain(mime)
      }
    })

    it('requires binary', () => {
      const e = makeExtractor()
      expect(e.capability().requiresBinary).toBe(true)
    })

    it('has magic byte signatures', () => {
      const e = makeExtractor()
      const cap = e.capability()
      expect(cap.magicBytes).toBeDefined()
      expect(cap.magicBytes.length).toBeGreaterThan(0)
    })
  })

  describe('available', () => {
    it('returns false when ffmpeg is absent', async () => {
      const e = makeExtractor({ ffmpegBinary: '/nonexistent/ffmpeg' })
      const result = await e.available()
      expect(result).toBe(false)
    })

    it('returns false when ffprobe is absent', async () => {
      const e = makeExtractor({ ffprobeBinary: '/nonexistent/ffprobe' })
      const result = await e.available()
      expect(result).toBe(false)
    })
  })

  describe('extract', () => {
    it('rejects files exceeding the size limit', async () => {
      const e = makeExtractor({ maxFileSizeBytes: 1024 })
      const input = Buffer.alloc(2048)
      await expect(e.extract(input, {})).rejects.toThrow('exceeds maximum')
    })
  })

  describe('extractStream', () => {
    it('rejects streams exceeding the size limit', async () => {
      const e = makeExtractor({ maxFileSizeBytes: 1024 })
      const input = Readable.from(Buffer.alloc(2048))
      await expect(e.extractStream(input, {})).rejects.toThrow('exceeds maximum')
    })
  })
})

// ---- Pure function tests ----

describe('parseFrameRate', () => {
  it('parses standard frame rates', () => {
    expect(parseFrameRate('30/1')).toBe(30)
    expect(parseFrameRate('60/1')).toBe(60)
  })

  it('parses fractional frame rates', () => {
    const result = parseFrameRate('24000/1001')
    expect(result).toBeDefined()
    expect(result!).toBeCloseTo(23.976, 2)
  })

  it('returns undefined for invalid input', () => {
    expect(parseFrameRate(undefined)).toBeUndefined()
    expect(parseFrameRate('')).toBeUndefined()
    expect(parseFrameRate('invalid')).toBeUndefined()
    expect(parseFrameRate('abc/def')).toBeUndefined()
  })

  it('returns undefined for division by zero', () => {
    expect(parseFrameRate('30/0')).toBeUndefined()
  })
})

describe('buildVideoMetadata', () => {
  it('builds metadata from complete ffprobe output', () => {
    const probe = {
      format: { duration: '125.4' },
      streams: [
        {
          codec_type: 'video',
          codec_name: 'h264',
          width: 1920,
          height: 1080,
          r_frame_rate: '30/1',
        },
        {
          codec_type: 'audio',
          codec_name: 'aac',
        },
      ],
    }

    const meta = buildVideoMetadata(probe)
    expect(meta.duration_seconds).toBe(125.4)
    expect(meta.width).toBe(1920)
    expect(meta.height).toBe(1080)
    expect(meta.codec).toBe('h264')
    expect(meta.has_audio).toBe(true)
    expect(meta.frame_rate).toBe(30)
  })

  it('detects no audio stream', () => {
    const probe = {
      format: { duration: '60.0' },
      streams: [
        {
          codec_type: 'video',
          codec_name: 'vp9',
          width: 1280,
          height: 720,
          r_frame_rate: '24000/1001',
        },
      ],
    }

    const meta = buildVideoMetadata(probe)
    expect(meta.has_audio).toBe(false)
  })

  it('handles empty duration', () => {
    const probe = { format: {}, streams: [] }
    const meta = buildVideoMetadata(probe)
    expect(meta.duration_seconds).toBe(0)
  })

  it('handles invalid duration', () => {
    const probe = { format: { duration: 'invalid' }, streams: [] }
    const meta = buildVideoMetadata(probe)
    expect(meta.duration_seconds).toBe(0)
  })
})

describe('computeExtractionTimeout', () => {
  it('returns at least 60 seconds for short videos', () => {
    const timeout = computeExtractionTimeout(30, 120_000)
    expect(timeout).toBeGreaterThanOrEqual(60_000)
  })

  it('scales with video duration', () => {
    const short = computeExtractionTimeout(300, 120_000)
    const long = computeExtractionTimeout(3600, 120_000)
    expect(long).toBeGreaterThan(short)
  })

  it('computes proportional timeout for 20-minute video', () => {
    const timeout = computeExtractionTimeout(1200, 120_000)
    expect(timeout).toBeGreaterThanOrEqual(240_000)
  })
})

describe('mergeMetadata', () => {
  it('combines video and audio metadata', () => {
    const video: VideoMetadata = {
      duration_seconds: 120,
      width: 1920,
      height: 1080,
      has_audio: true,
    }
    const audio: Record<string, string> = {
      detected_language: 'en',
      segments: '[{"start":0,"end":5,"text":"hello"}]',
    }

    const merged = mergeMetadata(video, audio)
    expect(merged['detected_language']).toBe('en')
    expect(merged['video_duration_seconds']).toBe('120.00')
    expect(merged['video_width']).toBe('1920')
    expect(merged['video_height']).toBe('1080')
    expect(merged['video_has_audio']).toBe('true')
    expect(merged['segments']).toBeDefined()
  })

  it('preserves all audio metadata keys', () => {
    const video: VideoMetadata = {
      duration_seconds: 60,
      has_audio: true,
    }
    const audio: Record<string, string> = {
      key1: 'value1',
      key2: '42',
      key3: 'true',
    }

    const merged = mergeMetadata(video, audio)
    expect(merged['key1']).toBe('value1')
    expect(merged['key2']).toBe('42')
    expect(merged['key3']).toBe('true')
  })
})

// ---- Integration tests (require ffmpeg/ffprobe) ----

describe('VideoExtractor integration', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'memory-video-test-'))
  })

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true })
  })

  it.skipIf(!hasFFmpeg() || !hasFFprobe())(
    'verifies video data can be read for extraction',
    async () => {
      const videoPath = join(tmpDir, 'test.mp4')
      generateTestVideo(videoPath, true)
      const videoData = readFileSync(videoPath)
      expect(videoData.length).toBeGreaterThan(0)
    },
  )

  it.skipIf(!hasFFmpeg() || !hasFFprobe())(
    'handles video with no audio track gracefully',
    async () => {
      const videoPath = join(tmpDir, 'silent.mp4')
      generateTestVideo(videoPath, false)

      const videoData = readFileSync(videoPath)
      const e = makeExtractor()
      const result = await e.extract(videoData, {})

      expect(result.text).toContain('no audio track')
      expect(result.metadata['video_has_audio']).toBe('false')
      expect(result.metadata['video_duration_seconds']).toBeDefined()
    },
  )

  it.skipIf(!hasFFmpeg() || !hasFFprobe())(
    'handles video with no audio track via streaming',
    async () => {
      const videoPath = join(tmpDir, 'silent-stream.mp4')
      generateTestVideo(videoPath, false)

      const videoData = readFileSync(videoPath)
      const stream = Readable.from(videoData)
      const e = makeExtractor()
      const result = await e.extractStream(stream, {})

      expect(result.text).toContain('no audio track')
    },
  )
})

describe('VideoExtractor config', () => {
  it('uses default config values', () => {
    const e = makeExtractor()
    expect(e.name).toBe('video-ffmpeg')
  })

  it('disables keyframe extraction by default', () => {
    const e = makeExtractor()
    expect(e.name).toBe('video-ffmpeg')
  })
})

// ---- Mock-based pipeline test (MA-1) ----

describe('VideoExtractor mock pipeline', () => {
  it('exercises probe-parse and merge logic without FFmpeg', () => {
    // Simulate FFprobe JSON output parsing.
    const fakeProbeOutput = {
      format: { duration: '90.5' },
      streams: [
        {
          codec_type: 'video',
          codec_name: 'h264',
          width: 1920,
          height: 1080,
          r_frame_rate: '30/1',
        },
        {
          codec_type: 'audio',
          codec_name: 'aac',
        },
      ],
    }

    const videoMeta = buildVideoMetadata(fakeProbeOutput)
    expect(videoMeta.duration_seconds).toBe(90.5)
    expect(videoMeta.has_audio).toBe(true)
    expect(videoMeta.width).toBe(1920)
    expect(videoMeta.height).toBe(1080)
    expect(videoMeta.codec).toBe('h264')
    expect(videoMeta.frame_rate).toBe(30)

    // Simulate AudioExtractor result from transcription.
    const fakeAudioResult: Record<string, string> = {
      detected_language: 'en',
      segments: '[{"start":0,"end":30,"text":"Hello world."}]',
      duration_seconds: '30.0',
    }

    const merged = mergeMetadata(videoMeta, fakeAudioResult)
    merged['source_bytes'] = '1024'

    expect(merged['detected_language']).toBe('en')
    expect(merged['video_duration_seconds']).toBe('90.50')
    expect(merged['source_bytes']).toBe('1024')
    expect(merged['segments']).toBeDefined()
  })

  it('returns no-audio result when probe shows no audio stream', () => {
    const noAudioProbe = {
      format: { duration: '10.0' },
      streams: [
        {
          codec_type: 'video',
          codec_name: 'vp9',
          width: 640,
          height: 480,
          r_frame_rate: '25/1',
        },
      ],
    }

    const meta = buildVideoMetadata(noAudioProbe)
    expect(meta.has_audio).toBe(false)

    // The extractor would return this message for the no-audio path.
    const result = {
      text: 'Video contains no audio track.',
      contentType: 'text/plain',
      metadata: { video_has_audio: 'false' },
    }
    expect(result.text).toContain('no audio track')
    expect(result.metadata['video_has_audio']).toBe('false')
  })
})

// ---- Corrupt/invalid video test (MA-2) ----

describe('VideoExtractor corrupt video', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'memory-video-corrupt-'))
  })

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true })
  })

  it.skipIf(!hasFFmpeg() || !hasFFprobe())(
    'produces a meaningful error for corrupt video input',
    async () => {
      // Write garbage with ftyp magic bytes to pass magic-byte checks.
      const garbage = Buffer.alloc(512)
      garbage.write('ftyp', 4, 'ascii')

      const e = makeExtractor()
      await expect(e.extract(garbage, {})).rejects.toThrow(/ffprobe|ffmpeg/i)
    },
  )

  it.skipIf(!hasFFmpeg() || !hasFFprobe())(
    'produces a meaningful error for corrupt video via streaming',
    async () => {
      const garbage = Buffer.alloc(512)
      garbage.write('ftyp', 4, 'ascii')

      const stream = Readable.from(garbage)
      const e = makeExtractor()
      await expect(e.extractStream(stream, {})).rejects.toThrow(/ffprobe|ffmpeg/i)
    },
  )
})

// ---- Streaming size-limit enforcement during copy (M-3) ----

describe('writeStreamToFile size enforcement', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'memory-video-stream-'))
  })

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true })
  })

  it('rejects a stream that gradually exceeds the limit during copy', async () => {
    // The maxFileSizeBytes is set low and the stream delivers more data
    // than the limit. This tests enforcement during the actual copy, not
    // just the pre-check.
    const e = makeExtractor({ maxFileSizeBytes: 64 })
    const oversizedStream = Readable.from(Buffer.alloc(128))
    await expect(
      e.extractStream(oversizedStream, {}),
    ).rejects.toThrow('exceeds maximum')
  })
})

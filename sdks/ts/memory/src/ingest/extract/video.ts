// SPDX-License-Identifier: Apache-2.0

/**
 * Video extraction via FFmpeg + AudioExtractor delegation. Extracts the
 * audio track from a video file, transcribes it via faster-whisper, and
 * returns timestamped text. The video is never fully buffered in a JS
 * Buffer when streaming — the source is piped directly to FFmpeg via
 * stdin or read from a temp file.
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Readable } from 'node:stream'

import type { AudioExtractor } from './audio.js'
import { checkBinaryAvailable, runSubprocess } from './subprocess.js'
import type { ExtractOptions, ExtractionResult, Extractor, ExtractorCapability } from './types.js'

// ---- Configuration ----

export type VideoExtractorConfig = {
  readonly ffmpegBinary?: string
  readonly ffprobeBinary?: string
  readonly maxFileSizeBytes?: number
  readonly extractionTimeout?: number
  readonly keyframeExtraction?: boolean
  readonly maxKeyframes?: number
  readonly audioExtractor: AudioExtractor
}

export type VideoMetadata = {
  readonly duration_seconds: number
  readonly width?: number
  readonly height?: number
  readonly codec?: string
  readonly has_audio: boolean
  readonly frame_rate?: number
}

// ---- Constants ----

const DEFAULT_MAX_VIDEO_SIZE = 2 * 1024 * 1024 * 1024 // 2 GiB
const DEFAULT_EXTRACTION_TIMEOUT = 120_000 // 120s per 10 min of video
const DEFAULT_MAX_KEYFRAMES = 20
const FFPROBE_TIMEOUT = 30_000

// ---- FFprobe types ----

type FFprobeStream = {
  readonly codec_type: string
  readonly codec_name: string
  readonly width?: number
  readonly height?: number
  readonly r_frame_rate?: string
}

type FFprobeFormat = {
  readonly duration?: string
}

type FFprobeOutput = {
  readonly format: FFprobeFormat
  readonly streams: readonly FFprobeStream[]
}

// ---- Resolved config ----

type ResolvedConfig = {
  readonly ffmpegBinary: string
  readonly ffprobeBinary: string
  readonly maxFileSizeBytes: number
  readonly extractionTimeout: number
  readonly keyframeExtraction: boolean
  readonly maxKeyframes: number
  readonly audioExtractor: AudioExtractor
}

const resolveConfig = (cfg: VideoExtractorConfig): ResolvedConfig => ({
  ffmpegBinary: cfg.ffmpegBinary ?? process.env['MEMORY_FFMPEG_PATH'] ?? 'ffmpeg',
  ffprobeBinary: cfg.ffprobeBinary ?? process.env['MEMORY_FFPROBE_PATH'] ?? 'ffprobe',
  maxFileSizeBytes: cfg.maxFileSizeBytes ?? DEFAULT_MAX_VIDEO_SIZE,
  extractionTimeout: cfg.extractionTimeout ?? DEFAULT_EXTRACTION_TIMEOUT,
  keyframeExtraction: cfg.keyframeExtraction ?? false,
  maxKeyframes: cfg.maxKeyframes ?? DEFAULT_MAX_KEYFRAMES,
  audioExtractor: cfg.audioExtractor,
})

// ---- Helpers ----

const parseFrameRate = (raw: string | undefined): number | undefined => {
  if (raw === undefined) return undefined
  const parts = raw.split('/')
  if (parts.length !== 2) return undefined
  const num = Number.parseFloat(parts[0] ?? '')
  const den = Number.parseFloat(parts[1] ?? '')
  if (Number.isNaN(num) || Number.isNaN(den) || den === 0) return undefined
  return num / den
}

const buildVideoMetadata = (probe: FFprobeOutput): VideoMetadata => {
  let width: number | undefined
  let height: number | undefined
  let codec: string | undefined
  let frameRate: number | undefined
  let hasAudio = false

  for (const stream of probe.streams) {
    switch (stream.codec_type) {
      case 'video':
        width = stream.width
        height = stream.height
        codec = stream.codec_name
        frameRate = parseFrameRate(stream.r_frame_rate)
        break
      case 'audio':
        hasAudio = true
        break
    }
  }

  const duration = probe.format.duration !== undefined ? Number.parseFloat(probe.format.duration) : 0

  return {
    duration_seconds: Number.isNaN(duration) ? 0 : duration,
    ...(width !== undefined ? { width } : {}),
    ...(height !== undefined ? { height } : {}),
    ...(codec !== undefined ? { codec } : {}),
    has_audio: hasAudio,
    ...(frameRate !== undefined ? { frame_rate: frameRate } : {}),
  }
}

const mergeMetadata = (
  video: VideoMetadata,
  audio: Readonly<Record<string, unknown>>,
): Record<string, unknown> => {
  const merged: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(audio)) {
    merged[k] = v
  }
  merged['video_info'] = video
  return merged
}

const computeExtractionTimeout = (durationSeconds: number, baseTimeout: number): number => {
  const tenMinBlocks = Math.max(1, durationSeconds / 600)
  const computed = tenMinBlocks * baseTimeout
  return Math.max(computed, 60_000)
}

const overrideTimeout = (): number | undefined => {
  const raw = process.env['MEMORY_EXTRACTOR_TIMEOUT_MS']
  if (raw === undefined || raw === '') return undefined
  const ms = Number.parseInt(raw, 10)
  return Number.isNaN(ms) ? undefined : ms
}

// ---- Video extractor ----

export class VideoExtractor implements Extractor {
  readonly name = 'video-ffmpeg' as const

  private readonly cfg: ResolvedConfig

  constructor(config: VideoExtractorConfig) {
    this.cfg = resolveConfig(config)
  }

  get capability(): ExtractorCapability {
    return {
      extensions: new Set(['.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v']),
      mimeTypes: new Set([
        'video/mp4',
        'video/x-msvideo',
        'video/x-matroska',
        'video/quicktime',
        'video/webm',
        'video/x-ms-wmv',
        'video/x-flv',
      ]),
      magicBytes: [
        { offset: 4, bytes: new Uint8Array([0x66, 0x74, 0x79, 0x70]) }, // MP4/MOV ftyp
        { offset: 0, bytes: new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]) }, // MKV/WebM (EBML)
        { offset: 0, bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46]) }, // RIFF (AVI)
      ],
      requiresBinary: true,
    }
  }

  async available(): Promise<boolean> {
    const [hasFFmpeg, hasFFprobe] = await Promise.all([
      checkBinaryAvailable(this.cfg.ffmpegBinary),
      checkBinaryAvailable(this.cfg.ffprobeBinary),
    ])
    if (!hasFFmpeg || !hasFFprobe) return false
    return this.cfg.audioExtractor.available()
  }

  /**
   * Extract content from a video provided as a Buffer. For large
   * files, prefer extractStream to avoid holding the entire video in
   * memory.
   */
  async extract(input: Buffer, opts: ExtractOptions): Promise<ExtractionResult> {
    if (input.length > this.cfg.maxFileSizeBytes) {
      throw new Error(
        `extract: video file size ${input.length} exceeds maximum ${this.cfg.maxFileSizeBytes} bytes`,
      )
    }
    return this.extractFromBuffer(input, opts)
  }

  /**
   * Extract content from a video stream without buffering the full
   * file in a JS Buffer. The stream is written to a temp file (for
   * FFprobe compatibility) with a size limit enforced during the copy.
   */
  async extractStream(
    input: Readable,
    sizeBound: number | undefined,
    opts: ExtractOptions,
  ): Promise<ExtractionResult> {
    if (sizeBound !== undefined && sizeBound > this.cfg.maxFileSizeBytes) {
      throw new Error(
        `extract: video file size ${sizeBound} exceeds maximum ${this.cfg.maxFileSizeBytes} bytes`,
      )
    }
    return this.extractFromStream(input, opts)
  }

  // ---- Private implementation ----

  private async extractFromBuffer(input: Buffer, opts: ExtractOptions): Promise<ExtractionResult> {
    const tmpDir = await mkdtemp(join(tmpdir(), 'memory-video-'))
    try {
      const tmpVideoPath = join(tmpDir, 'input.video')
      await writeFile(tmpVideoPath, input)
      return this.processVideoFile(tmpVideoPath, input.length, opts)
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  private async extractFromStream(input: Readable, opts: ExtractOptions): Promise<ExtractionResult> {
    const tmpDir = await mkdtemp(join(tmpdir(), 'memory-video-'))
    try {
      const tmpVideoPath = join(tmpDir, 'input.video')
      const written = await writeStreamToFile(tmpVideoPath, input, this.cfg.maxFileSizeBytes)
      return this.processVideoFile(tmpVideoPath, written, opts)
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  private async processVideoFile(
    videoPath: string,
    sourceBytes: number,
    opts: ExtractOptions,
  ): Promise<ExtractionResult> {
    // Probe metadata.
    const videoMeta = await this.probeMetadata(videoPath, opts.signal)

    if (!videoMeta.has_audio) {
      return {
        content: 'Video contains no audio track.',
        mime: 'text/plain',
        metadata: { video_info: videoMeta },
      }
    }

    // Extract audio via FFmpeg.
    const wavData = await this.extractAudioTrack(videoPath, videoMeta.duration_seconds, opts.signal)

    // Delegate transcription to AudioExtractor.
    const audioResult = await this.cfg.audioExtractor.extractFromBuffer(Buffer.from(wavData), opts)

    const metadata = mergeMetadata(videoMeta, audioResult.metadata)
    metadata['source_bytes'] = sourceBytes

    return {
      content: audioResult.content,
      mime: 'text/plain',
      metadata,
      ...(audioResult.language !== undefined ? { language: audioResult.language } : {}),
      ...(audioResult.confidence !== undefined ? { confidence: audioResult.confidence } : {}),
    }
  }

  private async probeMetadata(videoPath: string, signal?: AbortSignal): Promise<VideoMetadata> {
    const timeout = overrideTimeout() ?? FFPROBE_TIMEOUT

    const result = await runSubprocess(
      this.cfg.ffprobeBinary,
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', videoPath],
      undefined,
      { timeout, ...(signal !== undefined ? { signal } : {}) },
    )

    if (result.exitCode !== 0) {
      throw new Error(`extract: ffprobe failed (exit ${result.exitCode}): ${result.stderr}`)
    }

    const probe = JSON.parse(result.stdout.toString('utf8')) as FFprobeOutput
    return buildVideoMetadata(probe)
  }

  private async extractAudioTrack(
    videoPath: string,
    durationSeconds: number,
    signal?: AbortSignal,
  ): Promise<Buffer> {
    const timeout = computeExtractionTimeout(durationSeconds, this.cfg.extractionTimeout)

    return new Promise<Buffer>((resolve, reject) => {
      const ffmpeg: ChildProcess = spawn(
        this.cfg.ffmpegBinary,
        ['-i', videoPath, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-f', 'wav', 'pipe:1'],
        { stdio: ['ignore', 'pipe', 'pipe'] },
      )

      const chunks: Buffer[] = []
      let stderrText = ''

      const timer = setTimeout(() => {
        ffmpeg.kill('SIGKILL')
        reject(new Error(`extract: ffmpeg audio extraction timed out after ${timeout}ms`))
      }, timeout)

      const onSignalAbort = (): void => {
        ffmpeg.kill('SIGKILL')
        reject(new Error('extract: ffmpeg audio extraction aborted'))
      }
      signal?.addEventListener('abort', onSignalAbort, { once: true })

      ffmpeg.stdout?.on('data', (chunk: Buffer) => {
        chunks.push(chunk)
      })

      ffmpeg.stderr?.on('data', (chunk: Buffer) => {
        stderrText += chunk.toString('utf8')
      })

      ffmpeg.on('error', (err: Error) => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onSignalAbort)
        reject(new Error(`extract: ffmpeg failed: ${err.message}`))
      })

      ffmpeg.on('close', (code: number | null) => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onSignalAbort)
        if (code !== 0) {
          reject(new Error(`extract: ffmpeg exited ${code}: ${stderrText}`))
          return
        }
        resolve(Buffer.concat(chunks))
      })
    })
  }
}

// ---- Stream-to-file utility ----

/**
 * Write a Readable stream to a file, enforcing a maximum size limit.
 * Returns the number of bytes written.
 */
const writeStreamToFile = (filePath: string, input: Readable, maxSize: number): Promise<number> =>
  new Promise<number>((resolve, reject) => {
    const out = createWriteStream(filePath)
    let written = 0

    input.on('data', (chunk: Buffer) => {
      written += chunk.length
      if (written > maxSize) {
        input.destroy()
        out.destroy()
        reject(new Error(`extract: video stream exceeds maximum ${maxSize} bytes`))
        return
      }
      out.write(chunk)
    })

    input.on('end', () => {
      out.end(() => resolve(written))
    })

    input.on('error', (err: Error) => {
      out.destroy()
      reject(new Error(`extract: reading video stream: ${err.message}`))
    })

    out.on('error', (err: Error) => {
      reject(new Error(`extract: writing video file: ${err.message}`))
    })
  })

// ---- Exported for testing ----

export { buildVideoMetadata, computeExtractionTimeout, mergeMetadata, parseFrameRate }

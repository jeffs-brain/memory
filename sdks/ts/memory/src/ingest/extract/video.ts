// SPDX-License-Identifier: Apache-2.0

/**
 * Video extraction via FFmpeg + AudioExtractor delegation. Extracts the
 * audio track from a video file, transcribes it via faster-whisper, and
 * returns timestamped text. Implements the canonical Extractor interface
 * from P1-5.
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdtemp, rm, writeFile as fsWriteFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Transform, type Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

import type { AudioExtractor } from './audio.js'
import {
  extractKeyframes,
  formatKeyframeText,
  mergeKeyframeMetadata,
  resolveKeyframeConfig,
} from './keyframe.js'
import { checkBinaryAvailable } from './subprocess.js'
import type { ExtractOptions, ExtractResult, Extractor, ExtractorCapability } from './types.js'

// ---- Configuration ----

export type VideoExtractorConfig = {
  readonly ffmpegBinary?: string
  readonly ffprobeBinary?: string
  readonly maxFileSizeBytes?: number
  readonly extractionTimeout?: number
  readonly keyframeExtraction?: boolean
  readonly maxKeyframes?: number
  readonly audioExtractor: AudioExtractor
  readonly ocrExtractor?: Extractor
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
  readonly ocrExtractor?: Extractor
}

const resolveConfig = (cfg: VideoExtractorConfig): ResolvedConfig => ({
  ffmpegBinary: cfg.ffmpegBinary ?? process.env['MEMORY_FFMPEG_PATH'] ?? 'ffmpeg',
  ffprobeBinary: cfg.ffprobeBinary ?? process.env['MEMORY_FFPROBE_PATH'] ?? 'ffprobe',
  maxFileSizeBytes: cfg.maxFileSizeBytes ?? DEFAULT_MAX_VIDEO_SIZE,
  extractionTimeout: cfg.extractionTimeout ?? DEFAULT_EXTRACTION_TIMEOUT,
  keyframeExtraction: cfg.keyframeExtraction ?? false,
  maxKeyframes: cfg.maxKeyframes ?? DEFAULT_MAX_KEYFRAMES,
  audioExtractor: cfg.audioExtractor,
  ocrExtractor: cfg.ocrExtractor,
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
  audio: Readonly<Record<string, string>>,
): Record<string, string> => {
  const merged: Record<string, string> = {}
  for (const [k, v] of Object.entries(audio)) {
    merged[k] = v
  }
  merged['video_duration_seconds'] = String(video.duration_seconds.toFixed(2))
  merged['video_has_audio'] = String(video.has_audio)
  if (video.width !== undefined) merged['video_width'] = String(video.width)
  if (video.height !== undefined) merged['video_height'] = String(video.height)
  if (video.codec !== undefined) merged['video_codec'] = video.codec
  if (video.frame_rate !== undefined) merged['video_frame_rate'] = video.frame_rate.toFixed(3)
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

/** Collects all chunks from a Readable into a Buffer, respecting an optional byte limit. */
const bufferStream = async (source: Readable, maxBytes?: number): Promise<Buffer> => {
  const chunks: Buffer[] = []
  let totalBytes = 0

  for await (const chunk of source) {
    const buf = Buffer.isBuffer(chunk)
      ? chunk
      : typeof chunk === 'string'
        ? Buffer.from(chunk)
        : chunk instanceof Uint8Array
          ? Buffer.from(chunk)
          : Buffer.from(String(chunk))
    if (maxBytes !== undefined && maxBytes > 0) {
      const remaining = maxBytes - totalBytes
      if (remaining <= 0) break
      chunks.push(buf.subarray(0, remaining))
      totalBytes += Math.min(buf.length, remaining)
      if (totalBytes >= maxBytes) break
    } else {
      chunks.push(buf)
      totalBytes += buf.length
    }
  }

  return Buffer.concat(chunks)
}

/** Video content types handled by this extractor. */
const videoContentTypes: readonly string[] = [
  'video/mp4',
  'video/x-msvideo',
  'video/x-matroska',
  'video/quicktime',
  'video/webm',
  'video/x-ms-wmv',
  'video/x-flv',
]

// ---- Video extractor ----

export class VideoExtractor implements Extractor {
  readonly name = 'video-ffmpeg' as const

  readonly contentTypes = videoContentTypes

  private readonly cfg: ResolvedConfig

  constructor(config: VideoExtractorConfig) {
    this.cfg = resolveConfig(config)
  }

  capability(): ExtractorCapability {
    return {
      extensions: ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.flv', '.m4v'],
      mimeTypes: [...videoContentTypes],
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
   * Extract content from a video provided as a Buffer.
   */
  async extract(input: Buffer, opts: ExtractOptions): Promise<ExtractResult> {
    if (input.length > this.cfg.maxFileSizeBytes) {
      throw new Error(
        `ingest: video file size ${input.length} exceeds maximum ${this.cfg.maxFileSizeBytes} bytes`,
      )
    }
    return this.extractFromBuffer(input, opts)
  }

  /**
   * Extract content from a video stream by buffering into extract().
   */
  async extractStream(
    source: Readable,
    opts: ExtractOptions,
  ): Promise<ExtractResult> {
    const raw = await bufferStream(source, opts.maxBytes)
    return this.extract(raw, opts)
  }

  // ---- Private implementation ----

  private async extractFromBuffer(input: Buffer, opts: ExtractOptions): Promise<ExtractResult> {
    const tmpDir = await mkdtemp(join(tmpdir(), 'memory-video-'))
    try {
      const tmpVideoPath = join(tmpDir, 'input.video')
      await fsWriteFile(tmpVideoPath, input, { mode: 0o600 })
      return this.processVideoFile(tmpVideoPath, tmpDir, input.length, opts)
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  private async processVideoFile(
    videoPath: string,
    tmpDir: string,
    sourceBytes: number,
    opts: ExtractOptions,
  ): Promise<ExtractResult> {
    // Probe metadata.
    const videoMeta = await this.probeMetadata(videoPath, opts.signal)

    if (!videoMeta.has_audio) {
      return {
        text: 'Video contains no audio track.',
        contentType: 'text/plain',
        encoding: 'UTF-8',
        metadata: {
          video_duration_seconds: String(videoMeta.duration_seconds.toFixed(2)),
          video_has_audio: 'false',
          ...(videoMeta.width !== undefined ? { video_width: String(videoMeta.width) } : {}),
          ...(videoMeta.height !== undefined ? { video_height: String(videoMeta.height) } : {}),
          ...(videoMeta.codec !== undefined ? { video_codec: videoMeta.codec } : {}),
        },
        pages: 0,
        language: '',
        confidence: 0,
        skipped: false,
      }
    }

    // Extract audio via FFmpeg to a temp WAV file instead of buffering
    // the entire WAV payload in memory (~230 MB for a 2-hour video).
    const wavPath = await this.extractAudioToFile(videoPath, tmpDir, videoMeta.duration_seconds, opts.signal)

    // Read the WAV file for the AudioExtractor.
    const { readFile } = await import('node:fs/promises')
    const wavData = await readFile(wavPath)

    // Delegate transcription to AudioExtractor.
    const audioOpts: ExtractOptions = {
      contentType: 'audio/wav',
      fileName: 'extracted.wav',
      ...(opts.language !== undefined ? { language: opts.language } : {}),
    }
    const audioResult = await this.cfg.audioExtractor.extract(wavData, audioOpts)

    const metadata = mergeMetadata(videoMeta, audioResult.metadata)
    metadata['source_bytes'] = String(sourceBytes)

    let resultText = audioResult.text

    if (this.cfg.keyframeExtraction) {
      const kfCfg = resolveKeyframeConfig({
        ffmpegBinary: this.cfg.ffmpegBinary,
        maxKeyframes: this.cfg.maxKeyframes,
        ocrExtractor: this.cfg.ocrExtractor,
        timeout: this.cfg.extractionTimeout,
      })

      try {
        const keyframes = await extractKeyframes(videoPath, tmpDir, kfCfg, opts.signal)
        const kfText = formatKeyframeText(keyframes)
        if (kfText.length > 0) {
          resultText += kfText
        }
        mergeKeyframeMetadata(metadata, keyframes)
      } catch (kfErr: unknown) {
        metadata['keyframe_error'] =
          kfErr instanceof Error ? kfErr.message : String(kfErr)
      }
    }

    return {
      text: resultText,
      contentType: 'text/plain',
      encoding: 'UTF-8',
      metadata,
      pages: 0,
      language: audioResult.language,
      confidence: audioResult.confidence,
      skipped: false,
    }
  }

  private async probeMetadata(videoPath: string, signal?: AbortSignal): Promise<VideoMetadata> {
    const timeout = overrideTimeout() ?? FFPROBE_TIMEOUT

    const { runSubprocess } = await import('./subprocess.js')
    const result = await runSubprocess(
      this.cfg.ffprobeBinary,
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', videoPath],
      undefined,
      { timeout, ...(signal !== undefined ? { signal } : {}) },
    )

    if (result.exitCode !== 0) {
      throw new Error(`ingest: ffprobe failed (exit ${result.exitCode}): ${result.stderr}`)
    }

    const probe = JSON.parse(result.stdout.toString('utf8')) as FFprobeOutput
    return buildVideoMetadata(probe)
  }

  /**
   * Extracts the audio track to a temp WAV file instead of buffering
   * the entire WAV payload in memory. For a 2-hour video, the WAV data
   * can be ~230 MB; writing to disk avoids that allocation.
   */
  private async extractAudioToFile(
    videoPath: string,
    tmpDir: string,
    durationSeconds: number,
    signal?: AbortSignal,
  ): Promise<string> {
    const timeout = computeExtractionTimeout(durationSeconds, this.cfg.extractionTimeout)
    const wavPath = join(tmpDir, 'extracted.wav')

    return new Promise<string>((resolve, reject) => {
      const ffmpeg: ChildProcess = spawn(
        this.cfg.ffmpegBinary,
        ['-i', videoPath, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', wavPath],
        { stdio: ['ignore', 'ignore', 'pipe'] },
      )

      let stderrText = ''

      const timer = setTimeout(() => {
        ffmpeg.kill('SIGKILL')
        reject(new Error(`ingest: ffmpeg audio extraction timed out after ${timeout}ms`))
      }, timeout)

      const onSignalAbort = (): void => {
        ffmpeg.kill('SIGKILL')
        reject(new Error('ingest: ffmpeg audio extraction aborted'))
      }
      signal?.addEventListener('abort', onSignalAbort, { once: true })

      ffmpeg.stderr?.on('data', (chunk: Buffer) => {
        stderrText += chunk.toString('utf8')
      })

      ffmpeg.on('error', (err: Error) => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onSignalAbort)
        reject(new Error(`ingest: ffmpeg failed: ${err.message}`))
      })

      ffmpeg.on('close', (code: number | null) => {
        clearTimeout(timer)
        signal?.removeEventListener('abort', onSignalAbort)
        if (code !== 0) {
          reject(new Error(`ingest: ffmpeg exited ${code}: ${stderrText}`))
          return
        }
        resolve(wavPath)
      })
    })
  }
}

// ---- Stream-to-file utility ----

/**
 * Write a Readable stream to a file, enforcing a maximum size limit.
 * Returns the number of bytes written.
 */
const writeStreamToFile = async (filePath: string, input: Readable, maxSize: number): Promise<number> => {
  const out = createWriteStream(filePath, { mode: 0o600 })
  let written = 0

  const sizeEnforcer = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      written += chunk.length
      if (written > maxSize) {
        callback(new Error(`ingest: video stream exceeds maximum ${maxSize} bytes`))
        return
      }
      callback(null, chunk)
    },
  })

  await pipeline(input, sizeEnforcer, out)
  return written
}

// ---- Exported for testing ----

export { buildVideoMetadata, computeExtractionTimeout, mergeMetadata, parseFrameRate }

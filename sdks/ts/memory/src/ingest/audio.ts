// SPDX-License-Identifier: Apache-2.0

/**
 * AudioExtractor transcribes audio files using faster-whisper via a
 * Python subprocess. Supports MP3, WAV, FLAC, M4A, OGG, AAC, WMA,
 * and Opus formats. Produces timestamped transcription with word-level
 * detail when available.
 *
 * Mirrors the Go implementation in go/ingest/audio.go.
 */

import { execFile } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, extname } from 'node:path'
import type { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { Logger } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'
import { bufferStream } from './extractor.js'
import type { Extractor, ExtractOptions, ExtractResult, ExtractorCapability, MagicSignature } from './extractor.js'

/** Default configuration values for the AudioExtractor. */
const DEFAULT_PYTHON_BINARY = 'python3'
const DEFAULT_WHISPER_MODEL = 'base'
const DEFAULT_MAX_AUDIO_SIZE = 500 * 1024 * 1024 // 500 MB
const DEFAULT_TIMEOUT_PER_MIN_MS = 30_000 // 30s per minute of audio
const DEFAULT_MIN_TIMEOUT_MS = 120_000 // 120 seconds minimum
const PARAGRAPH_BREAK_SECONDS = 30.0
const AVAILABILITY_CACHE_TTL_MS = 5 * 60 * 1000 // 5 minutes

/**
 * Embedded Python script invoked via subprocess to transcribe audio
 * using faster-whisper. Outputs JSON to stdout.
 */
const WHISPER_SCRIPT = `
import sys, json
from faster_whisper import WhisperModel

audio_path = sys.argv[1]
model_size = sys.argv[2]
language = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "" else None

model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path, language=language, word_timestamps=True)

result = {"language": info.language, "language_probability": info.language_probability, "duration": info.duration, "segments": []}
for s in segments:
    seg = {"start": s.start, "end": s.end, "text": s.text}
    if s.words:
        seg["words"] = [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability} for w in s.words]
    result["segments"].append(seg)

json.dump(result, sys.stdout)
`

/** Script to check whether faster_whisper is importable. */
const AVAILABILITY_CHECK_SCRIPT = 'import faster_whisper; print("ok")'

/** Configuration for the AudioExtractor. */
export type AudioExtractorConfig = {
  /** Path to the Python 3 interpreter. Default: "python3". Override via MEMORY_WHISPER_PATH. */
  readonly pythonBinary?: string
  /** The faster-whisper model to use. Default: "base". */
  readonly modelSize?: string
  /** Language hint for transcription. Undefined means auto-detect. */
  readonly defaultLanguage?: string
  /** Maximum audio file size in bytes. Default: 500 MB. */
  readonly maxFileSizeBytes?: number
  /** Timeout budget per minute of estimated audio (ms). Default: 30000. */
  readonly timeoutPerMinuteMs?: number
  /** Minimum timeout regardless of file size (ms). Default: 120000. */
  readonly minTimeoutMs?: number
  /** Structured logger. Defaults to a no-op. */
  readonly logger?: Logger
}

/** A single timed segment of transcription. */
export type TranscriptionSegment = {
  readonly start: number
  readonly end: number
  readonly text: string
  readonly words?: readonly TranscriptionWord[]
}

/** A word-level timestamp entry. */
export type TranscriptionWord = {
  readonly start: number
  readonly end: number
  readonly word: string
  readonly probability: number
}

/** JSON structure returned by the Python subprocess. */
type WhisperOutput = {
  readonly language: string
  readonly language_probability: number
  readonly duration: number
  readonly segments: readonly TranscriptionSegment[]
}

/** MIME types handled by AudioExtractor. */
const AUDIO_CONTENT_TYPES: readonly string[] = [
  'audio/mpeg',
  'audio/wav',
  'audio/x-wav',
  'audio/flac',
  'audio/mp4',
  'audio/ogg',
  'audio/aac',
  'audio/x-ms-wma',
  'audio/opus',
]

/** File extensions handled by AudioExtractor. */
const AUDIO_EXTENSIONS: readonly string[] = [
  '.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.opus',
]

/** Magic byte patterns for audio format detection. */
const AUDIO_MAGIC_SIGNATURES: readonly MagicSignature[] = [
  { offset: 0, bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46]) },  // WAV (RIFF)
  { offset: 8, bytes: new Uint8Array([0x57, 0x41, 0x56, 0x45]) },  // WAV (WAVE)
  { offset: 0, bytes: new Uint8Array([0x66, 0x4c, 0x61, 0x43]) },  // FLAC
  { offset: 0, bytes: new Uint8Array([0x4f, 0x67, 0x67, 0x53]) },  // OGG
  { offset: 0, bytes: new Uint8Array([0x49, 0x44, 0x33]) },         // MP3 ID3
  { offset: 0, bytes: new Uint8Array([0xff, 0xfb]) },               // MP3 sync
  { offset: 0, bytes: new Uint8Array([0xff, 0xf3]) },               // MP3 sync
  { offset: 0, bytes: new Uint8Array([0xff, 0xf2]) },               // MP3 sync
]

/** Maps content type to file extension for temp file creation. */
const CONTENT_TYPE_EXT_MAP: Readonly<Record<string, string>> = {
  'audio/mpeg': '.mp3',
  'audio/wav': '.wav',
  'audio/x-wav': '.wav',
  'audio/flac': '.flac',
  'audio/mp4': '.m4a',
  'audio/ogg': '.ogg',
  'audio/aac': '.aac',
  'audio/x-ms-wma': '.wma',
  'audio/opus': '.opus',
}

/**
 * Derives a file extension from content type or filename.
 * Falls back to .wav when neither provides a hint.
 */
const extensionFromContentType = (contentType: string, fileName?: string): string => {
  if (fileName) {
    const ext = extname(fileName)
    if (ext) return ext
  }

  const normalised = contentType.split(';')[0]?.trim().toLowerCase() ?? ''
  return CONTENT_TYPE_EXT_MAP[normalised] ?? '.wav'
}

/**
 * Formats transcription segments into readable text with timestamp
 * markers. Inserts paragraph breaks every 30 seconds of audio.
 */
export const formatTranscription = (segments: readonly TranscriptionSegment[]): string => {
  if (segments.length === 0) return ''

  const parts: string[] = []
  let currentParagraphStart = 0
  let paragraphStarted = false

  for (const seg of segments) {
    if (!paragraphStarted || (seg.start - currentParagraphStart) >= PARAGRAPH_BREAK_SECONDS) {
      if (paragraphStarted) {
        parts.push('\n\n')
      }
      parts.push(formatTimestamp(seg.start))
      parts.push(' - ')
      parts.push(formatTimestamp(seg.end))
      parts.push(']\n')
      currentParagraphStart = seg.start
      paragraphStarted = true
    }
    parts.push(seg.text.trim())
    parts.push(' ')
  }

  return parts.join('').trim()
}

/** Converts seconds to [MM:SS format. */
const formatTimestamp = (seconds: number): string => {
  const totalSeconds = Math.floor(seconds)
  const minutes = Math.floor(totalSeconds / 60)
  const secs = totalSeconds % 60
  return `[${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

/**
 * Runs a subprocess using execFile (safe against shell injection) and
 * returns the result.
 */
const runSubprocess = (
  binary: string,
  args: readonly string[],
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<{ stdout: string; stderr: string }> =>
  new Promise((resolve, reject) => {
    const child = execFile(
      binary,
      [...args],
      {
        timeout: timeoutMs,
        maxBuffer: 50 * 1024 * 1024, // 50 MB stdout buffer
        signal,
      },
      (error, stdout, stderr) => {
        if (error) {
          const stderrStr = typeof stderr === 'string' ? stderr.trim() : ''
          const isTimeout = error.killed || (error as NodeJS.ErrnoException).code === 'ABORT_ERR'
          const msg = isTimeout
            ? `ingest: audio transcription timed out after ${timeoutMs}ms: ${stderrStr}`
            : `ingest: audio transcription failed (exit ${child.exitCode}): ${stderrStr}`
          reject(new Error(msg))
          return
        }
        resolve({
          stdout: typeof stdout === 'string' ? stdout : '',
          stderr: typeof stderr === 'string' ? stderr : '',
        })
      },
    )
  })

/**
 * Computes an appropriate timeout based on file size. Estimates audio
 * duration from file size using conservative ratios, then applies the
 * per-minute timeout budget with a minimum floor.
 */
const computeTimeout = (fileSize: number, timeoutPerMinMs: number, minTimeoutMs: number): number => {
  // Conservative: 1 MB/min for compressed audio.
  const estimatedMinutes = fileSize / (1024 * 1024)
  const timeout = estimatedMinutes * timeoutPerMinMs
  return Math.max(timeout, minTimeoutMs)
}

/**
 * Creates an AudioExtractor that transcribes audio files using
 * faster-whisper via a Python subprocess.
 */
export const createAudioExtractor = (config?: AudioExtractorConfig): Extractor => {
  const pythonBinary = config?.pythonBinary
    ?? process.env['MEMORY_WHISPER_PATH']
    ?? DEFAULT_PYTHON_BINARY

  const modelSize = config?.modelSize ?? DEFAULT_WHISPER_MODEL
  const defaultLanguage = config?.defaultLanguage
  const maxFileSize = config?.maxFileSizeBytes ?? DEFAULT_MAX_AUDIO_SIZE
  const timeoutPerMinMs = config?.timeoutPerMinuteMs ?? DEFAULT_TIMEOUT_PER_MIN_MS

  const logger = config?.logger ?? noopLogger

  const envTimeoutMs = process.env['MEMORY_EXTRACTOR_TIMEOUT_MS']
  const parsedEnvTimeout = envTimeoutMs ? Number.parseInt(envTimeoutMs, 10) : Number.NaN
  const minTimeoutMs = config?.minTimeoutMs
    ?? (Number.isFinite(parsedEnvTimeout) && parsedEnvTimeout > 0 ? parsedEnvTimeout : DEFAULT_MIN_TIMEOUT_MS)

  // Availability cache. Uses a pending-promise pattern to dedup concurrent
  // calls that would otherwise spawn multiple Python subprocesses before the
  // first one resolves.
  let cachedAvailable: boolean | undefined
  let cachedAt = 0
  let pendingCheck: Promise<boolean> | undefined

  const checkAvailability = async (): Promise<boolean> => {
    const now = Date.now()
    if (cachedAvailable !== undefined && (now - cachedAt) < AVAILABILITY_CACHE_TTL_MS) {
      return cachedAvailable
    }

    // Return the in-flight promise if another caller is already checking.
    if (pendingCheck !== undefined) {
      return pendingCheck
    }

    pendingCheck = (async () => {
      try {
        await runSubprocess(pythonBinary, ['-c', AVAILABILITY_CHECK_SCRIPT], 10_000)
        cachedAvailable = true
      } catch {
        cachedAvailable = false
        logger.warn('audio-whisper: faster-whisper not available', {
          python: pythonBinary,
        })
      }
      cachedAt = Date.now()
      pendingCheck = undefined
      return cachedAvailable
    })()

    return pendingCheck
  }

  /**
   * Transcribes audio from a temp file path and returns the formatted
   * result.
   */
  const transcribe = async (
    audioPath: string,
    fileSize: number,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult> => {
    const timeout = computeTimeout(fileSize, timeoutPerMinMs, minTimeoutMs)

    const args: string[] = ['-c', WHISPER_SCRIPT, audioPath, modelSize]
    const language = defaultLanguage ?? ''
    if (language) {
      args.push(language)
    }

    const { stdout } = await runSubprocess(pythonBinary, args, timeout, signal)

    const output: WhisperOutput = JSON.parse(stdout)
    const text = formatTranscription(output.segments)

    const metadata: Record<string, string> = {
      detected_language: output.language,
      language_probability: output.language_probability.toFixed(4),
      duration_seconds: output.duration.toFixed(2),
      segment_count: String(output.segments.length),
      segments: JSON.stringify(output.segments),
      model_size: modelSize,
      transcription_engine: 'faster-whisper',
    }

    return {
      text,
      contentType: opts.contentType ?? 'audio/wav',
      encoding: 'UTF-8',
      metadata,
      pages: 0,
      language: output.language,
      confidence: output.language_probability,
      skipped: false,
    }
  }

  const extractor: Extractor = {
    name: 'audio-whisper',
    contentTypes: AUDIO_CONTENT_TYPES,

    async extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      if (raw.length > maxFileSize) {
        throw new Error(
          `ingest: audio file size ${raw.length} bytes exceeds maximum ${maxFileSize} bytes`,
        )
      }

      if (raw.length === 0) {
        return {
          text: '',
          contentType: opts.contentType ?? 'audio/wav',
          encoding: 'UTF-8',
          metadata: {},
          pages: 0,
          language: '',
          confidence: 0,
          skipped: true,
          reason: 'empty audio input',
        }
      }

      // Write audio to a temp file for faster-whisper.
      const ext = extensionFromContentType(opts.contentType ?? 'audio/wav', opts.fileName)
      const tmpDir = await mkdtemp(join(tmpdir(), 'memory-audio-'))
      const tmpPath = join(tmpDir, `audio${ext}`)

      try {
        await writeFile(tmpPath, raw)
        return await transcribe(tmpPath, raw.length, opts, signal)
      } finally {
        await rm(tmpDir, { recursive: true, force: true }).catch(() => {
          // Best-effort cleanup; do not mask the primary error.
        })
      }
    },

    async extractStream(
      source: Readable,
      opts: ExtractOptions,
      signal?: AbortSignal,
    ): Promise<ExtractResult> {
      // Stream directly to a temp file instead of buffering the entire
      // audio payload in memory. Critical for large files (100 MB+).
      const ext = extensionFromContentType(opts.contentType ?? 'audio/wav', opts.fileName)
      const tmpDir = await mkdtemp(join(tmpdir(), 'memory-audio-stream-'))
      const tmpPath = join(tmpDir, `audio${ext}`)

      try {
        let written = 0
        const { Transform } = await import('node:stream')
        const sizeEnforcer = new Transform({
          transform(chunk: Buffer, _encoding, callback) {
            written += chunk.length
            if (written > maxFileSize) {
              callback(new Error(
                `ingest: audio stream size ${written} bytes exceeds maximum ${maxFileSize} bytes`,
              ))
              return
            }
            callback(null, chunk)
          },
        })

        const out = createWriteStream(tmpPath, { mode: 0o600 })
        await pipeline(source, sizeEnforcer, out)

        if (written === 0) {
          return {
            text: '',
            contentType: opts.contentType ?? 'audio/wav',
            encoding: 'UTF-8',
            metadata: {},
            pages: 0,
            language: '',
            confidence: 0,
            skipped: true,
            reason: 'empty audio input',
          }
        }

        return await transcribe(tmpPath, written, opts, signal)
      } finally {
        await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
      }
    },

    async available(): Promise<boolean> {
      return checkAvailability()
    },

    capability(): ExtractorCapability {
      return {
        extensions: AUDIO_EXTENSIONS,
        mimeTypes: AUDIO_CONTENT_TYPES,
        magicBytes: AUDIO_MAGIC_SIGNATURES,
        requiresBinary: true,
      }
    },
  }

  return extractor
}

// SPDX-License-Identifier: Apache-2.0

/**
 * Audio transcription extractor using faster-whisper via Python
 * subprocess. Both Go and TS implementations use the same subprocess
 * approach -- whisper.cpp Go bindings are not production-ready.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { extname } from 'node:path'

import { checkBinaryAvailable, runSubprocess } from './subprocess.js'
import type { ExtractOptions, ExtractionResult, Extractor, ExtractorCapability } from './types.js'

export type AudioExtractorConfig = {
  readonly pythonBinary?: string
  readonly modelSize?: string
  readonly defaultLanguage?: string
  readonly maxFileSizeBytes?: number
  readonly timeoutPerMinute?: number
  readonly minTimeout?: number
}

export type TranscriptionSegment = {
  readonly start: number
  readonly end: number
  readonly text: string
}

type WhisperOutput = {
  readonly language: string
  readonly segments: readonly TranscriptionSegment[]
}

const WHISPER_SCRIPT = `
import sys, json
from faster_whisper import WhisperModel
model = WhisperModel(sys.argv[2], device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], language=sys.argv[3] if len(sys.argv) > 3 else None)
result = {"language": info.language, "segments": []}
for s in segments:
    result["segments"].append({"start": s.start, "end": s.end, "text": s.text})
json.dump(result, sys.stdout)
`

const DEFAULT_MAX_AUDIO_SIZE = 500 * 1024 * 1024
const DEFAULT_TIMEOUT_PER_MINUTE = 30_000
const DEFAULT_MIN_TIMEOUT = 120_000

const resolveConfig = (cfg?: AudioExtractorConfig) => ({
  pythonBinary: cfg?.pythonBinary ?? process.env['MEMORY_PYTHON_PATH'] ?? 'python3',
  modelSize: cfg?.modelSize ?? 'base',
  defaultLanguage: cfg?.defaultLanguage,
  maxFileSizeBytes: cfg?.maxFileSizeBytes ?? DEFAULT_MAX_AUDIO_SIZE,
  timeoutPerMinute: cfg?.timeoutPerMinute ?? DEFAULT_TIMEOUT_PER_MINUTE,
  minTimeout: cfg?.minTimeout ?? DEFAULT_MIN_TIMEOUT,
})

const formatTimestamp = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

const formatTranscription = (segments: readonly TranscriptionSegment[]): string => {
  if (segments.length === 0) return ''

  const paragraphs: string[] = []
  let paragraphStart = 0
  let paragraphTexts: string[] = []

  for (const seg of segments) {
    if (seg.start - paragraphStart >= 30 && paragraphTexts.length > 0) {
      paragraphs.push(
        `[${formatTimestamp(paragraphStart)} - ${formatTimestamp(seg.start)}]\n${paragraphTexts.join(' ')}`,
      )
      paragraphStart = seg.start
      paragraphTexts = []
    }
    const text = seg.text.trim()
    if (text.length > 0) {
      paragraphTexts.push(text)
    }
  }

  if (paragraphTexts.length > 0) {
    const lastEnd = segments[segments.length - 1]?.end ?? 0
    paragraphs.push(
      `[${formatTimestamp(paragraphStart)} - ${formatTimestamp(lastEnd)}]\n${paragraphTexts.join(' ')}`,
    )
  }

  return paragraphs.join('\n\n')
}

export class AudioExtractor implements Extractor {
  readonly name = 'audio-whisper' as const

  private readonly cfg: ReturnType<typeof resolveConfig>

  constructor(config?: AudioExtractorConfig) {
    this.cfg = resolveConfig(config)
  }

  get capability(): ExtractorCapability {
    return {
      extensions: new Set(['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.opus']),
      mimeTypes: new Set([
        'audio/mpeg',
        'audio/wav',
        'audio/x-wav',
        'audio/flac',
        'audio/mp4',
        'audio/ogg',
        'audio/aac',
        'audio/x-ms-wma',
        'audio/opus',
      ]),
      magicBytes: [
        { offset: 0, bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46]) }, // RIFF (WAV)
        { offset: 0, bytes: new Uint8Array([0x66, 0x4c, 0x61, 0x43]) }, // fLaC
        { offset: 0, bytes: new Uint8Array([0x4f, 0x67, 0x67, 0x53]) }, // OggS
        { offset: 0, bytes: new Uint8Array([0x49, 0x44, 0x33]) }, // ID3 (MP3)
        { offset: 0, bytes: new Uint8Array([0xff, 0xfb]) }, // MP3 sync
      ],
      requiresBinary: true,
    }
  }

  async available(): Promise<boolean> {
    const hasPython = await checkBinaryAvailable(this.cfg.pythonBinary)
    if (!hasPython) return false

    try {
      const result = await runSubprocess(this.cfg.pythonBinary, ['-c', 'import faster_whisper'], undefined, {
        timeout: 10_000,
      })
      return result.exitCode === 0
    } catch {
      return false
    }
  }

  async extract(input: Buffer, opts: ExtractOptions): Promise<ExtractionResult> {
    if (input.length > this.cfg.maxFileSizeBytes) {
      throw new Error(
        `extract: audio file size ${input.length} exceeds maximum ${this.cfg.maxFileSizeBytes} bytes`,
      )
    }

    const tmpDir = await mkdtemp(join(tmpdir(), 'memory-audio-'))
    try {
      const ext = opts.filename !== undefined ? extname(opts.filename) : '.wav'
      const tmpFile = join(tmpDir, `input${ext}`)
      await writeFile(tmpFile, input)

      const timeout = this.computeTimeout(input.length)
      const args = ['-c', WHISPER_SCRIPT, tmpFile, this.cfg.modelSize]
      const language = opts.language ?? this.cfg.defaultLanguage
      if (language !== undefined) {
        args.push(language)
      }

      const subprocessOpts = {
        timeout,
        ...(opts.signal !== undefined ? { signal: opts.signal } : {}),
      }
      const result = await runSubprocess(this.cfg.pythonBinary, args, undefined, subprocessOpts)
      if (result.exitCode !== 0) {
        throw new Error(`extract: whisper exited ${result.exitCode}: ${result.stderr}`)
      }

      const output = JSON.parse(result.stdout.toString('utf8')) as WhisperOutput
      const content = formatTranscription(output.segments)
      const metadata: Record<string, unknown> = {
        detected_language: output.language,
        segments: output.segments,
      }
      if (output.segments.length > 0) {
        const last = output.segments[output.segments.length - 1]
        if (last !== undefined) {
          metadata['duration_seconds'] = last.end
        }
      }

      return {
        content,
        mime: 'text/plain',
        metadata,
        language: output.language,
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  /**
   * Convenience method used by VideoExtractor when delegating extracted
   * audio data for transcription.
   */
  async extractFromBuffer(wavData: Buffer, opts: ExtractOptions): Promise<ExtractionResult> {
    return this.extract(wavData, { ...opts, filename: 'extracted.wav' })
  }

  private computeTimeout(sizeBytes: number): number {
    const estimatedMinutes = sizeBytes / (1024 * 1024)
    const computed = estimatedMinutes * this.cfg.timeoutPerMinute
    return Math.max(computed, this.cfg.minTimeout)
  }
}

export { formatTranscription, formatTimestamp }

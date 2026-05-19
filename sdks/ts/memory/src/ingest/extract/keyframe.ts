// SPDX-License-Identifier: Apache-2.0

/**
 * Keyframe extraction from video files. Captures frames at a
 * configurable interval using FFmpeg, then runs OCR on each frame
 * via a delegate extractor. Results are appended to the video
 * extraction output.
 */

import { mkdir, readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'

import { checkBinaryAvailable, runSubprocess } from './subprocess.js'
import type { ExtractOptions, ExtractResult, Extractor } from './types.js'

/** Default interval between keyframe captures in seconds. */
const DEFAULT_KEYFRAME_INTERVAL_SECS = 30

/** Configuration for keyframe extraction. */
export type KeyframeConfig = {
  readonly ffmpegBinary: string
  readonly intervalSecs: number
  readonly maxKeyframes: number
  readonly ocrExtractor?: Extractor
  readonly timeout: number
}

/** Text extracted from a single keyframe. */
export type KeyframeResult = {
  readonly timestampSecs: number
  readonly text: string
  readonly confidence: number
}

/** Resolves keyframe config with defaults and env var overrides. */
export const resolveKeyframeConfig = (partial: Partial<KeyframeConfig>): KeyframeConfig => {
  const envInterval = process.env['MEMORY_KEYFRAME_INTERVAL_SECS']
  let intervalSecs = partial.intervalSecs ?? DEFAULT_KEYFRAME_INTERVAL_SECS
  if (partial.intervalSecs === undefined && envInterval !== undefined && envInterval !== '') {
    const parsed = Number.parseInt(envInterval, 10)
    if (!Number.isNaN(parsed) && parsed > 0) {
      intervalSecs = parsed
    }
  }

  const base: KeyframeConfig = {
    ffmpegBinary: partial.ffmpegBinary ?? process.env['MEMORY_FFMPEG_PATH'] ?? 'ffmpeg',
    intervalSecs,
    maxKeyframes: partial.maxKeyframes ?? 20,
    timeout: partial.timeout ?? 60_000,
  }
  if (partial.ocrExtractor !== undefined) {
    return { ...base, ocrExtractor: partial.ocrExtractor }
  }
  return base
}

/**
 * Extracts keyframe images from a video file at the configured
 * interval, runs OCR on each frame, and returns the collected text
 * results.
 */
export const extractKeyframes = async (
  videoPath: string,
  tmpDir: string,
  cfg: KeyframeConfig,
  signal?: AbortSignal,
): Promise<readonly KeyframeResult[]> => {
  if (cfg.ocrExtractor === undefined) {
    return []
  }

  const ocrAvailable = await cfg.ocrExtractor.available()
  if (!ocrAvailable) {
    return []
  }

  const framesDir = join(tmpDir, 'keyframes')
  await mkdir(framesDir, { recursive: true })

  const fpsFilter = `fps=1/${cfg.intervalSecs}`
  const outputPattern = join(framesDir, 'frame-%04d.png')

  const args = [
    '-i', videoPath,
    '-vf', fpsFilter,
    '-frames:v', String(cfg.maxKeyframes),
    '-vsync', 'vfr',
    outputPattern,
  ]

  const subprocessOpts = signal !== undefined
    ? { timeout: cfg.timeout, signal }
    : { timeout: cfg.timeout }

  const result = await runSubprocess(cfg.ffmpegBinary, args, undefined, subprocessOpts)
  if (result.exitCode !== 0) {
    throw new Error(
      `ingest: ffmpeg keyframe extraction exited with code ${result.exitCode}: ${result.stderr}`,
    )
  }

  const files = await readdir(framesDir)
  let frameFiles = files
    .filter((f) => f.startsWith('frame-') && f.endsWith('.png'))
    .sort()

  if (frameFiles.length > cfg.maxKeyframes) {
    frameFiles = frameFiles.slice(0, cfg.maxKeyframes)
  }

  const keyframeResults: KeyframeResult[] = []

  for (let i = 0; i < frameFiles.length; i++) {
    const frameFile = frameFiles[i]
    if (frameFile === undefined) continue

    const framePath = join(framesDir, frameFile)

    let imageData: Buffer
    try {
      imageData = await readFile(framePath)
    } catch {
      continue
    }

    const ocrOpts: ExtractOptions = {
      contentType: 'image/png',
      fileName: frameFile,
    }

    try {
      const ocrResult = await cfg.ocrExtractor.extract(imageData, ocrOpts, signal)
      if (!ocrResult.skipped && ocrResult.text.length > 0) {
        const timestampSecs = (i + 1) * cfg.intervalSecs

        keyframeResults.push({
          timestampSecs,
          text: ocrResult.text,
          confidence: ocrResult.confidence,
        })
      }
    } catch {
      continue
    }
  }

  return keyframeResults
}

/**
 * Formats keyframe OCR results into a structured text section suitable
 * for appending to the video extraction result.
 */
export const formatKeyframeText = (results: readonly KeyframeResult[]): string => {
  if (results.length === 0) return ''

  const lines: string[] = ['\n\n--- Keyframe Text ---']

  for (const kf of results) {
    const timestamp = formatKeyframeTimestamp(kf.timestampSecs)
    lines.push(`\n[${timestamp}]\n${kf.text}`)
  }

  return lines.join('\n')
}

/**
 * Converts seconds to MM:SS format.
 */
export const formatKeyframeTimestamp = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

/**
 * Adds keyframe-related metadata entries to the provided metadata
 * record. Returns a new record with the additions.
 */
export const mergeKeyframeMetadata = (
  metadata: Record<string, string>,
  results: readonly KeyframeResult[],
): void => {
  metadata['keyframe_count'] = String(results.length)

  if (results.length === 0) return

  let totalConf = 0
  for (const kf of results) {
    totalConf += kf.confidence
  }
  const avgConf = totalConf / results.length
  metadata['keyframe_avg_confidence'] = avgConf.toFixed(4)
}

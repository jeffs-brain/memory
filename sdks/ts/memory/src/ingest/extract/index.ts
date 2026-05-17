// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for the pluggable extractor subsystem.
 */

export type {
  ExtractResult,
  Extractor,
  ExtractOptions,
  ExtractorCapability,
  MagicSignature,
} from './types.js'

export {
  AudioExtractor,
  type AudioExtractorConfig,
  type TranscriptionSegment,
} from './audio.js'

export {
  VideoExtractor,
  type VideoExtractorConfig,
  type VideoMetadata,
} from './video.js'

export {
  extractKeyframes,
  formatKeyframeText,
  formatKeyframeTimestamp,
  mergeKeyframeMetadata,
  resolveKeyframeConfig,
  type KeyframeConfig,
  type KeyframeResult,
} from './keyframe.js'

export { checkBinaryAvailable, runSubprocess, resetBinaryCache } from './subprocess.js'

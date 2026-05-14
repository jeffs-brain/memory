import type { ExtensionAPI } from "@earendil-works/pi-coding-agent"
import { MemoryExtensionConfigSchema, type MemoryExtensionConfig } from "./config.js"

export type MemoryExtension = {
  flush: () => Promise<void>
  close: () => Promise<void>
}

export function createMemoryExtension(
  _pi: ExtensionAPI,
  _config?: MemoryExtensionConfig,
): MemoryExtension {
  throw new Error(
    "@jeffs-brain/memory-pi.createMemoryExtension: not implemented yet. W3 fills this in.",
  )
}

export default function (pi: ExtensionAPI): void {
  const raw = process.env.MEMORY_PI_CONFIG
    ? JSON.parse(process.env.MEMORY_PI_CONFIG)
    : {}
  const config = MemoryExtensionConfigSchema.parse(raw)
  createMemoryExtension(pi, config)
}

export { MemoryExtensionConfigSchema, type MemoryExtensionConfig } from "./config.js"

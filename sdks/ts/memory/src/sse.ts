// SPDX-License-Identifier: Apache-2.0

export type SseFrame = {
  readonly event: string
  readonly data: string
  readonly id?: string
}

const normaliseLineBreaks = (value: string): string => value.replace(/\r\n?/g, '\n')

const normaliseFieldValue = (value: string): string =>
  normaliseLineBreaks(value).replaceAll('\n', '')

export const formatSseFrame = (frame: SseFrame): string => {
  const lines = [`event: ${normaliseFieldValue(frame.event)}`]
  if (frame.id !== undefined) {
    lines.push(`id: ${normaliseFieldValue(frame.id)}`)
  }
  for (const line of normaliseLineBreaks(frame.data).split('\n')) {
    lines.push(`data: ${line}`)
  }
  lines.push('', '')
  return lines.join('\n')
}

export const createSseHeartbeat = (
  intervalMs: number,
  onHeartbeat: () => void,
  clock: Pick<typeof globalThis, 'setInterval' | 'clearInterval'> = globalThis,
): (() => void) => {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new RangeError('intervalMs must be greater than 0')
  }
  let stopped = false
  const timer = clock.setInterval(() => {
    if (!stopped) onHeartbeat()
  }, intervalMs)
  return () => {
    if (stopped) return
    stopped = true
    clock.clearInterval(timer)
  }
}

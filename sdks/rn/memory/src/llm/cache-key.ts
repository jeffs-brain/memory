const FNV_OFFSET = 0xcbf29ce484222325n
const FNV_PRIME = 0x100000001b3n
const MASK_64 = 0xffffffffffffffffn

export const hashText = (input: string): string => {
  let hash = FNV_OFFSET
  const bytes = new TextEncoder().encode(input)
  for (let index = 0; index < bytes.length; index += 1) {
    hash ^= BigInt(bytes[index] ?? 0)
    hash = (hash * FNV_PRIME) & MASK_64
  }
  return hash.toString(16).padStart(16, '0')
}

export const createCacheKey = (model: string, text: string): string => {
  return `${model}\x1f${hashText(text)}`
}

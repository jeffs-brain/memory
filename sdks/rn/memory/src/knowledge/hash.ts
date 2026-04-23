const SHA256_K = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
] as const

const SHA256_INIT = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
] as const

const textEncoder = new TextEncoder()
const textDecoder = new TextDecoder('utf-8')

export type HashContentInput = string | Uint8Array | ArrayBuffer

export const toBytes = (input: HashContentInput): Uint8Array => {
  if (typeof input === 'string') return textEncoder.encode(input)
  if (input instanceof Uint8Array) return input
  return new Uint8Array(input)
}

export const toText = (input: HashContentInput): string => {
  if (typeof input === 'string') return input
  return textDecoder.decode(toBytes(input))
}

export const contentByteLength = (input: HashContentInput): number => toBytes(input).byteLength

const rotateRight = (value: number, shift: number): number =>
  (value >>> shift) | (value << (32 - shift))

const add = (...values: readonly number[]): number =>
  values.reduce((sum, value) => (sum + value) >>> 0, 0)

const choose = (x: number, y: number, z: number): number => (x & y) ^ (~x & z)
const majority = (x: number, y: number, z: number): number => (x & y) ^ (x & z) ^ (y & z)
const sigma0 = (x: number): number => rotateRight(x, 2) ^ rotateRight(x, 13) ^ rotateRight(x, 22)
const sigma1 = (x: number): number => rotateRight(x, 6) ^ rotateRight(x, 11) ^ rotateRight(x, 25)
const gamma0 = (x: number): number => rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >>> 3)
const gamma1 = (x: number): number => rotateRight(x, 17) ^ rotateRight(x, 19) ^ (x >>> 10)

export const hashContent = (input: HashContentInput): string => {
  const bytes = toBytes(input)
  const bitLength = BigInt(bytes.byteLength) * 8n
  const paddedLength = Math.ceil((bytes.byteLength + 1 + 8) / 64) * 64
  const padded = new Uint8Array(paddedLength)
  padded.set(bytes, 0)
  padded[bytes.byteLength] = 0x80
  for (let offset = 0; offset < 8; offset += 1) {
    padded[paddedLength - 1 - offset] = Number((bitLength >> BigInt(offset * 8)) & 0xffn)
  }

  const words = new Uint32Array(64)
  const view = new DataView(padded.buffer)
  const hash: number[] = [...SHA256_INIT]

  for (let chunkOffset = 0; chunkOffset < paddedLength; chunkOffset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(chunkOffset + index * 4, false)
    }
    for (let index = 16; index < 64; index += 1) {
      words[index] = add(
        gamma1(words[index - 2] ?? 0),
        words[index - 7] ?? 0,
        gamma0(words[index - 15] ?? 0),
        words[index - 16] ?? 0,
      )
    }

    let [a, b, c, d, e, f, g, h] = hash
    for (let index = 0; index < 64; index += 1) {
      const temp1 = add(
        h ?? 0,
        sigma1(e ?? 0),
        choose(e ?? 0, f ?? 0, g ?? 0),
        SHA256_K[index] ?? 0,
        words[index] ?? 0,
      )
      const temp2 = add(sigma0(a ?? 0), majority(a ?? 0, b ?? 0, c ?? 0))
      h = g
      g = f
      f = e
      e = add(d ?? 0, temp1)
      d = c
      c = b
      b = a
      a = add(temp1, temp2)
    }

    hash[0] = add(hash[0] ?? 0, a ?? 0)
    hash[1] = add(hash[1] ?? 0, b ?? 0)
    hash[2] = add(hash[2] ?? 0, c ?? 0)
    hash[3] = add(hash[3] ?? 0, d ?? 0)
    hash[4] = add(hash[4] ?? 0, e ?? 0)
    hash[5] = add(hash[5] ?? 0, f ?? 0)
    hash[6] = add(hash[6] ?? 0, g ?? 0)
    hash[7] = add(hash[7] ?? 0, h ?? 0)
  }

  return hash.map((value) => value.toString(16).padStart(8, '0')).join('')
}

import { type Store, createMobileStore } from '../store/index.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'

export const createTestStore = async (): Promise<Store> =>
  createMobileStore({
    root: '/brains/knowledge',
    adapter: createMemoryFileAdapter(),
  })

export const makeWords = (count: number): string =>
  Array.from({ length: count }, (_, index) => `word${index + 1}`).join(' ')

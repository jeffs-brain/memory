import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: '@jeffs-brain/memory-openfga',
    include: ['src/**/*.test.ts'],
  },
})

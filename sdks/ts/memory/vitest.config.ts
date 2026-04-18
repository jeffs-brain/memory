import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: '@jeffs-brain/memory',
    include: ['src/**/*.test.ts'],
  },
})

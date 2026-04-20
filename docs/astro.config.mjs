// SPDX-License-Identifier: Apache-2.0

import starlight from '@astrojs/starlight'
import { defineConfig } from 'astro/config'

export default defineConfig({
  site: 'https://docs.jeffsbrain.com',
  integrations: [
    starlight({
      title: 'jeffs-brain/memory',
      description: 'Cross-language memory library for LLM agents.',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/jeffs-brain/memory',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/jeffs-brain/memory/edit/main/docs/',
      },
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        { label: 'Getting Started', autogenerate: { directory: 'getting-started' } },
        { label: 'MCP Integration', autogenerate: { directory: 'mcp' } },
        { label: 'Concepts', autogenerate: { directory: 'concepts' } },
        { label: 'Guides', autogenerate: { directory: 'guides' } },
        { label: 'Spec', autogenerate: { directory: 'spec' } },
        { label: 'Examples', autogenerate: { directory: 'examples' } },
        { label: 'Reference', autogenerate: { directory: 'reference' } },
      ],
    }),
  ],
})

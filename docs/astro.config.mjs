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
        { label: 'Getting Started', items: [{ autogenerate: { directory: 'getting-started' } }] },
        { label: 'MCP Integration', items: [{ autogenerate: { directory: 'mcp' } }] },
        { label: 'Concepts', items: [{ autogenerate: { directory: 'concepts' } }] },
        { label: 'Guides', items: [{ autogenerate: { directory: 'guides' } }] },
        { label: 'Spec', items: [{ autogenerate: { directory: 'spec' } }] },
        { label: 'Examples', items: [{ autogenerate: { directory: 'examples' } }] },
        { label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
      ],
    }),
  ],
})

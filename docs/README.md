# jeffs-brain/memory docs

Public documentation site for `jeffs-brain/memory`. Powered by [Astro Starlight](https://starlight.astro.build). Deployed to [docs.jeffsbrain.com](https://docs.jeffsbrain.com) on Cloudflare Pages.

21 pages across the landing section, per-SDK getting-started (TypeScript, Go, Python), per-agent MCP integration guides (Claude Code, Claude Desktop, Cursor, Windsurf, Zed), concept pages (brains, memory stages, retrieval, query DSL), spec mirrors, example walkthroughs, and the CLI reference.

## Develop

```bash
cd docs
bun install
bun run dev
```

Open http://localhost:4321. Hot reload covers every MDX file under `src/content/docs/`.

## Build

```bash
bun run build
```

Static site output lands in `./dist`.

## Preview the production build

```bash
bun run preview
```

## Deploy (Cloudflare Pages)

One-shot deploy from a local machine:

```bash
cd docs
bun run build
npx wrangler pages deploy ./dist --project-name jeffs-brain-docs
```

Wrangler reads `wrangler.toml`:

```toml
name = "jeffs-brain-docs"
compatibility_date = "2026-04-18"
pages_build_output_dir = "./dist"
```

For CI-based deploys, connect the `jeffs-brain/memory` repo in the Cloudflare Pages dashboard with:

- **Framework preset**: Astro
- **Build command**: `cd docs && bun install && bun run build`
- **Build output directory**: `docs/dist`
- **Root directory**: repo root
- **Environment variables**: none required for the site build.

Production domain `docs.jeffsbrain.com` is bound via **Custom domains** in the Pages project.

## Content layout

```
src/content/docs/
├── index.mdx                 # landing
├── getting-started/          # per-SDK quick starts
├── mcp/                      # per-agent MCP integration guides
├── concepts/                 # brains, memory stages, retrieval, query DSL
├── spec/                     # authoritative spec docs (copied from ../spec)
├── examples/                 # runnable example walkthroughs
└── reference/                # CLI reference
```

The `spec/` pages are generated from `../spec/*.md` at scaffold time; regenerate them whenever the upstream spec changes. A future task will automate this via an Astro content loader or prebuild step.

## Licence

Apache-2.0. See the repo-level `LICENSE`.

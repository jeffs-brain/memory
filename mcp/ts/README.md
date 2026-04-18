# @jeffs-brain/memory-mcp

Stdio Model Context Protocol server for [`@jeffs-brain/memory`](https://www.npmjs.com/package/@jeffs-brain/memory). Gives agents (Claude Code, Claude Desktop, Cursor, Windsurf, Zed) first-class access to a local or hosted Jeffs Brain.

## Install

```bash
npm i -g @jeffs-brain/memory-mcp
# or run on demand
npx -y @jeffs-brain/memory-mcp
```

## Claude Code

```bash
# Local brain (default: ~/.jeffs-brain)
claude mcp add jeffs-brain -- npx -y @jeffs-brain/memory-mcp

# Hosted brain
claude mcp add jeffs-brain --env JB_TOKEN=... -- npx -y @jeffs-brain/memory-mcp
```

## Claude Desktop, Cursor, Windsurf

Add to the app's MCP config:

```json
{
  "mcpServers": {
    "jeffs-brain": {
      "command": "npx",
      "args": ["-y", "@jeffs-brain/memory-mcp"],
      "env": { "JB_TOKEN": "optional-hosted-token" }
    }
  }
}
```

## Modes

Selected automatically:

- **Local** (no `JB_TOKEN`): FsStore rooted at `$JB_HOME` (default `~/.jeffs-brain/`), sqlite-backed search, Ollama embeddings at `http://localhost:11434` when available with BM25-only fallback. No API key required.
- **Hosted** (`JB_TOKEN` set): HttpStore against `$JB_ENDPOINT` (default `https://api.jeffsbrain.com`) using the token as a bearer credential.

## Environment variables

| Variable      | Default                      | Notes                                                     |
| ------------- | ---------------------------- | --------------------------------------------------------- |
| `JB_TOKEN`    | unset                        | When set, flips to hosted mode.                           |
| `JB_ENDPOINT` | `https://api.jeffsbrain.com` | Hosted API base URL.                                      |
| `JB_HOME`     | `~/.jeffs-brain`             | Local FsStore root.                                       |
| `JB_BRAIN`    | unset                        | Default brain id or slug applied when tool args omit `brain`. |
| `OLLAMA_HOST` | `http://localhost:11434`     | Local embedder endpoint. BM25-only if unreachable.        |

## Tools

Eleven `memory_*` tools shared with the Go and Python SDKs:

- `memory_remember`
- `memory_recall`
- `memory_search`
- `memory_ask`
- `memory_ingest_file`
- `memory_ingest_url`
- `memory_extract`
- `memory_reflect`
- `memory_consolidate`
- `memory_create_brain`
- `memory_list_brains`

Input and output shapes are specified in [`spec/MCP-TOOLS.md`](https://github.com/jeffs-brain/memory/blob/main/spec/MCP-TOOLS.md).

## Docs

- Repo README: https://github.com/jeffs-brain/memory#readme
- Protocol and MCP spec: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

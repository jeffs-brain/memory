# jeffs-brain-memory-mcp

Stdio Model Context Protocol server for [`jeffs-brain-memory`](https://pypi.org/project/jeffs-brain-memory/). Gives agents (Claude Code, Claude Desktop, Cursor, Windsurf, Zed) first-class access to a local or hosted Jeffs Brain.

One of three wire-compatible MCP wrappers; its counterparts are the TypeScript `@jeffs-brain/memory-mcp` package and the Go `cmd/memory-mcp` binary. All three expose the same 11 `memory_*` tools defined in [`spec/MCP-TOOLS.md`](https://github.com/jeffs-brain/memory/blob/main/spec/MCP-TOOLS.md).

## Install

```bash
uv tool install jeffs-brain-memory-mcp
# or run on demand
uvx jeffs-brain-memory-mcp
```

For a one-shot install across every supported agent, use the TS orchestrator:

```bash
npx @jeffs-brain/install
```

## Claude Code

```bash
# Local brain (default: ~/.jeffs-brain)
claude mcp add jeffs-brain -- uvx jeffs-brain-memory-mcp

# Hosted brain
claude mcp add jeffs-brain --env JB_TOKEN=... -- uvx jeffs-brain-memory-mcp
```

## Claude Desktop, Cursor, Windsurf

Add to the app's MCP config:

```json
{
  "mcpServers": {
    "jeffs-brain": {
      "command": "uvx",
      "args": ["jeffs-brain-memory-mcp"],
      "env": { "JB_TOKEN": "optional-hosted-token" }
    }
  }
}
```

## Zed (`context_servers`)

```json
{
  "context_servers": {
    "jeffs-brain": {
      "source": "custom",
      "command": "uvx",
      "args": ["jeffs-brain-memory-mcp"],
      "env": { "JB_TOKEN": "optional-hosted-token" }
    }
  }
}
```

## Modes

Selected automatically:

- **Local** (no `JB_TOKEN`): filesystem store rooted at `$JB_HOME` (default `~/.jeffs-brain/`), SQLite FTS5 search, Ollama embeddings at `http://localhost:11434` when available with BM25-only fallback. No API key required.
- **Hosted** (`JB_TOKEN` set): HTTP client against `$JB_ENDPOINT` (default `https://api.jeffsbrain.com`) using the token as a bearer credential.

## Environment variables

| Variable      | Default                      | Notes                                                     |
| ------------- | ---------------------------- | --------------------------------------------------------- |
| `JB_TOKEN`    | unset                        | When set, flips to hosted mode.                           |
| `JB_ENDPOINT` | `https://api.jeffsbrain.com` | Hosted API base URL.                                      |
| `JB_HOME`     | `~/.jeffs-brain`             | Local store root.                                         |
| `JB_BRAIN`    | unset                        | Default brain id or slug applied when tool args omit `brain`. |
| `OLLAMA_HOST` | `http://localhost:11434`     | Local embedder endpoint. BM25-only if unreachable.        |

## Tools

Eleven `memory_*` tools shared with the TypeScript and Go MCP wrappers:

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
- Protocol and MCP spec: https://github.com/jeffs-brain/memory/tree/main/spec
- Docs site: https://docs.jeffsbrain.com

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

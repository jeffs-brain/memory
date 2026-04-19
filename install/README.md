# @jeffs-brain/install

One-shot installer that wires the [`@jeffs-brain/memory-mcp`](../mcp/ts) server into every MCP-capable agent on your machine. Run once, get first-class `jeffs-brain` tools in Claude Code, Claude Desktop, Cursor, Windsurf, and Zed.

## Run

```bash
npx @jeffs-brain/install
```

Follow the prompts: pick which agents to wire up, choose `local` (default, zero-config) or `hosted`, and accept the default storage path (`~/.jeffs-brain`).

## Supported agents

| Agent          | Config path                                                                   |
| -------------- | ----------------------------------------------------------------------------- |
| Claude Code    | `~/.claude/claude.json` or `~/.config/claude/claude.json`                     |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), `%APPDATA%/Claude/claude_desktop_config.json` (Windows), `~/.config/Claude/claude_desktop_config.json` (Linux) |
| Cursor         | `~/.cursor/mcp.json` or `~/.config/cursor/mcp.json`                           |
| Windsurf       | `~/.codeium/windsurf/mcp_config.json`                                         |
| Zed            | `~/.config/zed/settings.json`                                                 |

If a config file is missing the installer creates it, bootstrapping parent directories as needed.

## Non-interactive

```bash
npx @jeffs-brain/install \
  --agents claude-code,cursor \
  --mode local \
  --storage ~/.jeffs-brain \
  --non-interactive
```

Hosted mode (CI or scripted install) picks the token up from `JB_TOKEN` or `--token`:

```bash
JB_TOKEN=jbp_xxx npx @jeffs-brain/install \
  --agents claude-code \
  --mode hosted \
  --endpoint https://api.jeffsbrain.com \
  --non-interactive
```

`--dry-run` prints the plan without writing anything.

### Flags

| Flag                | Notes                                                         |
| ------------------- | ------------------------------------------------------------- |
| `--agents`          | Comma list from `claude-code,claude-desktop,cursor,windsurf,zed`. |
| `--mode`            | `local` or `hosted`.                                          |
| `--storage`         | Override `JB_HOME` on the written env block.                  |
| `--endpoint`        | Hosted API URL; defaults to `https://api.jeffsbrain.com`.     |
| `--token`           | Hosted bearer token; `JB_TOKEN` in env is equivalent.         |
| `--non-interactive` | Skip prompts. All required values must be on the flag set.    |
| `--dry-run`         | Print the merged plan without writing.                        |

## What it writes

Each agent config is merged (never overwritten wholesale). If a `jeffs-brain` entry already exists, the file is backed up to `<config>.jbpre-<timestamp>.bak` and the entry is replaced.

### Claude Code / Claude Desktop / Cursor / Windsurf

```json
{
  "mcpServers": {
    "jeffs-brain": {
      "command": "npx",
      "args": ["-y", "@jeffs-brain/memory-mcp"],
      "env": { "JB_HOME": "/home/you/.jeffs-brain" }
    }
  }
}
```

### Zed (`context_servers`)

```json
{
  "context_servers": {
    "jeffs-brain": {
      "source": "custom",
      "command": "npx",
      "args": ["-y", "@jeffs-brain/memory-mcp"],
      "env": { "JB_HOME": "/home/you/.jeffs-brain" }
    }
  }
}
```

## Post-install smoke tests

```bash
# Claude Code
claude mcp add jeffs-brain --env JB_HOME=~/.jeffs-brain -- npx -y @jeffs-brain/memory-mcp

# Claude Desktop
# Restart Claude Desktop, then ask: "What MCP tools are available?"

# Cursor
# Reload window, open MCP panel, confirm jeffs-brain is listed.

# Windsurf
# Restart Windsurf, Cascade settings -> MCP, confirm jeffs-brain is green.

# Zed
# Restart Zed, run `zed: context server status` from the command palette.
```

## Roadmap

- Real device-flow OAuth for hosted mode (today it stubs device flow and prompts for a pasted token).
- OS keychain binding (macOS Keychain, libsecret, Windows Credential Manager).

## Licence

Apache-2.0. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

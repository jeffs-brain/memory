# Security Policy

## Supported Versions

Only the latest minor release of each published package is supported. Check the GitHub releases page for current versions across the TypeScript and Go SDKs. The Python SDK is not yet published.

## Reporting a Vulnerability

Report vulnerabilities privately to `jeff@jeffsbrain.com`, or via GitHub's private vulnerability reporting at https://github.com/jeffs-brain/memory/security/advisories/new.

Do not open public issues for security problems.

### What to include

- Affected SDK (TS, Go, or Python) and version
- Affected component (core SDK, MCP wrapper, install orchestrator, sibling adapter package)
- Steps to reproduce
- Impact assessment (CVSS score or narrative)
- Any known mitigation or workaround

### Response

- Acknowledgement within 72 hours
- Triage and severity assessment within 7 days
- Fix or mitigation plan communicated within 14 days for high-severity issues
- Coordinated disclosure once a fix is available; credit given unless you request otherwise

### Scope

In scope:

- Any SDK in this repository (`sdks/ts`, `go`, `sdks/py`)
- Any MCP wrapper in this repository (`mcp/ts`, `go/cmd/memory-mcp`, `mcp/py`)
- The pi extension `@jeffs-brain/memory-pi`
- The `@jeffs-brain/install` orchestrator under `install/`
- Sibling adapter packages published from this repo (`@jeffs-brain/memory-postgres`, `@jeffs-brain/memory-openfga`)
- Specification and conformance fixtures under `spec/`

Out of scope:

- The `jeffs-brain/platform` hosted service (report to the platform team separately)
- Third-party dependencies listed in `NOTICE` (report upstream, but feel free to cc us)
- Vulnerabilities that require a local attacker with existing access to the user's filesystem or OS keychain

## Running the daemon safely

`memory serve` reads and writes brains on the host it runs on. Its defaults are safe for local use; see the "Daemon security" section of [`spec/PROTOCOL.md`](./spec/PROTOCOL.md) for the full rules.

- It binds `127.0.0.1` by default and refuses a non-loopback address unless a bearer token is set (`--auth-token` or `JB_AUTH_TOKEN`). Use a long random token and put TLS in front of any daemon reachable from another machine.
- Without a token it answers only requests addressed to a loopback name, and refuses cross-origin browser requests.
- `ingest/file` reads a server-side path only when `--ingest-root` is set, and only inside that directory. Point it at a directory that holds nothing but documents meant for the brain.
- `ingest/url` refuses private, loopback, link-local and other non-public targets.

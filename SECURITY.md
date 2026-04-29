# Security Policy

## Supported Versions

Until v1.0, only the latest minor release of each SDK is supported. Check the GitHub releases page for current version status across TypeScript, Go, and Python SDKs.

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
- Any MCP wrapper in this repository (`mcp/ts`, `mcp/go`, `mcp/py`)
- The `@jeffs-brain/install` orchestrator under `install/`
- Sibling adapter packages published from this repo (`@jeffs-brain/memory-postgres`, `@jeffs-brain/memory-openfga`)
- Specification and conformance fixtures under `spec/`

Out of scope:

- The `jeffs-brain/platform` hosted service (report to the platform team separately)
- Third-party dependencies listed in `NOTICE` (report upstream, but feel free to cc us)
- Vulnerabilities that require a local attacker with existing access to the user's filesystem or OS keychain

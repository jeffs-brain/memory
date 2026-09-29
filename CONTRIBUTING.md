# Contributing

Thanks for your interest. All contributions to this repository are licensed under Apache-2.0.

By submitting a change, you agree to our [Code of Conduct](./CODE_OF_CONDUCT.md) and confirm that your contribution meets the Developer Certificate of Origin (see below).

## Local setup

The TypeScript packages use Bun workspaces:

```bash
bun install
bun run typecheck
bun run lint
bun run test
```

Node 20+ is required for the published packages. A local SQLite toolchain is needed for `better-sqlite3` in the `@jeffs-brain/memory` SDK. The media extractor tests need `ffmpeg` and `ffprobe` on the `PATH`, and the Postgres adapter tests need Docker; both skip when the tool is missing.

Go (`go/`):

```bash
cd go
go vet ./...
go test -race ./...
```

Python (`sdks/py/` and `mcp/py/`), with [uv](https://docs.astral.sh/uv/):

```bash
cd sdks/py
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new user-visible capability
- `fix:` bug fix
- `chore:` tooling, dependencies, CI
- `docs:` documentation only
- `refactor:` internal change with no behavioural effect
- `test:` tests only
- `spec:` changes to the cross-language behaviour contract under `spec/`

Keep subjects under 72 characters. The body explains the why; the diff covers the what.

## Developer Certificate of Origin

All commits must be signed off under the [DCO](https://developercertificate.org/):

```bash
git commit -s -m "feat: add retrieval strategy X"
```

There is no CLA. Sign-off is sufficient.

## Pull request process

1. Open an issue for substantive changes so we can align on approach before you write code.
2. Keep PRs focused, one concern per PR where possible.
3. Update or add tests alongside behaviour changes.
4. Update the spec in `spec/` before changing wire behaviour in any SDK. The spec is the source of truth for cross-language parity.
5. Run the checks for every SDK you touched (see "Local setup") before requesting review. CI runs the same commands.
6. PR descriptions should explain the change and link any related issue.

## Per-SDK notes

- **TypeScript (`sdks/ts/*`)**: the core package is `@jeffs-brain/memory`, with the adapters `@jeffs-brain/memory-postgres` and `@jeffs-brain/memory-openfga` and the pi extension `@jeffs-brain/memory-pi`. The MCP server lives at `mcp/ts` and the installer at `install/`.
- **Go (`go/`)**: the module `github.com/jeffs-brain/memory/go`, with the `memory` CLI and daemon under `cmd/memory` and the MCP server under `cmd/memory-mcp`.
- **Python (`sdks/py/`)**: not yet on PyPI. `memory serve` is production-ready; the rest of the local CLI is scaffolded. The MCP server lives at `mcp/py`.

## Conformance tests

Cross-SDK behaviour is pinned by fixtures in `spec/fixtures/` and conformance tests under `spec/conformance/`. The TypeScript store contract runs via:

```bash
npx vitest run sdks/ts/memory/src/store/contract.test.ts
```

Any SDK claiming conformance must pass the shared fixtures. Adding new behaviour means adding a fixture first, wiring it through the TS SDK, then porting to Go and Python as they ship.

## Spec changes

Wire format, storage layout, query DSL, and MCP tool contracts are defined under `spec/`. Changes to any SDK that alter observable behaviour must land the spec update in the same PR, or in a preceding PR that the SDK change references.

## Releasing

Releases are cut from `main` by pushing a tag; nothing publishes on merge.

1. Bump the versions in the same PR as the changes: each changed npm package's `package.json` (and any sibling ranges that need the new version), and `go/internal/version/version.go` for the Go module. Move the `CHANGELOG.md` entries from `Unreleased` to the new version.
2. After merge, tag the merge commit: `vX.Y.Z` publishes the npm packages and must equal `@jeffs-brain/memory`'s version; `go/vX.Y.Z` releases the Go module. Both must equal the Go version constant.
3. The `release` workflow runs typecheck, lint, the full test suites and the build as blocking steps, then publishes each npm package whose version is not already on the registry. A `workflow_dispatch` run against a branch runs the same gates as a dry run and publishes nothing.

Do not publish from a workstation: a release that does not come from a tag has no provenance and no changelog trail.

## Further reading

- [`SECURITY.md`](./SECURITY.md) - how to report vulnerabilities
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) - community standards
- [`LICENSE`](./LICENSE) - Apache-2.0 terms
- [`NOTICE`](./NOTICE) - bundled third-party components

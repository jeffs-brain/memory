# Contributing

Thanks for your interest. All contributions to this repository are licensed under Apache-2.0.

By submitting a change, you agree to our [Code of Conduct](./CODE_OF_CONDUCT.md) and confirm that your contribution meets the Developer Certificate of Origin (see below).

## Local setup

The TypeScript monorepo uses Bun workspaces:

```bash
bun install
bun run typecheck
bun run test
bun run lint
```

Node 20+ is required for the published packages. A local SQLite toolchain is needed for `better-sqlite3` in the `@jeffs-brain/memory` SDK.

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
5. Run `bun run typecheck`, `bun run test`, and `bun run lint` locally before requesting review.
6. PR descriptions should explain the change and link any related issue.

## Per-SDK notes

- **TypeScript (`sdks/ts/*`)**: shipping today at `0.0.x`. Core package is `@jeffs-brain/memory`, with adapters `@jeffs-brain/memory-postgres` and `@jeffs-brain/memory-openfga`. MCP wrapper lives at `mcp/ts`.
- **Go (`go/`)**: coming in Phase 3. Contributions welcome once the scaffold lands.
- **Python (`sdks/py/`)**: coming in Phase 4. Contributions welcome once the scaffold lands.

## Conformance tests

Cross-SDK behaviour is pinned by fixtures in `spec/fixtures/` and conformance tests under `spec/conformance/`. The TypeScript harness runs via:

```bash
bun test sdks/ts/memory/src/store/contract.test.ts
```

Any SDK claiming conformance must pass the shared fixtures. Adding new behaviour means adding a fixture first, wiring it through the TS SDK, then porting to Go and Python as they ship.

## Spec changes

Wire format, storage layout, query DSL, and MCP tool contracts are defined under `spec/`. Changes to any SDK that alter observable behaviour must land the spec update in the same PR, or in a preceding PR that the SDK change references.

## Further reading

- [`SECURITY.md`](./SECURITY.md) - how to report vulnerabilities
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) - community standards
- [`LICENSE`](./LICENSE) - Apache-2.0 terms
- [`NOTICE`](./NOTICE) - bundled third-party components

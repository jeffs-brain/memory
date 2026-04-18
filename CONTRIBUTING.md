# Contributing

Thanks for your interest. All contributions to this repository are licensed under Apache-2.0.

## Local setup

```bash
bun install
bun run typecheck
bun run test
```

## Parity across languages

[`spec/`](./spec) is the source of truth for cross-language parity. Any behavioural change to an SDK should be reflected in the spec first, and ideally covered by a conformance fixture under `spec/conformance`.

## Workflow

- Open an issue to discuss substantive changes before you raise a PR
- Keep PRs focused — one concern per PR where possible
- Add or update tests alongside behaviour changes
- Follow the existing code style; `bun run lint` checks formatting

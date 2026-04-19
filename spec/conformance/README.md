# HTTP conformance suite

This suite drives any Jeffs Brain SDK's HTTP store implementation against the canonical wire contract. The cases live in `http-contract.json` and are transport-agnostic: each case is a request or a sequence of requests, plus an expected response shape. A generic test harness can replay them against any language's `memory serve` binary.

## Running

1. Spin up the SDK's HTTP store in serve mode:

   - TypeScript: `bunx @jeffs-brain/memory serve --addr 127.0.0.1:<port>`
   - Go: `memory serve --addr :<port>`
   - Python: `uv run memory serve --addr 127.0.0.1:<port>` (or `python -m jeffs_brain_memory.cli.main serve --addr ...`)

2. Run the harness against the base URL and a freshly provisioned brain id. The harness:

   - Reads `http-contract.json`.
   - Substitutes `BRAIN_ID` and any `BASE64_*` placeholders.
   - For each case, executes the `setup` requests in order (failing fast on unexpected status), issues the primary `request`, asserts the response against `expectedResponse`, then runs any `followUp` and `teardown` steps.
   - For SSE cases, opens a named event stream, waits on specific frames, and closes when instructed.

3. Fail on any mismatch: status, body (exact, or subset assertion), JSON field equality, list-equality over returned paths, or SSE frame arrival.

## Assertion shapes

- `expectedResponse.status`: exact HTTP status.
- `expectedResponse.body`: structural match; ISO-8601 placeholders (`"<ISO-8601>"`) only check parseability.
- `expectedResponse.bodyBase64`: exact byte-for-byte match after base64 decoding.
- `expectedResponse.bodyAssertions`: list of fine-grained assertions over the returned JSON body (`items-include-path`, `items-exclude-path`, `items-files-equal`, `items-dirs-equal`, `json-field-equals`).
- `expectedResponse.streamAssertions`: assertions over an SSE stream (`expect-event`, etc.).

## Scope

The suite exercises the Store contract surface: read, write, append, delete, rename, exists, stat, list (with all flag combinations), batch commit, path validation, authentication header propagation, and the SSE `ready` + `change` frames. It deliberately does not cover higher-level endpoints like ingest, ask, or brain CRUD; those are driven through the cross-SDK eval runner at [`../../eval`](../../eval).

## Current parity

All three SDKs (TypeScript, Go, Python) pass 28 of the 29 cases.

## Notes

Some reference test cases in `e2e/http-store.contract.test.ts` rely on SDK-side journal materialisation (e.g. `HttpBatch.rename` emitted as write + delete at the server boundary). The JSON here captures the observable wire behaviour; SDKs that implement their own batch materialisation strategy must still produce the same observable server state at commit time.

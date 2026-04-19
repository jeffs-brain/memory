# @jeffs-brain/memory-openfga

OpenFGA authorisation adapter for [`@jeffs-brain/memory`](https://www.npmjs.com/package/@jeffs-brain/memory). Implements the `AccessControlProvider` contract by talking to an OpenFGA server (or any API-compatible proxy) with plain `fetch`. No SDK dependencies, no bundled HTTP client.

## Install

```bash
npm i @jeffs-brain/memory @jeffs-brain/memory-openfga
# or
bun add @jeffs-brain/memory @jeffs-brain/memory-openfga
```

## Usage

```ts
import { withAccessControl } from '@jeffs-brain/memory/acl'
import { createOpenFgaProvider } from '@jeffs-brain/memory-openfga'

const acl = createOpenFgaProvider({
  apiUrl: 'https://fga.example.com',
  storeId: 'store-1',
  modelId: 'model-42',
  token: process.env.OPENFGA_TOKEN,
})

const guarded = withAccessControl(
  store,
  acl,
  { kind: 'user', id: 'alice' },
  { resource: { type: 'brain', id: 'notes' } },
)

// `guarded` is a `Store`. Use it anywhere the unguarded store would go;
// every read/write/delete now runs through OpenFGA first.
```

See [`@jeffs-brain/memory/acl`](https://www.npmjs.com/package/@jeffs-brain/memory) for the provider contract and the in-box RBAC alternative. The shared FGA model lives at [`spec/openfga/schema.fga`](https://github.com/jeffs-brain/memory/blob/main/spec/openfga/schema.fga).

## Lifecycle

`acl.close?.()` is a no-op for this adapter (the `fetch` transport owns no
connection pool). Calling it is still recommended for forward compatibility
when you swap providers or move to a transport that holds real state.

## Feature support

- `AccessControlProvider` wire-compatible with `withAccessControl` and any caller using the contract directly.
- Pure `fetch`; no FGA SDK dependency.
- Works against any OpenFGA-compatible backend (self-hosted or managed).

## Docs

- Repo README: https://github.com/jeffs-brain/memory#readme
- Protocol and storage spec: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)
- Docs site: https://docs.jeffsbrain.com

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

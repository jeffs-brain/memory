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
import { createMemory } from '@jeffs-brain/memory'
import { createOpenFgaProvider } from '@jeffs-brain/memory-openfga'

const acl = createOpenFgaProvider({
  apiUrl: 'https://fga.example.com',
  storeId: 'store-1',
  modelId: 'model-42',
  token: process.env.OPENFGA_TOKEN,
})

const mem = createMemory({ store, provider, embedder, cursorStore, acl, scope: 'project', actorId: 'user:alex' })
```

See `@jeffs-brain/memory/acl` for the provider contract and the in-box RBAC alternative.

## Docs

- Repo README: https://github.com/jeffs-brain/memory#readme
- Protocol and storage spec: [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec)

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

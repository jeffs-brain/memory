# @jeffs-brain/memory-openfga

OpenFGA HTTP adapter for [`@jeffs-brain/memory`](../memory). Implements the
`AccessControlProvider` contract by talking to an OpenFGA server (or any
API-compatible proxy) via plain `fetch`.

## Usage

```ts
import { createOpenFgaProvider } from '@jeffs-brain/memory-openfga'

const acl = createOpenFgaProvider({
  apiUrl: 'https://fga.example.com',
  storeId: 'store-1',
  modelId: 'model-42',
  token: process.env.OPENFGA_TOKEN,
})
```

See `@jeffs-brain/memory/acl` for the provider contract and the in-box RBAC
alternative.

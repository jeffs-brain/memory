# acl_openfga

`httpx`-based adapter that implements `jeffs_brain_memory.acl.Provider`
against an [OpenFGA](https://openfga.dev) HTTP API. Use it from Python
consumers that want production-grade authorisation backed by a tuple store.

The authorisation model the adapter speaks is documented in
[`spec/openfga/schema.fga`](../../../../../spec/openfga/schema.fga).

## Usage

```python
from jeffs_brain_memory import wrap_store, Subject, Resource
from jeffs_brain_memory.acl_openfga import OpenFgaOptions, create_openfga_provider

provider = create_openfga_provider(
    OpenFgaOptions(api_url="https://fga.example.com", store_id="store-1")
)
guarded = wrap_store(
    store, provider, Subject(kind="user", id="alice"),
    resource=Resource(type="brain", id="notes"),
)
```

## Lifecycle

`provider.close()` releases the internal `httpx.AsyncClient` when it was
constructed by the adapter, and is a no-op when a client was supplied by
the caller via `OpenFgaOptions.client` (the caller owns that lifecycle).
`close()` is idempotent.

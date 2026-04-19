# aclopenfga

Pure `net/http` adapter that implements `acl.Provider` against an
[OpenFGA](https://openfga.dev) HTTP API. Use it from Go consumers that
want production-grade authorisation backed by a tuple store.

The authorisation model the adapter speaks lives at
[`spec/openfga/schema.fga`](../../../spec/openfga/schema.fga) and is
shared with the TypeScript and Python OpenFGA adapters.

## Usage

```go
import (
    "os"

    "github.com/jeffs-brain/memory/go/acl"
    "github.com/jeffs-brain/memory/go/aclopenfga"
)

provider, err := aclopenfga.NewProvider(aclopenfga.Options{
    APIURL:  "https://fga.example.com",
    StoreID: "store-1",
    ModelID: "model-42",                 // optional
    Token:   os.Getenv("OPENFGA_TOKEN"), // optional bearer
})
if err != nil { /* handle */ }
defer provider.Close()

guarded := acl.Wrap(
    store,
    provider,
    acl.Subject{Kind: acl.SubjectUser, ID: "alice"},
    acl.WrapOptions{Resource: acl.Resource{Type: acl.ResourceBrain, ID: "notes"}},
)

// `guarded` is a brain.Store. Every Read/Write/Delete runs through
// OpenFGA's check endpoint first; denials surface as *acl.ForbiddenError
// (which satisfies errors.Is(err, brain.ErrForbidden)).
```

The default action -> relation map matches the shared schema:

| Action | Relation     |
|--------|--------------|
| read   | `reader`     |
| write  | `writer`     |
| delete | `can_delete` |
| admin  | `admin`      |
| export | `can_export` |

## Lifecycle

`provider.Close()` is currently a no-op for both caller-supplied and
adapter-constructed `*http.Client` values: the standard library client
shares `http.DefaultTransport`, which is process-wide and must not be
torn down. Call it anyway for forward compatibility; a future transport
that owns real state (long-lived gRPC stream, persistent HTTP/2 pool)
will release it cleanly through the same hook.

## Errors

- `*HTTPError` for non-2xx responses (carries status, endpoint, truncated body).
- `*RequestError` wraps transport and JSON marshalling failures (`Unwrap` returns the cause).
- Denials from `Check` come back as `acl.CheckResult{Allowed: false, Reason: ...}` rather than errors; the `acl.Wrap` helper converts those into `*acl.ForbiddenError` at the Store boundary.

## See also

- [`acl/`](../acl) - the provider contract, in-process RBAC, and `Wrap`.
- [`spec/openfga/`](../../../spec/openfga) - the shared FGA model.
- Docs site: <https://docs.jeffsbrain.com/concepts/acl/>

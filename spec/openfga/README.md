# OpenFGA authorisation model

`schema.fga` is the canonical authorisation model used by every
language SDK's OpenFGA adapter (`@jeffs-brain/memory-openfga`,
`github.com/jeffs-brain/memory/go/aclopenfga`,
`jeffs_brain_memory.acl_openfga`).

Resource hierarchy: `workspace -> brain -> collection -> document`.
Subject kinds: `user`, `api_key`, `service`. Default action -> relation
map (used by every adapter):

| Action | Relation     |
|--------|--------------|
| read   | `reader`     |
| write  | `writer`     |
| delete | `can_delete` |
| admin  | `admin`      |
| export | `can_export` |

Load this model into an OpenFGA store before pointing an adapter at it.

```sh
fga model write --store-id <STORE_ID> --file schema.fga
```

The file lives here (and only here) so the three SDK adapters share a
single source of truth. If you change it, bump the schema version line
and update each adapter's tests if relation names move.

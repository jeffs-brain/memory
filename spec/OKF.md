# OKF content profile

Jeffs Brain stores knowledge as markdown files with YAML frontmatter. The content profile is compatible with Open Knowledge Format v0.1 while keeping the existing Store, HTTP, retrieval, MCP and authorisation contracts unchanged.

## Bundle

A brain root is an OKF bundle candidate. The canonical content areas are:

- `wiki/`: curated articles.
- `memory/`: short memory notes.
- `raw/`: ingested source documents and source mirrors.
- `conversations/`, `episodes/`, `reflections/`: session-derived documents.
- `ontology/`: ontology anchors and typed concepts.

Generated internal files that start with `_` remain supported. Producers that need strict OKF exchange can emit or materialise `index.md` and `log.md` views from those generated files.

## Frontmatter

New concept documents should include these OKF fields:

```yaml
---
type: Article
title: Human-readable title
description: One-line summary
resource: https://example.com/canonical-source
tags: [memory, retrieval]
timestamp: 2026-06-17T00:00:00Z
---
```

`type` is the only required OKF field for concept documents. Consumers must tolerate unknown type values and unknown additional fields.

## Legacy aliases

The SDK normalises existing Jeffs Brain fields onto OKF without rewriting the source file:

| OKF field | Legacy aliases |
| --- | --- |
| `title` | `title`, first markdown `#` heading, `name`, filename stem |
| `description` | `description`, `summary` |
| `timestamp` | `timestamp`, `modified`, `updated_at`, `updatedAt`, `created_at`, `createdAt`, `created` |
| `resource` | `resource`, `url`, `source_url`, `sourceUrl` |
| `tags` | `tags` list or comma-separated scalar |

Consumers may derive an effective type from the path for indexing and presentation, for example `wiki/` -> `Article`, `memory/` -> `Memory`, and `raw/` -> `Raw Document`. That derived value does not make a document strictly OKF-conformant unless the file itself has `type` frontmatter.

## Links

Consumers should treat both forms as graph edges:

- OKF markdown links, for example `[customers](/tables/customers.md)` and `[neighbour](./other.md)`.
- Legacy wikilinks, for example `[[tables/customers]]`.

Broken links are allowed. Lint tools may warn, but readers must still index and retrieve the document.

## Citations

For externally sourced claims, producers should add a `# Citations` section with numbered markdown links. Existing `sources` frontmatter and `## Sources` sections are still accepted as legacy provenance and should be preserved on rewrite.

## Validation levels

Validation is intentionally soft:

- Strict OKF conformance requires parseable frontmatter and a non-empty `type` field on every non-reserved `.md` file.
- Compatibility mode accepts legacy aliases and path-derived presentation metadata.
- Reserved `index.md` and `log.md` are never concept documents. A root `index.md` may declare `okf_version: "0.1"` in frontmatter.

// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"path"
	"strings"
)

// Logical path roots. Every [Path] starts with one of these prefixes.
const (
	memoryRoot       = "memory"
	wikiRoot         = "wiki"
	rawRoot          = "raw"
	rawDocumentsRoot = "raw/documents"
	sourcesRoot      = "raw/.sources"
	schemaFile       = "kb-schema.md"

	// schemaVersionName is the root-level schema version marker. Kept here
	// alongside the other root constants so store path resolution stays in
	// sync — any new root-level file must appear in this block AND in
	// fsstore.relative / logicalFromRel.
	schemaVersionName = "schema-version.yaml"
)

// ValidatePath returns an error if p violates the [Path] conventions:
// forward slashes only, no leading slash, no ".." components, no
// backslashes, non-empty. Callers rarely invoke this directly; helpers
// build valid paths, and [Store] implementations call it at their I/O
// boundaries.
func ValidatePath(p Path) error {
	s := string(p)
	if s == "" {
		return wrapInvalid("empty path")
	}
	if strings.ContainsRune(s, '\\') {
		return wrapInvalid("contains backslash: " + s)
	}
	if strings.ContainsRune(s, 0) {
		return wrapInvalid("contains null byte")
	}
	if strings.HasPrefix(s, "/") {
		return wrapInvalid("leading slash: " + s)
	}
	if strings.HasSuffix(s, "/") {
		return wrapInvalid("trailing slash: " + s)
	}
	// path.Clean collapses "." and ".." if the cleaned form differs the
	// input contained traversal or redundant components we want to reject.
	// The explicit ".." scan catches a cleaned path like "../x" that
	// remains canonical after Clean.
	if cleaned := path.Clean(s); cleaned != s {
		return wrapInvalid("non-canonical: " + s)
	}
	for _, part := range strings.Split(s, "/") {
		if part == ".." {
			return wrapInvalid("contains ..: " + s)
		}
	}
	return nil
}

func wrapInvalid(msg string) error {
	return &pathError{msg: msg}
}

type pathError struct{ msg string }

func (e *pathError) Error() string { return "brain: invalid path: " + e.msg }
func (e *pathError) Unwrap() error { return ErrInvalidPath }

// ---------- Memory ----------

// MemoryGlobalIndex returns the path of the global memory index file.
func MemoryGlobalIndex() Path { return Path(memoryRoot + "/global/MEMORY.md") }

// MemoryGlobalTopic returns the path of a global memory topic file. The
// slug is the bare name without extension; ".md" is appended.
func MemoryGlobalTopic(slug string) Path {
	return Path(path.Join(memoryRoot, "global", slug+".md"))
}

// MemoryGlobalPrefix returns the logical prefix for global memory.
func MemoryGlobalPrefix() Path { return Path(memoryRoot + "/global") }

// MemoryProjectIndex returns the path of a project's memory index file.
func MemoryProjectIndex(projectSlug string) Path {
	return Path(path.Join(memoryRoot, "project", projectSlug, "MEMORY.md"))
}

// MemoryProjectTopic returns the path of a project memory topic file.
func MemoryProjectTopic(projectSlug, slug string) Path {
	return Path(path.Join(memoryRoot, "project", projectSlug, slug+".md"))
}

// MemoryProjectPrefix returns the logical prefix for a project's memory.
func MemoryProjectPrefix(projectSlug string) Path {
	return Path(path.Join(memoryRoot, "project", projectSlug))
}

// MemoryProjectsPrefix returns the logical prefix for all project memory.
func MemoryProjectsPrefix() Path { return Path(memoryRoot + "/project") }

// ---------- Tenant ----------

// MemoryTenantTopic returns the path of a tenant-scoped memory topic file.
func MemoryTenantTopic(tenantSlug, slug string) Path {
	return Path(path.Join(memoryRoot, "tenant", sanitiseScopeID(tenantSlug), slug+".md"))
}

// MemoryTenantCodec returns the path of a tenant's codec file.
func MemoryTenantCodec(tenantSlug string) Path {
	return Path(path.Join("tenants", sanitiseScopeID(tenantSlug), "codec.md"))
}

// ---------- Thread ----------

// MemoryThreadTopic returns the path of a thread-scoped memory topic file.
func MemoryThreadTopic(threadID, slug string) Path {
	return Path(path.Join(memoryRoot, "thread", sanitiseScopeID(threadID), slug+".md"))
}

// MemoryThreadMemoryRoot returns the logical prefix for a thread's memory.
func MemoryThreadMemoryRoot(threadID string) Path {
	return Path(path.Join(memoryRoot, "thread", sanitiseScopeID(threadID)))
}

// ---------- Buffer ----------

// MemoryBufferPrefix returns the logical prefix for all buffer files.
func MemoryBufferPrefix() Path { return Path(memoryRoot + "/buffer") }

// MemoryBufferGlobal returns the path of the global observation buffer.
func MemoryBufferGlobal() Path { return Path(memoryRoot + "/buffer/global.md") }

// MemoryBufferProject returns the path of a project-scoped observation buffer.
func MemoryBufferProject(projectSlug string) Path {
	return Path(path.Join(memoryRoot, "buffer", "project", projectSlug+".md"))
}

// ---------- Wiki ----------

// WikiMasterIndex returns the path of the top-level wiki index.
func WikiMasterIndex() Path { return Path(wikiRoot + "/_index.md") }

// WikiConceptsIndex returns the path of the wiki concept map.
func WikiConceptsIndex() Path { return Path(wikiRoot + "/_concepts.md") }

// WikiHealth returns the path of the wiki health report.
func WikiHealth() Path { return Path(wikiRoot + "/_health.md") }

// WikiLog returns the path of the wiki operations log.
func WikiLog() Path { return Path(wikiRoot + "/_log.md") }

// WikiTopicIndex returns the path of a topic-specific wiki index.
func WikiTopicIndex(topic string) Path {
	return Path(path.Join(wikiRoot, topic, "_index.md"))
}

// WikiArticle returns the path of a wiki article under a topic.
func WikiArticle(topic, article string) Path {
	return Path(path.Join(wikiRoot, topic, article+".md"))
}

// WikiPrefix returns the logical root of the wiki tree.
func WikiPrefix() Path { return Path(wikiRoot) }

// WikiArticleFromRel returns a wiki article path from a wiki-relative path
// like "topic/article.md".
func WikiArticleFromRel(rel string) Path {
	return Path(path.Join(wikiRoot, rel))
}

// ---------- Raw ----------

// RawIngest returns the path of a raw ingest file.
func RawIngest(subdir, name string) Path {
	return Path(path.Join(rawRoot, subdir, name))
}

// RawDocument returns the path of an ingested document file under
// raw/documents. Callers supply the slug without extension; ".md" is
// appended.
func RawDocument(slug string) Path {
	return Path(path.Join(rawDocumentsRoot, slug+".md"))
}

// RawSource returns the sources path for a compiled raw file.
func RawSource(relPath string) Path {
	return Path(path.Join(sourcesRoot, relPath))
}

// RawPrefix returns the logical prefix for all raw ingests.
func RawPrefix() Path { return Path(rawRoot) }

// RawDocumentsPrefix returns the logical prefix where [knowledge.Base.Ingest]
// persists ingested documents. Search indexes this tree so ingested content
// is discoverable via BM25.
func RawDocumentsPrefix() Path { return Path(rawDocumentsRoot) }

// SourcesPrefix returns the logical prefix for compiled source files.
func SourcesPrefix() Path { return Path(sourcesRoot) }

// IsSourceRaw reports whether p lies under the .sources prefix.
func IsSourceRaw(p Path) bool {
	return strings.HasPrefix(string(p), sourcesRoot+"/") || string(p) == sourcesRoot
}

// ---------- Schema ----------

// Schema returns the path of the wiki schema document.
func Schema() Path { return Path(schemaFile) }

// sanitiseScopeID normalises a scope identifier (tenant slug, thread ID)
// to a safe path component: lowercase, only [a-z0-9-_], consecutive
// dashes collapsed, trimmed.
func sanitiseScopeID(raw string) string {
	raw = strings.ToLower(strings.TrimSpace(raw))
	var b strings.Builder
	prevDash := false
	for _, r := range raw {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			b.WriteRune(r)
			prevDash = false
		default:
			if !prevDash && b.Len() > 0 {
				b.WriteByte('-')
				prevDash = true
			}
		}
	}
	s := strings.TrimRight(b.String(), "-")
	if s == "" {
		return "unknown"
	}
	return s
}

// ---------- Introspection ----------

// IsGenerated reports whether the given path is a generated file (prefixed
// with underscore). Used by [Store.List] to filter by default.
func IsGenerated(p Path) bool {
	base := path.Base(string(p))
	return strings.HasPrefix(base, "_")
}

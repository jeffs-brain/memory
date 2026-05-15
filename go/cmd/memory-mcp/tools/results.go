// SPDX-License-Identifier: Apache-2.0

package tools

import "time"

// RememberResult is the typed response from memory_remember.
type RememberResult struct {
	ID         string   `json:"id"`
	Path       string   `json:"path"`
	ByteSize   int      `json:"byte_size"`
	ChunkCount int      `json:"chunk_count"`
	TookMs     int64    `json:"took_ms"`
	BrainID    string   `json:"brain_id"`
	Tags       []string `json:"tags,omitempty"`
}

// SearchHit is a single hit in a search result.
type SearchHit struct {
	Score   float64 `json:"score"`
	Path    string  `json:"path"`
	Title   string  `json:"title"`
	Summary string  `json:"summary"`
	Content string  `json:"content"`
	ChunkID string  `json:"chunk_id"`
	Scope   string  `json:"scope"`
}

// SearchResult is the typed response from memory_search.
type SearchResult struct {
	Query   string      `json:"query"`
	BrainID string      `json:"brain_id"`
	Hits    []SearchHit `json:"hits"`
	TookMs  int64       `json:"took_ms"`
}

// RecallChunk is a single chunk in a recall result.
type RecallChunk struct {
	ChunkID    string  `json:"chunk_id"`
	DocumentID string  `json:"document_id"`
	Score      float64 `json:"score"`
	Path       string  `json:"path"`
	Content    string  `json:"content"`
	Title      string  `json:"title"`
	Summary    string  `json:"summary"`
}

// RecallResult is the typed response from memory_recall.
type RecallResult struct {
	Query     string        `json:"query"`
	BrainID   string        `json:"brain_id"`
	SessionID string        `json:"session_id,omitempty"`
	Chunks    []RecallChunk `json:"chunks"`
}

// AskCitation is a citation reference in an ask result.
type AskCitation struct {
	Type        string `json:"type"`
	ChunkID     string `json:"chunk_id"`
	DocumentID  string `json:"document_id"`
	AnswerStart int    `json:"answer_start"`
	AnswerEnd   int    `json:"answer_end"`
	Quote       string `json:"quote"`
}

// AskRetrievedChunk is a retrieved chunk preview in an ask result.
type AskRetrievedChunk struct {
	ChunkID    string  `json:"chunk_id"`
	DocumentID string  `json:"document_id"`
	Score      float64 `json:"score"`
	Preview    string  `json:"preview"`
}

// AskResult is the typed response from memory_ask.
type AskResult struct {
	Answer    string              `json:"answer"`
	Citations []AskCitation       `json:"citations"`
	Retrieved []AskRetrievedChunk `json:"retrieved"`
}

// IngestResult is the typed response from memory_ingest_file.
type IngestResult struct {
	Status        string `json:"status"`
	DocumentID    string `json:"document_id"`
	Path          string `json:"path"`
	Hash          string `json:"hash"`
	ChunkCount    int    `json:"chunk_count"`
	EmbeddedCount int    `json:"embedded_count"`
	DurationMs    int64  `json:"duration_ms"`
	Reused        bool   `json:"reused"`
}

// IngestURLResult is the typed response from memory_ingest_url.
type IngestURLResult struct {
	Path             string       `json:"path"`
	Result           IngestResult `json:"result"`
	DocumentContent  string       `json:"-"` // Internal field, excluded from JSON.
}

// ExtractedMessage is a single extracted memory in an extract result.
type ExtractedMessage struct {
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
	Role      string `json:"role"`
	Content   string `json:"content"`
	CreatedAt string `json:"created_at"`
}

// ExtractDocument holds the document metadata from transcript-mode extraction.
type ExtractDocument struct {
	ID             string         `json:"id"`
	BrainID        string         `json:"brain_id"`
	Title          string         `json:"title"`
	Path           string         `json:"path"`
	Source         string         `json:"source"`
	ContentType    string         `json:"content_type"`
	ByteSize       int            `json:"byte_size"`
	ChecksumSHA256 string         `json:"checksum_sha256"`
	Metadata       map[string]int `json:"metadata"`
	CreatedAt      string         `json:"created_at"`
	UpdatedAt      string         `json:"updated_at"`
	DeletedAt      *time.Time     `json:"deleted_at"`
}

// ExtractResult is the typed response from memory_extract.
type ExtractResult struct {
	Mode     string             `json:"mode"`
	Messages []ExtractedMessage `json:"messages,omitempty"`
	Document *ExtractDocument   `json:"document,omitempty"`
}

// ExtractedMemory is a single memory in an extract-after-ingest result.
type ExtractedMemory struct {
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// ExtractAfterIngestResult is the typed response from extract-after-ingest.
type ExtractAfterIngestResult struct {
	FactsExtracted int               `json:"factsExtracted"`
	Memories       []ExtractedMemory `json:"memories"`
}

// ReflectionDetail holds the reflection outcome from memory_reflect.
type ReflectionDetail struct {
	Outcome              string `json:"outcome"`
	ShouldRecordEpisode  bool   `json:"should_record_episode"`
	Path                 string `json:"path"`
	RetryFeedback        string `json:"retry_feedback"`
}

// ReflectResult is the typed response from memory_reflect.
type ReflectResult struct {
	ReflectionStatus    string            `json:"reflection_status"`
	Reflection          *ReflectionDetail `json:"reflection,omitempty"`
	ReflectionAttempted bool              `json:"reflection_attempted"`
	EndedAt             string            `json:"ended_at"`
}

// CompileDetail holds the compilation result from memory_consolidate.
type CompileDetail struct {
	Compiled  int   `json:"compiled"`
	Chunks    int   `json:"chunks"`
	Skipped   int   `json:"skipped"`
	Errors    int   `json:"errors"`
	ElapsedMs int64 `json:"elapsed_ms"`
}

// ConsolidateResult is the typed response from memory_consolidate.
type ConsolidateResult struct {
	Result CompileDetail `json:"result"`
}

// BrainInfo is a single brain in a list-brains result.
type BrainInfo struct {
	ID         string `json:"id"`
	Slug       string `json:"slug"`
	Name       string `json:"name"`
	Visibility string `json:"visibility"`
	CreatedAt  string `json:"created_at,omitempty"`
}

// CreateBrainResult is the typed response from memory_create_brain.
type CreateBrainResult = BrainInfo

// ListBrainsResult is the typed response from memory_list_brains.
type ListBrainsResult struct {
	Items []BrainInfo `json:"items"`
}

// batchFileResult is the per-file entry in a batch ingest response.
type batchFileResult struct {
	Path       string `json:"path"`
	Status     string `json:"status"`
	Error      string `json:"error,omitempty"`
	DocumentID string `json:"documentId,omitempty"`
	Hash       string `json:"hash,omitempty"`
}

// batchResult is the typed response from memory_ingest_batch.
type batchResult struct {
	Total     int               `json:"total"`
	Succeeded int64             `json:"succeeded"`
	Failed    int64             `json:"failed"`
	Results   []batchFileResult `json:"results"`
}

// directoryResult is the typed response from memory_ingest_directory.
type directoryResult struct {
	JobGroupID     string            `json:"jobGroupId"`
	FilesQueued    int               `json:"filesQueued"`
	FilesSkipped   int               `json:"filesSkipped"`
	SkippedReasons []string          `json:"skippedReasons"`
	Async          bool              `json:"async"`
	Total          int               `json:"total"`
	Succeeded      int64             `json:"succeeded"`
	Failed         int64             `json:"failed"`
	Skipped        int               `json:"skipped"`
	Results        []batchFileResult `json:"results,omitempty"`
}

// ingestWithExtractionResult combines an ingest result with extraction.
type ingestWithExtractionResult struct {
	Ingest     any                       `json:"ingest"`
	Extraction *ExtractAfterIngestResult `json:"extraction"`
}

// ingestURLWithExtractionResult combines a URL ingest result with extraction.
type ingestURLWithExtractionResult struct {
	Ingest     *IngestURLResult          `json:"ingest"`
	Extraction *ExtractAfterIngestResult `json:"extraction"`
}

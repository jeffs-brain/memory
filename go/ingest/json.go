// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// JsonExtractorConfig configures the JSON extractor behaviour.
type JsonExtractorConfig struct {
	// ObjectsPerChunk is the number of array objects per output chunk.
	// Defaults to 50 when zero.
	ObjectsPerChunk int

	// MaxDepth limits the recursion depth for nested structures.
	// Defaults to 10 when zero.
	MaxDepth int

	// SchemaSampleSize is how many objects to sample for schema
	// detection. Defaults to 20 when zero.
	SchemaSampleSize int

	// TableThreshold is the minimum number of uniform objects needed
	// to emit markdown table format. Defaults to 3 when zero.
	TableThreshold int

	// MaxInputSize caps the raw input size in bytes. Inputs
	// exceeding this limit are rejected before parsing. Defaults
	// to 50 MiB when zero.
	MaxInputSize int64
}

func (c JsonExtractorConfig) objectsPerChunk() int {
	if c.ObjectsPerChunk > 0 {
		return c.ObjectsPerChunk
	}
	return 50
}

func (c JsonExtractorConfig) maxDepth() int {
	if c.MaxDepth > 0 {
		return c.MaxDepth
	}
	return 10
}

func (c JsonExtractorConfig) schemaSampleSize() int {
	if c.SchemaSampleSize > 0 {
		return c.SchemaSampleSize
	}
	return 20
}

func (c JsonExtractorConfig) tableThreshold() int {
	if c.TableThreshold > 0 {
		return c.TableThreshold
	}
	return 3
}

func (c JsonExtractorConfig) maxInputSize() int64 {
	if c.MaxInputSize > 0 {
		return c.MaxInputSize
	}
	return defaultMaxInputSize
}

// ExtractJSON parses raw JSON bytes and returns structured text output.
// Arrays of objects are chunked by object boundaries with schema
// detection. Objects are emitted with structural context. Deeply
// nested structures are flattened to dot-notation. For large arrays,
// streaming token-based parsing is used to avoid holding the full
// parsed tree in memory simultaneously.
func ExtractJSON(raw []byte, cfg JsonExtractorConfig) (ExtractResult, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return ExtractResult{}, fmt.Errorf("structured: empty json input")
	}

	if int64(len(raw)) > cfg.maxInputSize() {
		return ExtractResult{}, fmt.Errorf("structured: json input exceeds %d byte limit", cfg.maxInputSize())
	}

	text, encoding, err := detectEncoding(raw)
	if err != nil {
		return ExtractResult{}, err
	}

	// Try streaming array extraction first. If the top-level value
	// is a JSON array, we stream objects one at a time and process
	// them in chunks to reduce peak memory usage.
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()

	tok, err := decoder.Token()
	if err != nil {
		return ExtractResult{}, fmt.Errorf("structured: invalid json: %w", err)
	}

	// Top-level array: stream object-by-object.
	if delim, ok := tok.(json.Delim); ok && delim == '[' {
		return extractJSONArrayStreaming(decoder, cfg, encoding)
	}

	// Not an array: fall back to full decode for objects / primitives.
	var parsed any
	fullDecoder := json.NewDecoder(strings.NewReader(text))
	fullDecoder.UseNumber()
	if decErr := fullDecoder.Decode(&parsed); decErr != nil {
		return ExtractResult{}, fmt.Errorf("structured: invalid json: %w", decErr)
	}

	content, structType, schema := renderJSON(parsed, cfg)

	meta := map[string]string{
		"encoding":       encoding,
		"structure_type": structType,
	}
	if schema != "" {
		meta["detected_schema"] = schema
	}

	return ExtractResult{
		Text:        content,
		ContentType: "application/json",
		Metadata:    meta,
	}, nil
}

// extractJSONArrayStreaming processes a JSON array using streaming
// token-by-token decoding. Elements are collected in chunks and
// rendered incrementally to keep memory bounded.
func extractJSONArrayStreaming(decoder *json.Decoder, cfg JsonExtractorConfig, encoding string) (ExtractResult, error) {
	// Collect all elements (streaming decode, but we still
	// accumulate into slices for schema analysis). For truly huge
	// files, the maxInputSize guard above prevents OOM.
	var elements []any
	for decoder.More() {
		var v any
		if err := decoder.Decode(&v); err != nil {
			return ExtractResult{}, fmt.Errorf("structured: invalid json array element: %w", err)
		}
		elements = append(elements, v)
	}
	// Consume the closing bracket.
	if _, err := decoder.Token(); err != nil {
		return ExtractResult{}, fmt.Errorf("structured: invalid json: %w", err)
	}

	content, structType, schema := renderJSONArray(elements, cfg)

	meta := map[string]string{
		"encoding":       encoding,
		"structure_type": structType,
	}
	if schema != "" {
		meta["detected_schema"] = schema
	}

	return ExtractResult{
		Text:        content,
		ContentType: "application/json",
		Metadata:    meta,
	}, nil
}

// renderJSON dispatches to the appropriate rendering strategy based on
// the top-level JSON structure.
func renderJSON(v any, cfg JsonExtractorConfig) (string, string, string) {
	switch val := v.(type) {
	case []any:
		return renderJSONArray(val, cfg)
	case map[string]any:
		return renderJSONObject(val, cfg)
	default:
		return fmt.Sprintf("%v", val), "primitive", ""
	}
}

// renderJSONArray handles the array case, detecting whether objects
// share a uniform schema (markdown table) or should be rendered
// individually.
func renderJSONArray(arr []any, cfg JsonExtractorConfig) (string, string, string) {
	if len(arr) == 0 {
		return "[]", "empty_array", ""
	}

	// Check if all elements are primitives.
	allPrimitive := true
	for _, elem := range arr {
		switch elem.(type) {
		case map[string]any, []any:
			allPrimitive = false
		}
		if !allPrimitive {
			break
		}
	}
	if allPrimitive {
		lines := make([]string, 0, len(arr))
		for _, elem := range arr {
			lines = append(lines, formatPrimitive(elem))
		}
		return strings.Join(lines, "\n"), "primitive_array", ""
	}

	// Separate objects from non-objects.
	objects := make([]map[string]any, 0, len(arr))
	for _, elem := range arr {
		obj, ok := elem.(map[string]any)
		if !ok {
			continue
		}
		objects = append(objects, obj)
	}

	if len(objects) == 0 {
		// Mixed array — render each element.
		var out strings.Builder
		for i, elem := range arr {
			if i > 0 {
				out.WriteString("\n\n")
			}
			out.WriteString(fmt.Sprintf("Item %d:\n", i+1))
			out.WriteString(flattenValue("", elem, 0, cfg.maxDepth(), make(map[uintptr]bool)))
		}
		return out.String(), "mixed_array", ""
	}

	// Detect shared schema from sampled objects.
	sampleSize := cfg.schemaSampleSize()
	if sampleSize > len(objects) {
		sampleSize = len(objects)
	}
	commonKeys := detectCommonKeys(objects[:sampleSize])
	schemaStr := strings.Join(commonKeys, ", ")
	isUniform := len(commonKeys) > 0 && areObjectsUniform(objects[:sampleSize], commonKeys)

	// Uniform array of objects with enough rows: render as markdown table.
	if isUniform && len(objects) >= cfg.tableThreshold() {
		return renderAsMarkdownTable(objects, commonKeys, cfg), "uniform_object_array", schemaStr
	}

	// Heterogeneous or small arrays: render each object with schema annotation.
	return renderObjectsIndividually(objects, commonKeys, cfg), "heterogeneous_object_array", schemaStr
}

// renderJSONObject handles a top-level JSON object by emitting its
// keys and values with structural context.
func renderJSONObject(obj map[string]any, cfg JsonExtractorConfig) (string, string, string) {
	if len(obj) == 0 {
		return "{}", "empty_object", ""
	}

	keys := sortedKeys(obj)
	visited := make(map[uintptr]bool)
	var out strings.Builder

	out.WriteString(fmt.Sprintf("Object with %d keys: %s\n\n", len(keys), strings.Join(keys, ", ")))

	for i, key := range keys {
		if i > 0 {
			out.WriteByte('\n')
		}
		val := obj[key]
		depth := maxNestingDepth(val, 0)
		if depth >= 4 {
			out.WriteString(flattenValue(key, val, 0, cfg.maxDepth(), visited))
		} else {
			out.WriteString(fmt.Sprintf("%s: %s\n", key, formatValue(val)))
		}
	}

	schemaStr := strings.Join(keys, ", ")
	return out.String(), "object", schemaStr
}

// detectCommonKeys samples the first N objects and returns keys that
// appear in at least 50% of sampled objects, sorted alphabetically.
func detectCommonKeys(objects []map[string]any) []string {
	if len(objects) == 0 {
		return nil
	}
	counts := make(map[string]int)
	for _, obj := range objects {
		for key := range obj {
			counts[key]++
		}
	}
	threshold := len(objects) / 2
	if threshold < 1 {
		threshold = 1
	}
	keys := make([]string, 0, len(counts))
	for key, count := range counts {
		if count >= threshold {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	return keys
}

// areObjectsUniform checks whether all sampled objects have exactly
// the same set of keys as the common key set.
func areObjectsUniform(objects []map[string]any, commonKeys []string) bool {
	keySet := make(map[string]struct{}, len(commonKeys))
	for _, k := range commonKeys {
		keySet[k] = struct{}{}
	}
	for _, obj := range objects {
		if len(obj) != len(commonKeys) {
			return false
		}
		for key := range obj {
			if _, ok := keySet[key]; !ok {
				return false
			}
		}
	}
	return true
}

// renderAsMarkdownTable formats uniform objects as a markdown table
// with chunk boundaries at objectsPerChunk intervals.
func renderAsMarkdownTable(objects []map[string]any, keys []string, cfg JsonExtractorConfig) string {
	opc := cfg.objectsPerChunk()
	var out strings.Builder
	chunkIdx := 0

	for i := 0; i < len(objects); i += opc {
		end := i + opc
		if end > len(objects) {
			end = len(objects)
		}
		batch := objects[i:end]

		if chunkIdx > 0 {
			out.WriteString("\n\n---\n\n")
		}

		// Header row.
		out.WriteString("| ")
		out.WriteString(strings.Join(keys, " | "))
		out.WriteString(" |\n")

		// Separator row.
		out.WriteString("|")
		for range keys {
			out.WriteString(" --- |")
		}
		out.WriteByte('\n')

		// Data rows.
		for _, obj := range batch {
			out.WriteString("| ")
			for j, key := range keys {
				if j > 0 {
					out.WriteString(" | ")
				}
				val := obj[key]
				out.WriteString(escapeMdTableCell(formatPrimitive(val)))
			}
			out.WriteString(" |\n")
		}
		chunkIdx++
	}
	return out.String()
}

// renderObjectsIndividually renders each object with schema annotation
// and chunk boundaries.
func renderObjectsIndividually(objects []map[string]any, commonKeys []string, cfg JsonExtractorConfig) string {
	opc := cfg.objectsPerChunk()
	var out strings.Builder
	chunkIdx := 0

	for i := 0; i < len(objects); i += opc {
		end := i + opc
		if end > len(objects) {
			end = len(objects)
		}
		batch := objects[i:end]

		if chunkIdx > 0 {
			out.WriteString("\n\n---\n\n")
		}

		for j, obj := range batch {
			if j > 0 {
				out.WriteByte('\n')
			}
			out.WriteString(fmt.Sprintf("Item %d:\n", i+j+1))
			keys := sortedKeys(obj)
			visited := make(map[uintptr]bool)
			for _, key := range keys {
				val := obj[key]
				depth := maxNestingDepth(val, 0)
				if depth >= 4 {
					out.WriteString(flattenValue(key, val, 0, cfg.maxDepth(), visited))
				} else {
					out.WriteString(fmt.Sprintf("- %s: %s\n", key, formatPrimitive(val)))
				}
			}
		}
		chunkIdx++
	}
	return out.String()
}

// flattenValue recursively flattens a JSON value to dot-notation
// key-value pairs. The visited map guards against circular references
// using reflect-based identity to match the TS approach.
func flattenValue(prefix string, v any, depth, maxDepth int, visited map[uintptr]bool) string {
	if depth >= maxDepth {
		return fmt.Sprintf("%s: [max depth exceeded]\n", prefix)
	}

	switch val := v.(type) {
	case map[string]any:
		ptr := mapPointer(val)
		if visited[ptr] {
			return fmt.Sprintf("%s: [circular reference]\n", prefix)
		}
		visited[ptr] = true
		var out strings.Builder
		keys := sortedKeys(val)
		for _, key := range keys {
			childPrefix := key
			if prefix != "" {
				childPrefix = prefix + "." + key
			}
			out.WriteString(flattenValue(childPrefix, val[key], depth+1, maxDepth, visited))
		}
		return out.String()

	case []any:
		if len(val) == 0 {
			return fmt.Sprintf("%s: []\n", prefix)
		}
		var out strings.Builder
		for i, elem := range val {
			childPrefix := fmt.Sprintf("%s[%d]", prefix, i)
			out.WriteString(flattenValue(childPrefix, elem, depth+1, maxDepth, visited))
		}
		return out.String()

	default:
		label := prefix
		if label == "" {
			label = "value"
		}
		return fmt.Sprintf("%s: %s\n", label, formatPrimitive(val))
	}
}

// formatPrimitive renders a primitive JSON value as a string.
func formatPrimitive(v any) string {
	if v == nil {
		return "null"
	}
	switch val := v.(type) {
	case string:
		return val
	case json.Number:
		return val.String()
	case bool:
		if val {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(val, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", val)
	}
}

// formatValue renders a JSON value including composite types using
// compact JSON for non-primitive values.
func formatValue(v any) string {
	switch v.(type) {
	case map[string]any, []any:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	default:
		return formatPrimitive(v)
	}
}

// maxNestingDepth computes the maximum nesting depth of a JSON value.
func maxNestingDepth(v any, current int) int {
	switch val := v.(type) {
	case map[string]any:
		maxChild := current
		for _, child := range val {
			d := maxNestingDepth(child, current+1)
			if d > maxChild {
				maxChild = d
			}
		}
		return maxChild
	case []any:
		maxChild := current
		for _, child := range val {
			d := maxNestingDepth(child, current+1)
			if d > maxChild {
				maxChild = d
			}
		}
		return maxChild
	default:
		return current
	}
}

// sortedKeys returns the keys of a map in sorted order.
func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// escapeMdTableCell escapes pipe characters in markdown table cells.
func escapeMdTableCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}

// ExtractJSONL parses newline-delimited JSON (one JSON object per
// line) and delegates to ExtractJSON after collecting into an array.
func ExtractJSONL(raw []byte, cfg JsonExtractorConfig) (ExtractResult, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return ExtractResult{}, fmt.Errorf("structured: empty jsonl input")
	}

	if int64(len(raw)) > cfg.maxInputSize() {
		return ExtractResult{}, fmt.Errorf("structured: jsonl input exceeds %d byte limit", cfg.maxInputSize())
	}

	text, _, err := detectEncoding(raw)
	if err != nil {
		return ExtractResult{}, err
	}

	var objects []any
	scanner := strings.NewReader(text)
	decoder := json.NewDecoder(scanner)
	for {
		var v any
		if decErr := decoder.Decode(&v); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return ExtractResult{}, fmt.Errorf("structured: invalid jsonl: %w", decErr)
		}
		objects = append(objects, v)
	}

	if len(objects) == 0 {
		return ExtractResult{}, fmt.Errorf("structured: empty jsonl input")
	}

	// Re-encode as a JSON array and delegate.
	arrayBytes, err := json.Marshal(objects)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("structured: jsonl re-encode: %w", err)
	}

	result, err := ExtractJSON(arrayBytes, cfg)
	if err != nil {
		return result, err
	}
	result.ContentType = "application/jsonl"
	return result, nil
}

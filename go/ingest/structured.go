// SPDX-License-Identifier: Apache-2.0

// Package ingest provides content extraction and chunking for the
// ingestion pipeline. The structured extractors handle CSV, TSV,
// and JSON documents with schema-aware chunking that preserves
// record boundaries and column context.
package ingest

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// -------------------------------------------------------------------
// Shared types
// -------------------------------------------------------------------

// ExtractionResult holds the text output plus metadata from a
// structured data extraction pass.
type ExtractionResult struct {
	Content  string
	MIME     string
	Metadata map[string]string
}

// -------------------------------------------------------------------
// Encoding detection
// -------------------------------------------------------------------

// detectEncoding inspects the leading bytes of raw for a BOM marker.
// Returns the decoded string, the encoding name, and any error.
func detectEncoding(raw []byte) (string, string, error) {
	// UTF-8 BOM (EF BB BF).
	if len(raw) >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF {
		stripped := raw[3:]
		if !utf8.Valid(stripped) {
			return "", "", fmt.Errorf("structured: invalid UTF-8 after BOM")
		}
		return string(stripped), "utf-8-bom", nil
	}

	// UTF-16 LE BOM (FF FE).
	if len(raw) >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
		decoded, err := decodeUTF16LE(raw[2:])
		if err != nil {
			return "", "", fmt.Errorf("structured: UTF-16 LE decode: %w", err)
		}
		return decoded, "utf-16-le", nil
	}

	// UTF-16 BE BOM (FE FF).
	if len(raw) >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
		decoded, err := decodeUTF16BE(raw[2:])
		if err != nil {
			return "", "", fmt.Errorf("structured: UTF-16 BE decode: %w", err)
		}
		return decoded, "utf-16-be", nil
	}

	// Try UTF-8.
	if utf8.Valid(raw) {
		return string(raw), "utf-8", nil
	}

	// Fallback: Latin-1 (ISO-8859-1) — every byte is valid.
	var b strings.Builder
	b.Grow(len(raw))
	for _, by := range raw {
		b.WriteRune(rune(by))
	}
	return b.String(), "latin-1", nil
}

// decodeUTF16LE decodes UTF-16 little-endian bytes to a Go string.
func decodeUTF16LE(data []byte) (string, error) {
	if len(data)%2 != 0 {
		return "", fmt.Errorf("odd byte count for UTF-16")
	}
	var b strings.Builder
	b.Grow(len(data) / 2)
	for i := 0; i+1 < len(data); i += 2 {
		cp := rune(data[i]) | rune(data[i+1])<<8
		b.WriteRune(cp)
	}
	return b.String(), nil
}

// decodeUTF16BE decodes UTF-16 big-endian bytes to a Go string.
func decodeUTF16BE(data []byte) (string, error) {
	if len(data)%2 != 0 {
		return "", fmt.Errorf("odd byte count for UTF-16")
	}
	var b strings.Builder
	b.Grow(len(data) / 2)
	for i := 0; i+1 < len(data); i += 2 {
		cp := rune(data[i])<<8 | rune(data[i+1])
		b.WriteRune(cp)
	}
	return b.String(), nil
}

// -------------------------------------------------------------------
// CSV Extractor
// -------------------------------------------------------------------

// CsvExtractorConfig configures the CSV extractor behaviour.
type CsvExtractorConfig struct {
	// RowsPerChunk is the number of data rows per output chunk.
	// Defaults to 50 when zero.
	RowsPerChunk int

	// MaxRows caps the total number of data rows processed.
	// Defaults to 100000 when zero.
	MaxRows int

	// ForceDelimiter overrides auto-detection when set to a non-zero
	// rune. Common values: ',', ';', '\t', '|'.
	ForceDelimiter rune
}

func (c CsvExtractorConfig) rowsPerChunk() int {
	if c.RowsPerChunk > 0 {
		return c.RowsPerChunk
	}
	return 50
}

func (c CsvExtractorConfig) maxRows() int {
	if c.MaxRows > 0 {
		return c.MaxRows
	}
	return 100000
}

// ExtractCSV parses raw CSV (or TSV) bytes and returns schema-enriched
// text output. Each chunk preserves column headers so that downstream
// search sees self-contained context per row group.
func ExtractCSV(raw []byte, cfg CsvExtractorConfig) (ExtractionResult, error) {
	if len(raw) == 0 {
		return ExtractionResult{}, fmt.Errorf("structured: empty csv file")
	}

	text, encoding, err := detectEncoding(raw)
	if err != nil {
		return ExtractionResult{}, err
	}

	delimiter := cfg.ForceDelimiter
	if delimiter == 0 {
		delimiter = detectDelimiter(text)
	}

	reader := csv.NewReader(strings.NewReader(text))
	reader.Comma = delimiter
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1 // allow ragged rows

	allRows, err := reader.ReadAll()
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("structured: csv parse error: %w", err)
	}
	if len(allRows) == 0 {
		return ExtractionResult{}, fmt.Errorf("structured: empty csv file")
	}

	headers, dataRows := splitHeaders(allRows)
	if len(dataRows) > cfg.maxRows() {
		dataRows = dataRows[:cfg.maxRows()]
	}

	rpc := cfg.rowsPerChunk()
	var out strings.Builder
	chunkCount := 0

	for i := 0; i < len(dataRows); i += rpc {
		end := i + rpc
		if end > len(dataRows) {
			end = len(dataRows)
		}
		batch := dataRows[i:end]

		if chunkCount > 0 {
			out.WriteString("\n\n---\n\n")
		}

		for j, row := range batch {
			rowNum := i + j + 1
			out.WriteString(fmt.Sprintf("Row %d:\n", rowNum))
			for k, header := range headers {
				val := ""
				if k < len(row) {
					val = row[k]
				}
				out.WriteString(fmt.Sprintf("- %s: %s\n", header, val))
			}
			if j < len(batch)-1 {
				out.WriteByte('\n')
			}
		}
		chunkCount++
	}

	columnCount := len(headers)
	rowCount := len(dataRows)
	mime := "text/csv"
	if delimiter == '\t' {
		mime = "text/tab-separated-values"
	}

	return ExtractionResult{
		Content: out.String(),
		MIME:    mime,
		Metadata: map[string]string{
			"encoding":     encoding,
			"delimiter":    string(delimiter),
			"column_count": strconv.Itoa(columnCount),
			"row_count":    strconv.Itoa(rowCount),
			"chunk_count":  strconv.Itoa(chunkCount),
		},
	}, nil
}

// detectDelimiter tries comma, semicolon, tab, and pipe on the first
// 10 lines and picks the delimiter producing the most consistent
// column count.
func detectDelimiter(text string) rune {
	candidates := []rune{',', ';', '\t', '|'}
	lines := firstNLines(text, 10)
	if len(lines) == 0 {
		return ','
	}

	type scored struct {
		delim rune
		score float64
	}
	scores := make([]scored, 0, len(candidates))

	for _, d := range candidates {
		counts := make([]int, 0, len(lines))
		for _, line := range lines {
			n := strings.Count(line, string(d)) + 1
			counts = append(counts, n)
		}
		if counts[0] <= 1 {
			scores = append(scores, scored{d, 0})
			continue
		}
		// Score = column count * consistency. Consistency = 1 - stddev/mean.
		mean := float64(0)
		for _, c := range counts {
			mean += float64(c)
		}
		mean /= float64(len(counts))
		variance := float64(0)
		for _, c := range counts {
			diff := float64(c) - mean
			variance += diff * diff
		}
		variance /= float64(len(counts))
		stddev := math.Sqrt(variance)
		consistency := 1.0
		if mean > 0 {
			consistency = 1.0 - stddev/mean
		}
		scores = append(scores, scored{d, mean * consistency})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	if len(scores) > 0 && scores[0].score > 0 {
		return scores[0].delim
	}
	return ','
}

// splitHeaders separates the header row from data rows. The first row
// is treated as headers when all values are non-empty, none is purely
// numeric, and all values are unique. Otherwise synthetic headers
// (Column_1, Column_2, ...) are generated.
func splitHeaders(rows [][]string) ([]string, [][]string) {
	if len(rows) == 0 {
		return nil, nil
	}
	first := rows[0]
	if looksLikeHeaders(first) {
		return first, rows[1:]
	}
	headers := make([]string, len(first))
	for i := range first {
		headers[i] = fmt.Sprintf("Column_%d", i+1)
	}
	return headers, rows
}

// looksLikeHeaders returns true when all values in the row are
// non-empty, none is purely numeric, and all are unique.
func looksLikeHeaders(row []string) bool {
	if len(row) == 0 {
		return false
	}
	seen := make(map[string]struct{}, len(row))
	for _, v := range row {
		v = strings.TrimSpace(v)
		if v == "" {
			return false
		}
		if isNumeric(v) {
			return false
		}
		lower := strings.ToLower(v)
		if _, dup := seen[lower]; dup {
			return false
		}
		seen[lower] = struct{}{}
	}
	return true
}

// isNumeric reports whether s parses as a number.
func isNumeric(s string) bool {
	_, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	return err == nil
}

// firstNLines returns up to n non-empty lines from text.
func firstNLines(text string, n int) []string {
	lines := strings.Split(text, "\n")
	out := make([]string, 0, n)
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		out = append(out, line)
		if len(out) >= n {
			break
		}
	}
	return out
}

// -------------------------------------------------------------------
// JSON Extractor
// -------------------------------------------------------------------

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

// ExtractJSON parses raw JSON bytes and returns structured text output.
// Arrays of objects are chunked by object boundaries with schema
// detection. Objects are emitted with structural context. Deeply
// nested structures are flattened to dot-notation.
func ExtractJSON(raw []byte, cfg JsonExtractorConfig) (ExtractionResult, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return ExtractionResult{}, fmt.Errorf("structured: empty json input")
	}

	text, encoding, err := detectEncoding(raw)
	if err != nil {
		return ExtractionResult{}, err
	}

	var parsed interface{}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if decErr := decoder.Decode(&parsed); decErr != nil {
		return ExtractionResult{}, fmt.Errorf("structured: invalid json: %w", decErr)
	}

	content, structType, schema := renderJSON(parsed, cfg)

	meta := map[string]string{
		"encoding":       encoding,
		"structure_type": structType,
	}
	if schema != "" {
		meta["detected_schema"] = schema
	}

	return ExtractionResult{
		Content:  content,
		MIME:     "application/json",
		Metadata: meta,
	}, nil
}

// renderJSON dispatches to the appropriate rendering strategy based on
// the top-level JSON structure.
func renderJSON(v interface{}, cfg JsonExtractorConfig) (string, string, string) {
	switch val := v.(type) {
	case []interface{}:
		return renderJSONArray(val, cfg)
	case map[string]interface{}:
		return renderJSONObject(val, cfg)
	default:
		return fmt.Sprintf("%v", val), "primitive", ""
	}
}

// renderJSONArray handles the array case, detecting whether objects
// share a uniform schema (markdown table) or should be rendered
// individually.
func renderJSONArray(arr []interface{}, cfg JsonExtractorConfig) (string, string, string) {
	if len(arr) == 0 {
		return "[]", "empty_array", ""
	}

	// Check if all elements are primitives.
	allPrimitive := true
	for _, elem := range arr {
		switch elem.(type) {
		case map[string]interface{}, []interface{}:
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
	objects := make([]map[string]interface{}, 0, len(arr))
	for _, elem := range arr {
		obj, ok := elem.(map[string]interface{})
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
			out.WriteString(flattenValue("", elem, 0, cfg.maxDepth(), make(map[interface{}]bool)))
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
func renderJSONObject(obj map[string]interface{}, cfg JsonExtractorConfig) (string, string, string) {
	if len(obj) == 0 {
		return "{}", "empty_object", ""
	}

	keys := sortedKeys(obj)
	visited := make(map[interface{}]bool)
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
func detectCommonKeys(objects []map[string]interface{}) []string {
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
func areObjectsUniform(objects []map[string]interface{}, commonKeys []string) bool {
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
func renderAsMarkdownTable(objects []map[string]interface{}, keys []string, cfg JsonExtractorConfig) string {
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
func renderObjectsIndividually(objects []map[string]interface{}, commonKeys []string, cfg JsonExtractorConfig) string {
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
			visited := make(map[interface{}]bool)
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
// key-value pairs. The visited map guards against circular references.
func flattenValue(prefix string, v interface{}, depth, maxDepth int, visited map[interface{}]bool) string {
	if depth >= maxDepth {
		return fmt.Sprintf("%s: [max depth exceeded]\n", prefix)
	}

	switch val := v.(type) {
	case map[string]interface{}:
		if visited[&val] {
			return fmt.Sprintf("%s: [circular reference]\n", prefix)
		}
		visited[&val] = true
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

	case []interface{}:
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
func formatPrimitive(v interface{}) string {
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
func formatValue(v interface{}) string {
	switch v.(type) {
	case map[string]interface{}, []interface{}:
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
func maxNestingDepth(v interface{}, current int) int {
	switch val := v.(type) {
	case map[string]interface{}:
		maxChild := current
		for _, child := range val {
			d := maxNestingDepth(child, current+1)
			if d > maxChild {
				maxChild = d
			}
		}
		return maxChild
	case []interface{}:
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
func sortedKeys(m map[string]interface{}) []string {
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

// -------------------------------------------------------------------
// JSONL support
// -------------------------------------------------------------------

// ExtractJSONL parses newline-delimited JSON (one JSON object per
// line) and delegates to ExtractJSON after collecting into an array.
func ExtractJSONL(raw []byte, cfg JsonExtractorConfig) (ExtractionResult, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return ExtractionResult{}, fmt.Errorf("structured: empty jsonl input")
	}

	text, _, err := detectEncoding(raw)
	if err != nil {
		return ExtractionResult{}, err
	}

	var objects []interface{}
	scanner := strings.NewReader(text)
	decoder := json.NewDecoder(scanner)
	for {
		var v interface{}
		if decErr := decoder.Decode(&v); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return ExtractionResult{}, fmt.Errorf("structured: invalid jsonl: %w", decErr)
		}
		objects = append(objects, v)
	}

	if len(objects) == 0 {
		return ExtractionResult{}, fmt.Errorf("structured: empty jsonl input")
	}

	// Re-encode as a JSON array and delegate.
	arrayBytes, err := json.Marshal(objects)
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("structured: jsonl re-encode: %w", err)
	}

	result, err := ExtractJSON(arrayBytes, cfg)
	if err != nil {
		return result, err
	}
	result.MIME = "application/jsonl"
	return result, nil
}

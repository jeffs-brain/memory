// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"encoding/csv"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// CsvExtractorConfig configures the CSV extractor behaviour.
type CsvExtractorConfig struct {
	// RowsPerChunk is the number of data rows per output chunk.
	// Defaults to 50 when zero.
	RowsPerChunk int

	// MaxRows caps the total number of data rows processed.
	// Defaults to 100000 when zero.
	MaxRows int

	// MaxInputSize caps the raw input size in bytes. Inputs
	// exceeding this limit are rejected before parsing. Defaults
	// to 50 MiB when zero.
	MaxInputSize int64

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

func (c CsvExtractorConfig) maxInputSize() int64 {
	if c.MaxInputSize > 0 {
		return c.MaxInputSize
	}
	return defaultMaxInputSize
}

// formulaPrefixes lists characters that spreadsheet applications
// interpret as formula triggers when a cell value begins with them.
// Values starting with these characters are escaped with a leading
// single-quote to prevent CSV injection.
var formulaPrefixes = [...]byte{'=', '+', '-', '@'}

// sanitiseCSVValue escapes values that could trigger formula injection
// in spreadsheet applications. A leading single-quote is prepended to
// any value starting with a formula trigger character.
func sanitiseCSVValue(s string) string {
	if len(s) == 0 {
		return s
	}
	first := s[0]
	for _, prefix := range formulaPrefixes {
		if first == prefix {
			return "'" + s
		}
	}
	return s
}

// ExtractCSV parses raw CSV (or TSV) bytes and returns schema-enriched
// text output. Each chunk preserves column headers so that downstream
// search sees self-contained context per row group. Rows are processed
// in a streaming fashion to avoid loading the entire file at once.
func ExtractCSV(raw []byte, cfg CsvExtractorConfig) (ExtractResult, error) {
	if len(raw) == 0 {
		return ExtractResult{}, fmt.Errorf("structured: empty csv file")
	}

	if int64(len(raw)) > cfg.maxInputSize() {
		return ExtractResult{}, fmt.Errorf("structured: csv input exceeds %d byte limit", cfg.maxInputSize())
	}

	text, encoding, err := detectEncoding(raw)
	if err != nil {
		return ExtractResult{}, err
	}

	delimiter := cfg.ForceDelimiter
	if delimiter == 0 {
		delimiter = detectDelimiter(text)
	}

	reader := csv.NewReader(strings.NewReader(text))
	reader.Comma = delimiter
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1 // allow ragged rows

	// Stream rows one at a time instead of ReadAll.
	firstRow, err := reader.Read()
	if err != nil {
		return ExtractResult{}, fmt.Errorf("structured: csv parse error: %w", err)
	}

	hasHeaders := looksLikeHeaders(firstRow)
	var headers []string
	var pendingDataRow []string

	if hasHeaders {
		headers = firstRow
	} else {
		headers = make([]string, len(firstRow))
		for i := range firstRow {
			headers[i] = fmt.Sprintf("Column_%d", i+1)
		}
		pendingDataRow = firstRow
	}

	rpc := cfg.rowsPerChunk()
	maxR := cfg.maxRows()
	var out strings.Builder
	chunkCount := 0
	rowCount := 0
	batchRows := make([][]string, 0, rpc)

	// Flush the pending first data row if not a header.
	if pendingDataRow != nil {
		batchRows = append(batchRows, pendingDataRow)
		rowCount++
	}

	flushBatch := func(startIdx int) {
		if len(batchRows) == 0 {
			return
		}
		if chunkCount > 0 {
			out.WriteString("\n\n---\n\n")
		}
		for j, row := range batchRows {
			rowNum := startIdx + j + 1
			out.WriteString(fmt.Sprintf("Row %d:\n", rowNum))
			for k, header := range headers {
				val := ""
				if k < len(row) {
					val = sanitiseCSVValue(row[k])
				}
				out.WriteString(fmt.Sprintf("- %s: %s\n", header, val))
			}
			if j < len(batchRows)-1 {
				out.WriteByte('\n')
			}
		}
		chunkCount++
	}

	batchStartIdx := 0
	for {
		if rowCount >= maxR {
			break
		}
		row, readErr := reader.Read()
		if readErr != nil {
			break
		}
		batchRows = append(batchRows, row)
		rowCount++

		if len(batchRows) >= rpc {
			flushBatch(batchStartIdx)
			batchStartIdx += len(batchRows)
			batchRows = batchRows[:0]
		}
	}
	flushBatch(batchStartIdx)

	columnCount := len(headers)
	mime := "text/csv"
	if delimiter == '\t' {
		mime = "text/tab-separated-values"
	}

	return ExtractResult{
		Text:        out.String(),
		ContentType: mime,
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

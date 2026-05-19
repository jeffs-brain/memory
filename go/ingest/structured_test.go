// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"fmt"
	"strings"
	"testing"
)

// -------------------------------------------------------------------
// CSV Extractor Tests
// -------------------------------------------------------------------

func TestExtractCSV_BasicWithHeaders(t *testing.T) {
	input := []byte("Name,Age,City\nAlice,30,London\nBob,25,Paris\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "- Name: Alice") {
		t.Errorf("expected schema-enriched output with '- Name: Alice', got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "- Age: 30") {
		t.Errorf("expected '- Age: 30' in output")
	}
	if !strings.Contains(result.Text, "- City: London") {
		t.Errorf("expected '- City: London' in output")
	}
	if result.Metadata["column_count"] != "3" {
		t.Errorf("expected column_count=3, got %s", result.Metadata["column_count"])
	}
	if result.Metadata["row_count"] != "2" {
		t.Errorf("expected row_count=2, got %s", result.Metadata["row_count"])
	}
}

func TestExtractCSV_SemicolonDelimiter(t *testing.T) {
	input := []byte("Name;Age;City\nAlice;30;London\nBob;25;Paris\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["delimiter"] != ";" {
		t.Errorf("expected delimiter=';', got %q", result.Metadata["delimiter"])
	}
	if !strings.Contains(result.Text, "- Name: Alice") {
		t.Errorf("expected schema-enriched output, got:\n%s", result.Text)
	}
}

func TestExtractCSV_TabDelimiter(t *testing.T) {
	input := []byte("Name\tAge\tCity\nAlice\t30\tLondon\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["delimiter"] != "\t" {
		t.Errorf("expected tab delimiter, got %q", result.Metadata["delimiter"])
	}
	if result.ContentType != "text/tab-separated-values" {
		t.Errorf("expected MIME text/tab-separated-values, got %s", result.ContentType)
	}
}

func TestExtractCSV_PipeDelimiter(t *testing.T) {
	input := []byte("Name|Age|City\nAlice|30|London\nBob|25|Paris\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["delimiter"] != "|" {
		t.Errorf("expected pipe delimiter, got %q", result.Metadata["delimiter"])
	}
}

func TestExtractCSV_NoHeaders(t *testing.T) {
	input := []byte("1,2,3\n4,5,6\n7,8,9\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "Column_1") {
		t.Errorf("expected synthetic headers with Column_1, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "Column_2") {
		t.Errorf("expected Column_2 in output")
	}
	// All three rows should be data rows when first row is numeric.
	if result.Metadata["row_count"] != "3" {
		t.Errorf("expected row_count=3, got %s", result.Metadata["row_count"])
	}
}

func TestExtractCSV_EmptyInput(t *testing.T) {
	_, err := ExtractCSV([]byte{}, CsvExtractorConfig{})
	if err == nil {
		t.Fatal("expected error for empty input")
	}
	if !strings.Contains(err.Error(), "empty csv") {
		t.Errorf("expected 'empty csv' error, got: %v", err)
	}
}

func TestExtractCSV_UTF8BOM(t *testing.T) {
	bom := []byte{0xEF, 0xBB, 0xBF}
	content := append(bom, []byte("Name,Age\nAlice,30\n")...)
	result, err := ExtractCSV(content, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["encoding"] != "utf-8-bom" {
		t.Errorf("expected encoding=utf-8-bom, got %s", result.Metadata["encoding"])
	}
	if !strings.Contains(result.Text, "- Name: Alice") {
		t.Errorf("BOM should be stripped, expected normal content, got:\n%s", result.Text)
	}
}

func TestExtractCSV_Latin1Encoding(t *testing.T) {
	// Latin-1 encoded: "Naïve,Café\nAlice,Zürich\n"
	// ï = 0xEF, é = 0xE9, ü = 0xFC in Latin-1
	input := []byte("Na\xEFve,Caf\xE9\nAlice,Z\xFCrich\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["encoding"] != "latin-1" {
		t.Errorf("expected encoding=latin-1, got %s", result.Metadata["encoding"])
	}
	if !strings.Contains(result.Text, "Café") {
		t.Errorf("expected decoded Latin-1 content with 'Café', got:\n%s", result.Text)
	}
}

func TestExtractCSV_RowChunking(t *testing.T) {
	var b strings.Builder
	b.WriteString("Name,Value\n")
	for i := 1; i <= 120; i++ {
		b.WriteString("item,")
		b.WriteString(strings.Repeat("x", 5))
		b.WriteByte('\n')
	}
	result, err := ExtractCSV([]byte(b.String()), CsvExtractorConfig{RowsPerChunk: 50})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 120 rows / 50 per chunk = 3 chunks. Chunks separated by "---".
	chunks := strings.Count(result.Text, "---")
	if chunks != 2 {
		t.Errorf("expected 2 chunk separators for 3 chunks, got %d", chunks)
	}
	if result.Metadata["row_count"] != "120" {
		t.Errorf("expected row_count=120, got %s", result.Metadata["row_count"])
	}
}

func TestExtractCSV_ForceDelimiter(t *testing.T) {
	input := []byte("Name;Age\nAlice;30\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{ForceDelimiter: ','})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// With comma forced, the semicolons are part of the value.
	if result.Metadata["delimiter"] != "," {
		t.Errorf("expected forced comma delimiter, got %q", result.Metadata["delimiter"])
	}
}

func TestExtractCSV_DuplicateHeaders(t *testing.T) {
	input := []byte("Name,Name,Age\nAlice,Bob,30\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Duplicate values in first row means it is not treated as headers.
	if !strings.Contains(result.Text, "Column_1") {
		t.Errorf("expected synthetic headers due to duplicate first row, got:\n%s", result.Text)
	}
}

// -------------------------------------------------------------------
// JSON Extractor Tests
// -------------------------------------------------------------------

func TestExtractJSON_SimpleObject(t *testing.T) {
	input := []byte(`{"name":"Alice","age":30,"city":"London"}`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["structure_type"] != "object" {
		t.Errorf("expected structure_type=object, got %s", result.Metadata["structure_type"])
	}
	if !strings.Contains(result.Text, "age:") {
		t.Errorf("expected key 'age' in output, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "Object with 3 keys") {
		t.Errorf("expected structural context, got:\n%s", result.Text)
	}
}

func TestExtractJSON_UniformArrayOfObjects(t *testing.T) {
	input := []byte(`[{"name":"Alice","age":30},{"name":"Bob","age":25},{"name":"Carol","age":35}]`)
	result, err := ExtractJSON(input, JsonExtractorConfig{TableThreshold: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["structure_type"] != "uniform_object_array" {
		t.Errorf("expected structure_type=uniform_object_array, got %s", result.Metadata["structure_type"])
	}
	// Should render as a markdown table.
	if !strings.Contains(result.Text, "| age |") || !strings.Contains(result.Text, "| name |") {
		t.Errorf("expected markdown table headers, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "| ---") {
		t.Errorf("expected markdown table separator, got:\n%s", result.Text)
	}
}

func TestExtractJSON_HeterogeneousArrayOfObjects(t *testing.T) {
	input := []byte(`[{"a":1,"b":2},{"c":3,"d":4}]`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Two objects with completely different keys should not render as table.
	if result.Metadata["structure_type"] != "heterogeneous_object_array" {
		t.Errorf("expected structure_type=heterogeneous_object_array, got %s", result.Metadata["structure_type"])
	}
	if !strings.Contains(result.Text, "Item 1:") {
		t.Errorf("expected individual item rendering, got:\n%s", result.Text)
	}
}

func TestExtractJSON_DeeplyNested(t *testing.T) {
	input := []byte(`{"a":{"b":{"c":{"d":{"e":"deep"}}}}}`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 5 levels deep => should flatten to dot-notation.
	if !strings.Contains(result.Text, "a.b.c.d.e: deep") {
		t.Errorf("expected dot-notation flattening, got:\n%s", result.Text)
	}
}

func TestExtractJSON_EmptyObject(t *testing.T) {
	input := []byte(`{}`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "{}" {
		t.Errorf("expected '{}', got %q", result.Text)
	}
}

func TestExtractJSON_EmptyArray(t *testing.T) {
	input := []byte(`[]`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "[]" {
		t.Errorf("expected '[]', got %q", result.Text)
	}
}

func TestExtractJSON_InvalidJSON(t *testing.T) {
	input := []byte(`{invalid`)
	_, err := ExtractJSON(input, JsonExtractorConfig{})
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "invalid json") {
		t.Errorf("expected 'invalid json' error, got: %v", err)
	}
}

func TestExtractJSON_PrimitiveArray(t *testing.T) {
	input := []byte(`[1,2,3,4,5]`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Metadata["structure_type"] != "primitive_array" {
		t.Errorf("expected structure_type=primitive_array, got %s", result.Metadata["structure_type"])
	}
	lines := strings.Split(result.Text, "\n")
	if len(lines) != 5 {
		t.Errorf("expected 5 lines for 5 primitives, got %d", len(lines))
	}
}

func TestExtractJSON_EmptyInput(t *testing.T) {
	_, err := ExtractJSON([]byte{}, JsonExtractorConfig{})
	if err == nil {
		t.Fatal("expected error for empty input")
	}
	if !strings.Contains(err.Error(), "empty json") {
		t.Errorf("expected 'empty json' error, got: %v", err)
	}
}

func TestExtractJSON_ObjectChunking(t *testing.T) {
	var objects []string
	for i := 0; i < 120; i++ {
		objects = append(objects, `{"id":1,"name":"test"}`)
	}
	input := []byte("[" + strings.Join(objects, ",") + "]")
	result, err := ExtractJSON(input, JsonExtractorConfig{ObjectsPerChunk: 50, TableThreshold: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 120 objects / 50 per chunk = 3 chunks. Chunks separated by "---".
	chunks := strings.Count(result.Text, "---")
	// Chunk separators include the markdown table separator rows plus
	// the inter-chunk "---" dividers. Each chunk has one "| --- |" row.
	// We expect 2 inter-chunk dividers.
	if chunks < 2 {
		t.Errorf("expected at least 2 chunk separators, got %d", chunks)
	}
}

func TestExtractJSON_SchemaDetection(t *testing.T) {
	input := []byte(`[
		{"name":"Alice","age":30,"city":"London"},
		{"name":"Bob","age":25,"city":"Paris"},
		{"name":"Carol","age":35,"city":"Berlin"}
	]`)
	result, err := ExtractJSON(input, JsonExtractorConfig{TableThreshold: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	schema := result.Metadata["detected_schema"]
	if schema == "" {
		t.Error("expected detected_schema in metadata")
	}
	if !strings.Contains(schema, "name") {
		t.Errorf("expected 'name' in detected schema, got %q", schema)
	}
}

func TestExtractJSON_NullValue(t *testing.T) {
	input := []byte(`{"key":null}`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "null") {
		t.Errorf("expected 'null' in output, got:\n%s", result.Text)
	}
}

func TestExtractJSON_BooleanValues(t *testing.T) {
	input := []byte(`{"active":true,"deleted":false}`)
	result, err := ExtractJSON(input, JsonExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "true") {
		t.Errorf("expected 'true' in output")
	}
	if !strings.Contains(result.Text, "false") {
		t.Errorf("expected 'false' in output")
	}
}

// -------------------------------------------------------------------
// JSONL Extractor Tests
// -------------------------------------------------------------------

func TestExtractJSONL_Basic(t *testing.T) {
	input := []byte("{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n")
	result, err := ExtractJSONL(input, JsonExtractorConfig{TableThreshold: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ContentType != "application/jsonl" {
		t.Errorf("expected MIME application/jsonl, got %s", result.ContentType)
	}
}

func TestExtractJSONL_Empty(t *testing.T) {
	_, err := ExtractJSONL([]byte{}, JsonExtractorConfig{})
	if err == nil {
		t.Fatal("expected error for empty JSONL")
	}
}

// -------------------------------------------------------------------
// Encoding Detection Tests
// -------------------------------------------------------------------

func TestDetectEncoding_UTF8_Structured(t *testing.T) {
	input := []byte("hello world")
	text, enc, err := detectEncoding(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if enc != "utf-8" {
		t.Errorf("expected utf-8, got %s", enc)
	}
	if text != "hello world" {
		t.Errorf("expected 'hello world', got %q", text)
	}
}

func TestDetectEncoding_UTF8BOM_Structured(t *testing.T) {
	input := append([]byte{0xEF, 0xBB, 0xBF}, []byte("hello")...)
	text, enc, err := detectEncoding(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if enc != "utf-8-bom" {
		t.Errorf("expected utf-8-bom, got %s", enc)
	}
	if text != "hello" {
		t.Errorf("expected 'hello', got %q", text)
	}
}

func TestDetectEncoding_Latin1_Structured(t *testing.T) {
	// 0x80 is not valid start of UTF-8 sequence.
	input := []byte{0x80, 0x81, 0x82}
	_, enc, err := detectEncoding(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if enc != "latin-1" {
		t.Errorf("expected latin-1, got %s", enc)
	}
}

// -------------------------------------------------------------------
// Delimiter Detection Tests
// -------------------------------------------------------------------

func TestDetectDelimiter_Comma(t *testing.T) {
	text := "a,b,c\n1,2,3\n4,5,6\n"
	d := detectDelimiter(text)
	if d != ',' {
		t.Errorf("expected comma, got %q", string(d))
	}
}

func TestDetectDelimiter_Semicolon(t *testing.T) {
	text := "a;b;c\n1;2;3\n4;5;6\n"
	d := detectDelimiter(text)
	if d != ';' {
		t.Errorf("expected semicolon, got %q", string(d))
	}
}

func TestDetectDelimiter_Tab(t *testing.T) {
	text := "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
	d := detectDelimiter(text)
	if d != '\t' {
		t.Errorf("expected tab, got %q", string(d))
	}
}

func TestDetectDelimiter_Pipe(t *testing.T) {
	text := "a|b|c\n1|2|3\n4|5|6\n"
	d := detectDelimiter(text)
	if d != '|' {
		t.Errorf("expected pipe, got %q", string(d))
	}
}

// -------------------------------------------------------------------
// Header Detection Tests
// -------------------------------------------------------------------

func TestLooksLikeHeaders_Valid(t *testing.T) {
	row := []string{"Name", "Age", "City"}
	if !looksLikeHeaders(row) {
		t.Error("expected true for text-only unique headers")
	}
}

func TestLooksLikeHeaders_NumericFirstRow(t *testing.T) {
	row := []string{"1", "2", "3"}
	if looksLikeHeaders(row) {
		t.Error("expected false for all-numeric row")
	}
}

func TestLooksLikeHeaders_EmptyValue(t *testing.T) {
	row := []string{"Name", "", "City"}
	if looksLikeHeaders(row) {
		t.Error("expected false when row contains empty value")
	}
}

func TestLooksLikeHeaders_Duplicates(t *testing.T) {
	row := []string{"Name", "Name", "Age"}
	if looksLikeHeaders(row) {
		t.Error("expected false for duplicate values")
	}
}

// -------------------------------------------------------------------
// CSV Injection Protection Tests
// -------------------------------------------------------------------

func TestExtractCSV_SanitisesFormulaInjection(t *testing.T) {
	input := []byte("Name,Formula\nAlice,=SUM(A1)\nBob,+CMD\nCarol,-1+1\nDave,@INDIRECT(A1)\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Values starting with formula characters should be escaped with a leading quote.
	if !strings.Contains(result.Text, "'=SUM(A1)") {
		t.Errorf("expected '=SUM(A1) (escaped), got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "'+CMD") {
		t.Errorf("expected '+CMD (escaped), got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "'-1+1") {
		t.Errorf("expected '-1+1 (escaped), got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "'@INDIRECT(A1)") {
		t.Errorf("expected '@INDIRECT(A1) (escaped), got:\n%s", result.Text)
	}
}

func TestExtractCSV_DoesNotSanitiseSafeValues(t *testing.T) {
	input := []byte("Name,Value\nAlice,hello\nBob,42\n")
	result, err := ExtractCSV(input, CsvExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(result.Text, "'hello") {
		t.Errorf("safe values should not be escaped, got:\n%s", result.Text)
	}
}

func TestExtractCSV_MaxInputSize(t *testing.T) {
	input := []byte("Name,Age\nAlice,30\n")
	_, err := ExtractCSV(input, CsvExtractorConfig{MaxInputSize: 5})
	if err == nil {
		t.Fatal("expected error for input exceeding max size")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected 'exceeds' error, got: %v", err)
	}
}

// -------------------------------------------------------------------
// XML Extractor Tests
// -------------------------------------------------------------------

func TestExtractXML_BasicDocument(t *testing.T) {
	input := []byte(`<root><item>hello</item><item>world</item></root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ContentType != "application/xml" {
		t.Errorf("expected MIME application/xml, got %s", result.ContentType)
	}
	if result.Metadata["root_element"] != "root" {
		t.Errorf("expected root_element=root, got %s", result.Metadata["root_element"])
	}
	if !strings.Contains(result.Text, "root/item: hello") {
		t.Errorf("expected element path context, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "root/item: world") {
		t.Errorf("expected second item, got:\n%s", result.Text)
	}
}

func TestExtractXML_WithAttributes(t *testing.T) {
	input := []byte(`<root><user id="1" name="Alice">text content</user></root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "root/user@id: 1") {
		t.Errorf("expected attribute rendering, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "root/user@name: Alice") {
		t.Errorf("expected name attribute, got:\n%s", result.Text)
	}
	if !strings.Contains(result.Text, "root/user: text content") {
		t.Errorf("expected text content, got:\n%s", result.Text)
	}
}

func TestExtractXML_NamespaceStripping(t *testing.T) {
	input := []byte(`<ns:root xmlns:ns="http://example.com"><ns:item>value</ns:item></ns:root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Namespace prefix should be stripped, only local name used.
	if !strings.Contains(result.Text, "root/item: value") {
		t.Errorf("expected stripped namespace, got:\n%s", result.Text)
	}
}

func TestExtractXML_CDATA(t *testing.T) {
	input := []byte(`<root><data><![CDATA[some <special> content]]></data></root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "some <special> content") {
		t.Errorf("expected CDATA content preserved, got:\n%s", result.Text)
	}
}

func TestExtractXML_ProcessingInstructionIgnored(t *testing.T) {
	input := []byte(`<?xml version="1.0"?><root><item>value</item></root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "root/item: value") {
		t.Errorf("expected content despite PI, got:\n%s", result.Text)
	}
}

func TestExtractXML_EmptyInput(t *testing.T) {
	_, err := ExtractXML([]byte{}, XmlExtractorConfig{})
	if err == nil {
		t.Fatal("expected error for empty input")
	}
	if !strings.Contains(err.Error(), "empty xml") {
		t.Errorf("expected 'empty xml' error, got: %v", err)
	}
}

func TestExtractXML_NestedElements(t *testing.T) {
	input := []byte(`<root><parent><child><grandchild>deep</grandchild></child></parent></root>`)
	result, err := ExtractXML(input, XmlExtractorConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "root/parent/child/grandchild: deep") {
		t.Errorf("expected nested path context, got:\n%s", result.Text)
	}
}

func TestExtractXML_Chunking(t *testing.T) {
	var b strings.Builder
	b.WriteString("<root>")
	for i := 0; i < 60; i++ {
		b.WriteString(fmt.Sprintf("<item>value_%d</item>", i))
	}
	b.WriteString("</root>")

	result, err := ExtractXML([]byte(b.String()), XmlExtractorConfig{ElementsPerChunk: 50})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chunks := strings.Count(result.Text, "---")
	if chunks < 1 {
		t.Errorf("expected at least 1 chunk separator for 60 elements with 50 per chunk, got %d", chunks)
	}
}

func TestExtractXML_MaxInputSize(t *testing.T) {
	input := []byte(`<root><item>value</item></root>`)
	_, err := ExtractXML(input, XmlExtractorConfig{MaxInputSize: 5})
	if err == nil {
		t.Fatal("expected error for input exceeding max size")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected 'exceeds' error, got: %v", err)
	}
}

// -------------------------------------------------------------------
// Extractor Interface Tests
// -------------------------------------------------------------------

func TestCSVExtractor_Interface(t *testing.T) {
	var e Extractor = &CSVExtractor{}
	if e.Name() != "csv" {
		t.Errorf("expected name 'csv', got %q", e.Name())
	}
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("expected Available() to return true")
	}
	cap := e.Capability()
	if len(cap.MIMETypes) == 0 {
		t.Error("expected non-empty capability MIMETypes")
	}
	result, err := e.Extract(t.Context(), []byte("Name,Age\nAlice,30\n"), ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "- Name: Alice") {
		t.Errorf("expected extracted content, got:\n%s", result.Text)
	}
}

func TestJSONExtractor_Interface(t *testing.T) {
	var e Extractor = &JSONExtractor{}
	if e.Name() != "json" {
		t.Errorf("expected name 'json', got %q", e.Name())
	}
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("expected Available() to return true")
	}
	result, err := e.Extract(t.Context(), []byte(`{"key":"value"}`), ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "key:") {
		t.Errorf("expected key in content, got:\n%s", result.Text)
	}
}

func TestJSONLExtractor_Interface(t *testing.T) {
	var e Extractor = &JSONLExtractor{}
	if e.Name() != "jsonl" {
		t.Errorf("expected name 'jsonl', got %q", e.Name())
	}
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("expected Available() to return true")
	}
	result, err := e.Extract(t.Context(), []byte("{\"a\":1}\n{\"a\":2}\n"), ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ContentType != "application/jsonl" {
		t.Errorf("expected ContentType application/jsonl, got %s", result.ContentType)
	}
}

func TestXMLExtractor_Interface(t *testing.T) {
	var e Extractor = &XMLExtractor{}
	if e.Name() != "xml" {
		t.Errorf("expected name 'xml', got %q", e.Name())
	}
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Error("expected Available() to return true")
	}
	cap := e.Capability()
	if len(cap.MIMETypes) == 0 {
		t.Error("expected non-empty capability MIMETypes")
	}
	result, err := e.Extract(t.Context(), []byte("<root><item>hello</item></root>"), ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "root/item: hello") {
		t.Errorf("expected extracted content, got:\n%s", result.Text)
	}
}

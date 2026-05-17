// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"encoding/xml"
	"fmt"
	"strconv"
	"strings"
)

// XmlExtractorConfig configures the XML extractor behaviour.
type XmlExtractorConfig struct {
	// ElementsPerChunk is the number of top-level child elements per
	// output chunk. Defaults to 50 when zero.
	ElementsPerChunk int

	// MaxDepth limits the recursion depth for nested elements.
	// Defaults to 10 when zero.
	MaxDepth int

	// MaxInputSize caps the raw input size in bytes. Inputs
	// exceeding this limit are rejected before parsing. Defaults
	// to 50 MiB when zero.
	MaxInputSize int64
}

func (c XmlExtractorConfig) elementsPerChunk() int {
	if c.ElementsPerChunk > 0 {
		return c.ElementsPerChunk
	}
	return 50
}

func (c XmlExtractorConfig) maxDepth() int {
	if c.MaxDepth > 0 {
		return c.MaxDepth
	}
	return 10
}

func (c XmlExtractorConfig) maxInputSize() int64 {
	if c.MaxInputSize > 0 {
		return c.MaxInputSize
	}
	return defaultMaxInputSize
}

// xmlNode represents a parsed XML element with its children, text,
// and attributes for rendering.
type xmlNode struct {
	localName  string
	attributes []xmlAttr
	children   []*xmlNode
	text       strings.Builder
}

type xmlAttr struct {
	name  string
	value string
}

// ExtractXML parses raw XML bytes and returns structured text output
// with element path context. CDATA sections are treated as text,
// namespaces are stripped to their local name, and processing
// instructions are ignored. Parsing uses a streaming xml.Decoder to
// build a lightweight tree, then renders it with path context.
func ExtractXML(raw []byte, cfg XmlExtractorConfig) (ExtractResult, error) {
	if len(raw) == 0 {
		return ExtractResult{}, fmt.Errorf("structured: empty xml input")
	}

	if int64(len(raw)) > cfg.maxInputSize() {
		return ExtractResult{}, fmt.Errorf("structured: xml input exceeds %d byte limit", cfg.maxInputSize())
	}

	text, encoding, err := detectEncoding(raw)
	if err != nil {
		return ExtractResult{}, err
	}

	decoder := xml.NewDecoder(strings.NewReader(text))
	decoder.Strict = false

	root, err := parseXMLTree(decoder, cfg.maxDepth())
	if err != nil {
		return ExtractResult{}, fmt.Errorf("structured: xml parse error: %w", err)
	}

	if root == nil {
		return ExtractResult{}, fmt.Errorf("structured: empty xml document")
	}

	// Render root's children as chunks.
	content, elementCount := renderXMLNodes(root, cfg)

	return ExtractResult{
		Text:        content,
		ContentType: "application/xml",
		Metadata: map[string]string{
			"encoding":      encoding,
			"root_element":  root.localName,
			"element_count": strconv.Itoa(elementCount),
		},
	}, nil
}

// parseXMLTree builds a lightweight tree from an xml.Decoder,
// stripping namespace prefixes and ignoring processing instructions.
// CDATA is folded into the parent element's text content.
func parseXMLTree(decoder *xml.Decoder, maxDepth int) (*xmlNode, error) {
	var root *xmlNode
	var stack []*xmlNode

	for {
		tok, err := decoder.Token()
		if err != nil {
			break
		}

		switch t := tok.(type) {
		case xml.StartElement:
			node := &xmlNode{
				localName: t.Name.Local,
			}
			for _, a := range t.Attr {
				node.attributes = append(node.attributes, xmlAttr{
					name:  a.Name.Local,
					value: a.Value,
				})
			}

			if len(stack) > 0 {
				parent := stack[len(stack)-1]
				parent.children = append(parent.children, node)
			}

			// Only descend if within depth limit.
			if len(stack) < maxDepth {
				stack = append(stack, node)
			}

			if root == nil {
				root = node
			}

		case xml.EndElement:
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}

		case xml.CharData:
			trimmed := strings.TrimSpace(string(t))
			if trimmed != "" && len(stack) > 0 {
				current := stack[len(stack)-1]
				if current.text.Len() > 0 {
					current.text.WriteByte(' ')
				}
				current.text.WriteString(trimmed)
			}

		case xml.ProcInst:
			// Processing instructions are ignored.
			continue
		}
	}

	return root, nil
}

// renderXMLNodes renders the XML tree as structured text with element
// path context and chunk boundaries.
func renderXMLNodes(root *xmlNode, cfg XmlExtractorConfig) (string, int) {
	if len(root.children) == 0 {
		// Single root with text only.
		txt := root.text.String()
		if txt == "" {
			return fmt.Sprintf("<%s/>", root.localName), 1
		}
		return fmt.Sprintf("%s: %s", root.localName, txt), 1
	}

	epc := cfg.elementsPerChunk()
	var parts []string
	totalElements := 0

	for i := 0; i < len(root.children); i += epc {
		end := i + epc
		if end > len(root.children) {
			end = len(root.children)
		}
		batch := root.children[i:end]

		var out strings.Builder
		for j, child := range batch {
			if j > 0 {
				out.WriteByte('\n')
			}
			count := renderXMLNode(&out, child, root.localName, 0, cfg.maxDepth())
			totalElements += count
		}
		parts = append(parts, out.String())
	}

	return strings.Join(parts, "\n\n---\n\n"), totalElements
}

// renderXMLNode recursively renders a single XML node with its
// element path context into the builder. Returns the count of
// elements rendered.
func renderXMLNode(out *strings.Builder, node *xmlNode, parentPath string, depth, maxDepth int) int {
	if depth >= maxDepth {
		out.WriteString(fmt.Sprintf("%s/%s: [max depth exceeded]\n", parentPath, node.localName))
		return 1
	}

	currentPath := parentPath + "/" + node.localName
	count := 1

	// Render attributes.
	for _, attr := range node.attributes {
		out.WriteString(fmt.Sprintf("%s@%s: %s\n", currentPath, attr.name, attr.value))
	}

	// Render text content.
	txt := node.text.String()
	if txt != "" {
		out.WriteString(fmt.Sprintf("%s: %s\n", currentPath, txt))
	}

	// Recurse into children.
	for _, child := range node.children {
		count += renderXMLNode(out, child, currentPath, depth+1, maxDepth)
	}

	return count
}

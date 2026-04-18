// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync/atomic"

	"github.com/jeffs-brain/memory/go/llm"
)

// ErrTransient is the sentinel returned when a provider call should be
// retried. The SDK's llm package does not classify HTTP status codes for
// us, so the judge layer wraps its own provider calls and tags errors
// whose message includes recognised transient signals (429 / 5xx / 529
// / deadline exceeded) with this sentinel.
var ErrTransient = errors.New("lme: transient provider error")

// ErrSchemaValidation is returned when the model's final attempt still
// fails schema validation.
var ErrSchemaValidation = errors.New("lme: schema validation failed")

// SchemaValidationError carries the last raw payload the model emitted
// alongside the human-readable validator reason.
type SchemaValidationError struct {
	RawPayload []byte
	Reason     string
}

func (e *SchemaValidationError) Error() string {
	return fmt.Sprintf("lme: schema validation failed: %s", e.Reason)
}

func (e *SchemaValidationError) Unwrap() error { return ErrSchemaValidation }

// transientRetryCounter tracks process-wide transient retries scheduled.
// Exposed via [TransientRetriesTotal] for the runner summary.
var transientRetryCounter atomic.Int64

// IncTransientRetry bumps the process-wide counter.
func IncTransientRetry() { transientRetryCounter.Add(1) }

// TransientRetriesTotal returns the current counter value.
func TransientRetriesTotal() int64 { return transientRetryCounter.Load() }

// ResetTransientRetries zeroes the counter. Test-only helper.
func ResetTransientRetries() { transientRetryCounter.Store(0) }

// isTransientErr reports whether a provider error should be treated as
// retryable. Matches common signals from upstream providers while
// staying provider-agnostic.
func isTransientErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrTransient) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, hint := range []string{"502", "503", "504", "529", "429", "timeout", "deadline", "connection reset"} {
		if strings.Contains(msg, hint) {
			return true
		}
	}
	return false
}

// structuredOpt customises [completeJSON].
type structuredOpt func(*structuredCfg)

type structuredCfg struct {
	maxRetries int
	usageHook  func(Usage)
	logger     *slog.Logger
}

// withMaxRetries caps the number of validation attempts. Zero or
// negative falls back to the default (5).
func withMaxRetries(n int) structuredOpt {
	return func(c *structuredCfg) {
		if n > 0 {
			c.maxRetries = n
		}
	}
}

// withUsageHook registers a callback fired once per provider call.
func withUsageHook(f func(Usage)) structuredOpt {
	return func(c *structuredCfg) { c.usageHook = f }
}

// defaultJSONRetryMax is the cap when withMaxRetries is not supplied.
const defaultJSONRetryMax = 5

// completeJSON wraps [llm.Provider.Complete] with schema validation and
// a bounded retry loop. Each attempt extracts the JSON payload from the
// response, validates it against schema, and either returns the bytes
// or appends the validation error as a corrective user-role message
// before retrying.
//
// The final error is [ErrSchemaValidation] wrapped in
// [*SchemaValidationError] so callers can recover the raw payload.
func completeJSON(
	ctx context.Context,
	p llm.Provider,
	req llm.CompleteRequest,
	schema json.RawMessage,
	opts ...structuredOpt,
) (json.RawMessage, error) {
	if p == nil {
		return nil, fmt.Errorf("lme: completeJSON: nil provider")
	}
	if len(schema) == 0 {
		return nil, fmt.Errorf("lme: completeJSON: empty schema")
	}

	cfg := structuredCfg{
		maxRetries: defaultJSONRetryMax,
		logger:     slog.Default(),
	}
	for _, o := range opts {
		if o != nil {
			o(&cfg)
		}
	}

	// Inject a system-prompt instruction so providers without native
	// structured decoding (the SDK's baseline Provider surface) still
	// produce JSON output.
	instr := structuredOutputInstruction(schema)

	messages := make([]llm.Message, 0, len(req.Messages)+1)
	if instr != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: instr})
	}
	messages = append(messages, req.Messages...)

	var lastPayload []byte
	var lastReason string

	for attempt := 1; attempt <= cfg.maxRetries; attempt++ {
		req.Messages = messages

		resp, err := p.Complete(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("lme: completeJSON: provider call (attempt %d/%d): %w", attempt, cfg.maxRetries, err)
		}
		if cfg.usageHook != nil {
			cfg.usageHook(usageFromResponse(resp))
		}

		payload, extractErr := extractJSON(resp.Text)
		if extractErr != nil {
			lastPayload = []byte(resp.Text)
			lastReason = extractErr.Error()
			messages = appendCorrection(messages, resp, lastReason)
			continue
		}

		if err := validateAgainstSchema(payload, schema); err != nil {
			lastPayload = payload
			lastReason = err.Error()
			messages = appendCorrection(messages, resp, lastReason)
			continue
		}

		return payload, nil
	}

	return nil, &SchemaValidationError{RawPayload: lastPayload, Reason: lastReason}
}

// fencedJSONRe matches a fenced JSON code block.
var fencedJSONRe = regexp.MustCompile("(?s)```(?:json)?\\s*\\n(.*?)\\n```")

// extractJSON pulls JSON out of a model response, tolerating fenced code
// blocks, leading prose, and surrounding whitespace.
func extractJSON(content string) (json.RawMessage, error) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, fmt.Errorf("empty response")
	}

	if m := fencedJSONRe.FindStringSubmatch(trimmed); len(m) == 2 {
		inner := strings.TrimSpace(m[1])
		if inner != "" {
			return json.RawMessage(inner), nil
		}
	}

	for i, r := range trimmed {
		if r != '{' && r != '[' {
			continue
		}
		candidate := trimmed[i:]
		dec := json.NewDecoder(strings.NewReader(candidate))
		var raw json.RawMessage
		if err := dec.Decode(&raw); err == nil {
			return raw, nil
		}
	}

	return nil, fmt.Errorf("no JSON object or array found in response")
}

// appendCorrection adds the model's previous attempt as an assistant
// turn followed by a user-role critique that explains the validation
// failure. Returning a fresh slice keeps the caller's history untouched
// between attempts.
func appendCorrection(prev []llm.Message, resp llm.CompleteResponse, reason string) []llm.Message {
	out := make([]llm.Message, 0, len(prev)+2)
	out = append(out, prev...)
	if resp.Text != "" {
		out = append(out, llm.Message{Role: llm.RoleAssistant, Content: resp.Text})
	}
	out = append(out, llm.Message{
		Role: llm.RoleUser,
		Content: fmt.Sprintf(
			"Your previous response failed schema validation: %s. Return only valid JSON matching the schema.",
			reason,
		),
	})
	return out
}

// structuredOutputInstruction returns the system instruction that primes
// a non-structured provider to emit JSON matching the schema.
func structuredOutputInstruction(schema json.RawMessage) string {
	if len(schema) == 0 {
		return ""
	}
	return fmt.Sprintf(
		"Respond with ONLY a JSON object matching this schema. No prose before or after.\n\n<output_schema>\n%s\n</output_schema>",
		string(schema),
	)
}

// validateAgainstSchema performs a minimal JSON Schema validation
// sufficient for the judge's fixed `{verdict, rationale}` shape: verifies
// that the payload parses as JSON, that every required key is present,
// and that enum constraints on the verdict field are honoured. Kept
// deliberately small so the LME package does not pull in a full JSON
// Schema compiler; the judge schema is the only consumer.
func validateAgainstSchema(payload, schema json.RawMessage) error {
	var instance map[string]any
	if err := json.Unmarshal(payload, &instance); err != nil {
		return fmt.Errorf("payload is not a JSON object: %w", err)
	}

	var schemaDoc struct {
		Required   []string `json:"required"`
		Properties map[string]struct {
			Type string   `json:"type"`
			Enum []string `json:"enum"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(schema, &schemaDoc); err != nil {
		return fmt.Errorf("schema is not valid JSON: %w", err)
	}

	for _, key := range schemaDoc.Required {
		if _, ok := instance[key]; !ok {
			return fmt.Errorf("missing required property %q", key)
		}
	}

	for key, prop := range schemaDoc.Properties {
		val, present := instance[key]
		if !present {
			continue
		}
		if prop.Type == "string" {
			s, ok := val.(string)
			if !ok {
				return fmt.Errorf("property %q must be a string", key)
			}
			if len(prop.Enum) > 0 {
				allowed := false
				for _, e := range prop.Enum {
					if s == e {
						allowed = true
						break
					}
				}
				if !allowed {
					return fmt.Errorf("property %q value %q is not in allowed enum", key, s)
				}
			}
		}
	}

	return nil
}

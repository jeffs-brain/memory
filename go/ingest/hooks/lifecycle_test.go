// SPDX-License-Identifier: Apache-2.0

package hooks

import (
	"context"
	"errors"
	"testing"
)

// testLogger captures log calls for assertions.
type testLogger struct {
	warns []string
	infos []string
}

func (l *testLogger) Debug(_ string, _ ...map[string]string) {}
func (l *testLogger) Info(msg string, _ ...map[string]string)  { l.infos = append(l.infos, msg) }
func (l *testLogger) Warn(msg string, _ ...map[string]string)  { l.warns = append(l.warns, msg) }
func (l *testLogger) Error(_ string, _ ...map[string]string) {}

// orderPlugin records its invocation order and allows configurable returns.
type orderPlugin struct {
	BaseIngestPlugin
	order           *[]string
	detectReturn    bool
	detectErr       error
	startReturn     bool
	startErr        error
	endErr          error
}

func (p *orderPlugin) OnDocumentDetected(_ context.Context, _ DocumentDetectedEvent) (bool, error) {
	*p.order = append(*p.order, p.PluginName+":detect")
	return p.detectReturn, p.detectErr
}

func (p *orderPlugin) OnIngestStart(_ context.Context, _ IngestHookEvent) (bool, error) {
	*p.order = append(*p.order, p.PluginName+":start")
	return p.startReturn, p.startErr
}

func (p *orderPlugin) OnIngestEnd(_ context.Context, _ IngestHookEvent) error {
	*p.order = append(*p.order, p.PluginName+":end")
	return p.endErr
}

func TestFireDocumentDetected_AllPluginsCalledInOrder(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, detectReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "b"}, order: order, detectReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order, detectReturn: true},
	}
	logger := &testLogger{}

	cancelled := FireDocumentDetected(context.Background(), plugins, DocumentDetectedEvent{BrainID: "b1"}, logger)

	if cancelled {
		t.Error("expected not cancelled")
	}
	expected := []string{"a:detect", "b:detect", "c:detect"}
	if len(*order) != len(expected) {
		t.Fatalf("expected %v, got %v", expected, *order)
	}
	for i, v := range expected {
		if (*order)[i] != v {
			t.Errorf("order[%d] = %q, want %q", i, (*order)[i], v)
		}
	}
}

func TestFireDocumentDetected_CancelledByPlugin(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, detectReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "blocker"}, order: order, detectReturn: false},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order, detectReturn: true},
	}
	logger := &testLogger{}

	cancelled := FireDocumentDetected(context.Background(), plugins, DocumentDetectedEvent{BrainID: "b1"}, logger)

	if !cancelled {
		t.Error("expected cancelled")
	}
	// c should not have been called since blocker returned false
	if len(*order) != 2 {
		t.Errorf("expected 2 invocations, got %d: %v", len(*order), *order)
	}
}

func TestFireDocumentDetected_ErrorSwallowed(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, detectReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "crasher"}, order: order, detectReturn: true, detectErr: errors.New("boom")},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order, detectReturn: true},
	}
	logger := &testLogger{}

	cancelled := FireDocumentDetected(context.Background(), plugins, DocumentDetectedEvent{BrainID: "b1"}, logger)

	if cancelled {
		t.Error("expected not cancelled after error (fail open)")
	}
	if len(*order) != 3 {
		t.Errorf("expected 3 invocations, got %d: %v", len(*order), *order)
	}
	if len(logger.warns) == 0 {
		t.Error("expected a warning to be logged")
	}
}

func TestFireIngestStart_AllPluginsCalledInOrder(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, startReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "b"}, order: order, startReturn: true},
	}
	logger := &testLogger{}

	cancelled := FireIngestStart(context.Background(), plugins, IngestHookEvent{BrainID: "b1", Path: "test.md"}, logger)

	if cancelled {
		t.Error("expected not cancelled")
	}
	expected := []string{"a:start", "b:start"}
	if len(*order) != len(expected) {
		t.Fatalf("expected %v, got %v", expected, *order)
	}
	for i, v := range expected {
		if (*order)[i] != v {
			t.Errorf("order[%d] = %q, want %q", i, (*order)[i], v)
		}
	}
}

func TestFireIngestStart_CancelledByPlugin(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, startReturn: true},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "blocker"}, order: order, startReturn: false},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order, startReturn: true},
	}
	logger := &testLogger{}

	cancelled := FireIngestStart(context.Background(), plugins, IngestHookEvent{BrainID: "b1"}, logger)

	if !cancelled {
		t.Error("expected cancelled")
	}
}

func TestFireIngestStart_ErrorSwallowed(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "crasher"}, order: order, startReturn: true, startErr: errors.New("fail")},
	}
	logger := &testLogger{}

	cancelled := FireIngestStart(context.Background(), plugins, IngestHookEvent{BrainID: "b1"}, logger)

	if cancelled {
		t.Error("expected not cancelled after error")
	}
	if len(logger.warns) == 0 {
		t.Error("expected a warning to be logged")
	}
}

func TestFireIngestEnd_ReverseOrder(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "b"}, order: order},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order},
	}
	logger := &testLogger{}

	FireIngestEnd(context.Background(), plugins, IngestHookEvent{BrainID: "b1"}, logger)

	expected := []string{"c:end", "b:end", "a:end"}
	if len(*order) != len(expected) {
		t.Fatalf("expected %v, got %v", expected, *order)
	}
	for i, v := range expected {
		if (*order)[i] != v {
			t.Errorf("order[%d] = %q, want %q", i, (*order)[i], v)
		}
	}
}

func TestFireIngestEnd_ErrorSwallowedContinuesRemaining(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "crasher"}, order: order, endErr: errors.New("end fail")},
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "c"}, order: order},
	}
	logger := &testLogger{}

	FireIngestEnd(context.Background(), plugins, IngestHookEvent{BrainID: "b1"}, logger)

	// Reverse order: c, crasher (errors but continues), a
	expected := []string{"c:end", "crasher:end", "a:end"}
	if len(*order) != len(expected) {
		t.Fatalf("expected %v, got %v", expected, *order)
	}
	for i, v := range expected {
		if (*order)[i] != v {
			t.Errorf("order[%d] = %q, want %q", i, (*order)[i], v)
		}
	}
	if len(logger.warns) == 0 {
		t.Error("expected a warning to be logged for crasher")
	}
}

func TestFireIngestEnd_EmptyPlugins(t *testing.T) {
	logger := &testLogger{}

	// Should not panic with empty slice.
	FireIngestEnd(context.Background(), nil, IngestHookEvent{BrainID: "b1"}, logger)
	FireIngestEnd(context.Background(), []IngestPlugin{}, IngestHookEvent{BrainID: "b1"}, logger)
}

func TestBaseIngestPlugin_Defaults(t *testing.T) {
	p := &BaseIngestPlugin{PluginName: "test-base"}

	if p.Name() != "test-base" {
		t.Errorf("expected 'test-base', got %q", p.Name())
	}

	proceed, err := p.OnDocumentDetected(context.Background(), DocumentDetectedEvent{})
	if !proceed || err != nil {
		t.Errorf("expected (true, nil), got (%v, %v)", proceed, err)
	}

	proceed, err = p.OnIngestStart(context.Background(), IngestHookEvent{})
	if !proceed || err != nil {
		t.Errorf("expected (true, nil), got (%v, %v)", proceed, err)
	}

	err = p.OnIngestEnd(context.Background(), IngestHookEvent{})
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
}

func TestFireDocumentDetected_NilLogger(t *testing.T) {
	order := &[]string{}
	plugins := []IngestPlugin{
		&orderPlugin{BaseIngestPlugin: BaseIngestPlugin{PluginName: "a"}, order: order, detectReturn: false},
	}

	// Should not panic with nil logger.
	cancelled := FireDocumentDetected(context.Background(), plugins, DocumentDetectedEvent{BrainID: "b1"}, nil)
	if !cancelled {
		t.Error("expected cancelled")
	}
}

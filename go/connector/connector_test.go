// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/connector"
)

// stubConnector is a minimal Connector implementation for testing the
// registry. It does not perform real syncs.
type stubConnector struct {
	name string
}

func (s *stubConnector) Name() string { return s.name }

func (s *stubConnector) Configure(_ connector.ConnectorConfigMap) error { return nil }

func (s *stubConnector) FetchAll(_ context.Context) (<-chan connector.ConnectorDocument, <-chan error) {
	docs := make(chan connector.ConnectorDocument)
	errs := make(chan error, 1)
	close(docs)
	close(errs)
	return docs, errs
}

func (s *stubConnector) FetchSince(_ context.Context, _ connector.SyncCursor) (<-chan connector.ConnectorDocument, <-chan error) {
	docs := make(chan connector.ConnectorDocument)
	errs := make(chan error, 1)
	close(docs)
	close(errs)
	return docs, errs
}

func (s *stubConnector) Start(_ context.Context) error { return nil }
func (s *stubConnector) Stop() error                   { return nil }
func (s *stubConnector) Health() connector.HealthStatus {
	return connector.HealthStatus{
		Status:             connector.StatusConnected,
		RateLimitRemaining: -1,
	}
}

func newStubFactory(name string) connector.ConnectorFactory {
	return func(cfg connector.ConnectorConfig) connector.Connector {
		return &stubConnector{name: name}
	}
}

func TestRegistry_RegisterAndLookup(t *testing.T) {
	reg := connector.NewRegistry()

	if err := reg.Register("slack", newStubFactory("slack")); err != nil {
		t.Fatalf("Register slack: %v", err)
	}

	if err := reg.Register("gdrive", newStubFactory("gdrive")); err != nil {
		t.Fatalf("Register gdrive: %v", err)
	}

	cfg := connector.ConnectorConfig{BrainID: "brain-1"}
	conn, err := reg.Lookup("slack", cfg)
	if err != nil {
		t.Fatalf("Lookup slack: %v", err)
	}
	if conn.Name() != "slack" {
		t.Errorf("connector name = %q, want %q", conn.Name(), "slack")
	}
}

func TestRegistry_RegisterDuplicate(t *testing.T) {
	reg := connector.NewRegistry()
	if err := reg.Register("slack", newStubFactory("slack")); err != nil {
		t.Fatalf("first Register: %v", err)
	}
	err := reg.Register("slack", newStubFactory("slack"))
	if err == nil {
		t.Fatal("expected error on duplicate registration")
	}
}

func TestRegistry_LookupNotFound(t *testing.T) {
	reg := connector.NewRegistry()
	_, err := reg.Lookup("nonexistent", connector.ConnectorConfig{})
	if err == nil {
		t.Fatal("expected error for unknown connector")
	}
}

func TestRegistry_Names(t *testing.T) {
	reg := connector.NewRegistry()
	_ = reg.Register("slack", newStubFactory("slack"))
	_ = reg.Register("gdrive", newStubFactory("gdrive"))
	_ = reg.Register("notion", newStubFactory("notion"))

	names := reg.Names()
	sort.Strings(names)
	want := []string{"gdrive", "notion", "slack"}
	if len(names) != len(want) {
		t.Fatalf("Names returned %d, want %d", len(names), len(want))
	}
	for i, n := range names {
		if n != want[i] {
			t.Errorf("names[%d] = %q, want %q", i, n, want[i])
		}
	}
}

func TestRegistry_Has(t *testing.T) {
	reg := connector.NewRegistry()
	_ = reg.Register("slack", newStubFactory("slack"))

	if !reg.Has("slack") {
		t.Error("Has(slack) = false, want true")
	}
	if reg.Has("notion") {
		t.Error("Has(notion) = true, want false")
	}
}

func TestConnectorConfig_EffectivePollInterval(t *testing.T) {
	tests := []struct {
		name     string
		interval time.Duration
		want     time.Duration
	}{
		{"zero uses default", 0, connector.DefaultPollInterval},
		{"custom interval", 30 * time.Second, 30 * time.Second},
		{"negative uses default", -1 * time.Second, connector.DefaultPollInterval},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := connector.ConnectorConfig{PollInterval: tt.interval}
			got := cfg.EffectivePollInterval()
			if got != tt.want {
				t.Errorf("EffectivePollInterval = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHealthStatus_Types(t *testing.T) {
	health := connector.HealthStatus{
		Status:             connector.StatusConnected,
		ErrorCount:         0,
		RateLimitRemaining: -1,
	}
	if health.Status != connector.StatusConnected {
		t.Errorf("Status = %q, want %q", health.Status, connector.StatusConnected)
	}

	health.Status = connector.StatusDegraded
	if health.Status != connector.StatusDegraded {
		t.Errorf("Status = %q, want %q", health.Status, connector.StatusDegraded)
	}

	health.Status = connector.StatusDisconnected
	if health.Status != connector.StatusDisconnected {
		t.Errorf("Status = %q, want %q", health.Status, connector.StatusDisconnected)
	}
}

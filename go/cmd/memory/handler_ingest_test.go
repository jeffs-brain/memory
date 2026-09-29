// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/internal/httpd"
	"github.com/jeffs-brain/memory/go/knowledge"
)

func postIngest(t *testing.T, c *http.Client, url string, body map[string]any) (int, string) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	resp, err := c.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("post %s: %v", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(out)
}

func TestIngestFile_PathDisabledWithoutIngestRoot(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "noroot")

	secret := filepath.Join(t.TempDir(), "secret.md")
	if err := os.WriteFile(secret, []byte("top secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	status, body := postIngest(t, c, srv.URL+"/v1/brains/noroot/ingest/file", map[string]any{"path": secret})
	if status != http.StatusForbidden {
		t.Fatalf("status = %d, want 403; body=%s", status, body)
	}
	if strings.Contains(body, "top secret") {
		t.Fatalf("response leaked file content: %s", body)
	}
}

func TestIngestFile_PathConfinedToIngestRoot(t *testing.T) {
	daemon, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "rooted")

	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "notes.md"), []byte("# notes\n\nhedgehogs"), 0o600); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "outside.md")
	if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "escape.md")); err != nil {
		t.Fatal(err)
	}
	resolved, err := resolveIngestRoot(root)
	if err != nil {
		t.Fatalf("resolveIngestRoot: %v", err)
	}
	daemon.IngestRoot = resolved

	url := srv.URL + "/v1/brains/rooted/ingest/file"
	cases := []struct {
		name string
		path string
		want int
	}{
		{name: "relative inside root", path: "notes.md", want: http.StatusOK},
		{name: "absolute inside root", path: filepath.Join(root, "notes.md"), want: http.StatusOK},
		{name: "absolute outside root", path: outside, want: http.StatusForbidden},
		{name: "dot-dot traversal", path: "../" + filepath.Base(filepath.Dir(outside)) + "/outside.md", want: http.StatusForbidden},
		{name: "symlink escaping root", path: "escape.md", want: http.StatusForbidden},
		{name: "missing file outside root", path: "../does-not-exist.md", want: http.StatusForbidden},
		{name: "missing file", path: "absent.md", want: http.StatusBadRequest},
		{name: "directory", path: ".", want: http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, body := postIngest(t, c, url, map[string]any{"path": tc.path})
			if status != tc.want {
				t.Fatalf("status = %d, want %d; body=%s", status, tc.want, body)
			}
		})
	}
}

func TestIngestFile_RequiresPathOrContent(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "empty")
	status, body := postIngest(t, c, srv.URL+"/v1/brains/empty/ingest/file", map[string]any{})
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", status, body)
	}
}

func TestIngestFile_UnsupportedContentIsValidationError(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "badtype")
	status, body := postIngest(t, c, srv.URL+"/v1/brains/badtype/ingest/file", map[string]any{
		"path":          "blob.bin",
		"contentType":   "application/x-unknown",
		"contentBase64": "AAECAw==",
	})
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", status, body)
	}
}

func TestIngestURL_RefusesLoopbackAndBadSchemes(t *testing.T) {
	_, srv := newTestDaemon(t)
	c := srv.Client()
	mustCreateBrain(t, c, srv.URL, "ssrf")
	url := srv.URL + "/v1/brains/ssrf/ingest/url"
	for _, target := range []string{
		srv.URL + "/v1/brains",
		"http://169.254.169.254/latest/meta-data/",
		"file:///etc/passwd",
	} {
		status, body := postIngest(t, c, url, map[string]any{"url": target})
		if status != http.StatusBadRequest {
			t.Fatalf("ingest %s: status = %d, want 400; body=%s", target, status, body)
		}
	}
}

func TestResolveIngestRoot(t *testing.T) {
	if got, err := resolveIngestRoot(""); err != nil || got != "" {
		t.Fatalf("empty root = %q, %v", got, err)
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveIngestRoot(file); err == nil {
		t.Fatal("a file must not be accepted as an ingest root")
	}
	if _, err := resolveIngestRoot(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("a missing directory must not be accepted as an ingest root")
	}
}

func TestWriteIngestError(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
		wantDetail string
	}{
		{name: "path ingest disabled", err: errPathIngestDisabled, wantStatus: http.StatusForbidden, wantCode: "forbidden"},
		{name: "outside root", err: errPathOutsideRoot, wantStatus: http.StatusForbidden, wantCode: "forbidden"},
		{name: "invalid url", err: fmt.Errorf("%w: bad", knowledge.ErrInvalidURL), wantStatus: http.StatusBadRequest, wantCode: "validation_error"},
		{name: "blocked address", err: fmt.Errorf("%w: 10.0.0.1", knowledge.ErrBlockedAddress), wantStatus: http.StatusBadRequest, wantCode: "validation_error"},
		{name: "invalid content", err: fmt.Errorf("%w: empty", knowledge.ErrInvalidContent), wantStatus: http.StatusBadRequest, wantCode: "validation_error"},
		{
			name:       "upstream failure",
			err:        fmt.Errorf("ingest: %w", &knowledge.FetchError{StatusCode: http.StatusServiceUnavailable}),
			wantStatus: http.StatusBadGateway,
			wantCode:   "bad_gateway",
		},
		{
			name:       "anything else",
			err:        errors.New("secret detail /var/lib/private.db"),
			wantStatus: http.StatusInternalServerError,
			wantCode:   "internal_error",
			wantDetail: "internal error",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			writeIngestError(rec, tt.err)
			if rec.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			var p httpd.Problem
			if err := json.Unmarshal(rec.Body.Bytes(), &p); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if p.Code != tt.wantCode {
				t.Errorf("code = %q, want %q", p.Code, tt.wantCode)
			}
			if tt.wantDetail != "" && p.Detail != tt.wantDetail {
				t.Errorf("detail = %q, want %q", p.Detail, tt.wantDetail)
			}
		})
	}
}

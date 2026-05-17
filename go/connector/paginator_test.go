// SPDX-License-Identifier: Apache-2.0

package connector_test

import (
	"context"
	"errors"
	"testing"

	"github.com/jeffs-brain/memory/go/connector"
)

func TestPaginate_SinglePage(t *testing.T) {
	fetchPage := func(_ context.Context, cursor string) (connector.PageResult[string], error) {
		if cursor != "" {
			t.Fatalf("unexpected cursor: %q", cursor)
		}
		return connector.PageResult[string]{
			Items: []string{"a", "b", "c"},
		}, nil
	}

	items, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
	})
	if err != nil {
		t.Fatalf("CollectPages: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
}

func TestPaginate_MultiPage(t *testing.T) {
	page := 0
	fetchPage := func(_ context.Context, cursor string) (connector.PageResult[string], error) {
		defer func() { page++ }()
		switch page {
		case 0:
			return connector.PageResult[string]{
				Items:      []string{"a", "b"},
				NextCursor: "page2",
			}, nil
		case 1:
			return connector.PageResult[string]{
				Items:      []string{"c", "d"},
				NextCursor: "page3",
			}, nil
		case 2:
			return connector.PageResult[string]{
				Items: []string{"e"},
			}, nil
		default:
			t.Fatal("unexpected page")
			return connector.PageResult[string]{}, nil
		}
	}

	items, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
	})
	if err != nil {
		t.Fatalf("CollectPages: %v", err)
	}
	if len(items) != 5 {
		t.Fatalf("got %d items, want 5", len(items))
	}
	want := []string{"a", "b", "c", "d", "e"}
	for i, v := range items {
		if v != want[i] {
			t.Errorf("items[%d] = %q, want %q", i, v, want[i])
		}
	}
}

func TestPaginate_MaxPagesGuard(t *testing.T) {
	fetchPage := func(_ context.Context, _ string) (connector.PageResult[string], error) {
		return connector.PageResult[string]{
			Items:      []string{"x"},
			NextCursor: "always-more",
		}, nil
	}

	_, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
		MaxPages:  3,
	})
	if err == nil {
		t.Fatal("expected max pages error")
	}
	if !errors.Is(err, connector.ErrMaxPagesExceeded) {
		t.Errorf("error should wrap ErrMaxPagesExceeded: %v", err)
	}
}

func TestPaginate_EmptyFirstPage(t *testing.T) {
	fetchPage := func(_ context.Context, _ string) (connector.PageResult[string], error) {
		return connector.PageResult[string]{}, nil
	}

	items, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
	})
	if err != nil {
		t.Fatalf("CollectPages: %v", err)
	}
	if len(items) != 0 {
		t.Fatalf("got %d items, want 0", len(items))
	}
}

func TestPaginate_FetchError(t *testing.T) {
	sentinel := errors.New("api error")
	fetchPage := func(_ context.Context, _ string) (connector.PageResult[string], error) {
		return connector.PageResult[string]{}, sentinel
	}

	_, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("error should wrap sentinel: %v", err)
	}
}

func TestPaginate_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	callCount := 0
	fetchPage := func(_ context.Context, _ string) (connector.PageResult[string], error) {
		callCount++
		if callCount > 1 {
			cancel()
		}
		return connector.PageResult[string]{
			Items:      []string{"x"},
			NextCursor: "more",
		}, nil
	}

	_, err := connector.CollectPages(ctx, connector.PaginatorConfig[string]{
		FetchPage: fetchPage,
	})
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}

func TestPaginate_WithRateLimiter(t *testing.T) {
	rl := connector.NewRateLimiter(connector.RateLimiterConfig{
		MaxTokens:  100,
		RefillRate: 100,
	})
	defer rl.Close()

	page := 0
	fetchPage := func(_ context.Context, _ string) (connector.PageResult[string], error) {
		defer func() { page++ }()
		switch {
		case page < 2:
			return connector.PageResult[string]{
				Items:      []string{"item"},
				NextCursor: "next",
			}, nil
		default:
			return connector.PageResult[string]{
				Items: []string{"last"},
			}, nil
		}
	}

	items, err := connector.CollectPages(context.Background(), connector.PaginatorConfig[string]{
		FetchPage:   fetchPage,
		RateLimiter: rl,
	})
	if err != nil {
		t.Fatalf("CollectPages: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
}

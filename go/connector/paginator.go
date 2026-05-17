// SPDX-License-Identifier: Apache-2.0

package connector

import (
	"context"
	"errors"
	"fmt"
)

// Pagination sentinel errors.
var (
	// ErrMaxPagesExceeded is returned when the paginator hits the
	// configured maximum page count guard.
	ErrMaxPagesExceeded = errors.New("connector: max pages exceeded")
)

// PageResult holds a single page of results from a paginated API.
type PageResult[T any] struct {
	// Items contains the results for this page.
	Items []T
	// NextCursor holds the cursor value for fetching the next page.
	// Empty when there are no more pages.
	NextCursor string
	// Total is the total number of items across all pages, if known.
	// Zero when the API does not report totals.
	Total int
}

// FetchPageFunc is a function that fetches a single page of results.
// An empty cursor requests the first page.
type FetchPageFunc[T any] func(ctx context.Context, cursor string) (PageResult[T], error)

// PaginatorConfig configures a paginator.
type PaginatorConfig[T any] struct {
	// FetchPage is the function that fetches a single page.
	FetchPage FetchPageFunc[T]
	// PageSize is a hint for the page size. The actual size depends on
	// the API. Defaults to 100.
	PageSize int
	// MaxPages is the maximum number of pages to fetch before stopping.
	// Prevents infinite loops when an API always returns a next cursor.
	// Defaults to 10000.
	MaxPages int
	// RateLimiter is an optional rate limiter. When set, Acquire(1) is
	// called before each page fetch.
	RateLimiter *RateLimiter
}

func (c PaginatorConfig[T]) withDefaults() PaginatorConfig[T] {
	out := c
	if out.PageSize <= 0 {
		out.PageSize = 100
	}
	if out.MaxPages <= 0 {
		out.MaxPages = 10000
	}
	return out
}

// Paginate iterates through all pages, sending items to the returned
// channel. The error channel receives at most one error. Both channels
// are closed when pagination completes.
func Paginate[T any](ctx context.Context, config PaginatorConfig[T]) (<-chan T, <-chan error) {
	cfg := config.withDefaults()
	items := make(chan T)
	errs := make(chan error, 1)

	go func() {
		defer close(items)
		defer close(errs)

		cursor := ""
		for page := 0; page < cfg.MaxPages; page++ {
			if cfg.RateLimiter != nil {
				if err := cfg.RateLimiter.Acquire(ctx, 1); err != nil {
					errs <- fmt.Errorf("connector: paginator rate limit: %w", err)
					return
				}
			}

			result, err := cfg.FetchPage(ctx, cursor)
			if err != nil {
				errs <- fmt.Errorf("connector: paginator fetch page %d: %w", page, err)
				return
			}

			for _, item := range result.Items {
				select {
				case items <- item:
				case <-ctx.Done():
					errs <- ctx.Err()
					return
				}
			}

			if result.NextCursor == "" {
				return
			}
			cursor = result.NextCursor
		}

		errs <- ErrMaxPagesExceeded
	}()

	return items, errs
}

// CollectPages is a convenience function that collects all paginated
// items into a single slice. Suitable for moderate result sets.
func CollectPages[T any](ctx context.Context, config PaginatorConfig[T]) ([]T, error) {
	itemsCh, errsCh := Paginate(ctx, config)
	var items []T
	for item := range itemsCh {
		items = append(items, item)
	}
	if err, ok := <-errsCh; ok && err != nil {
		return items, err
	}
	return items, nil
}

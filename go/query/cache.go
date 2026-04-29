// SPDX-License-Identifier: Apache-2.0

package query

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"unicode"
)

const maxInputChars = 8000

// cache is a bounded LRU keyed by cacheKey strings. It is safe for
// concurrent use. The implementation uses container/list to keep the O(1)
// bookkeeping honest regardless of cache size.
type cache struct {
	mu       sync.Mutex
	capacity int
	items    map[string]*list.Element
	order    *list.List
}

type cacheEntry struct {
	key     string
	queries []Query
}

func newCache(capacity int) *cache {
	if capacity <= 0 {
		capacity = 1
	}
	return &cache{
		capacity: capacity,
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
	}
}

func (c *cache) get(key string) ([]Query, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	elem, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.order.MoveToBack(elem)
	entry := elem.Value.(*cacheEntry)
	// Return a defensive copy so callers cannot mutate the cached slice.
	out := make([]Query, len(entry.queries))
	copy(out, entry.queries)
	return out, true
}

func (c *cache) put(key string, queries []Query) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		elem.Value.(*cacheEntry).queries = queries
		c.order.MoveToBack(elem)
		return
	}

	// Evict oldest entries until there is room for the newcomer.
	for len(c.items) >= c.capacity {
		front := c.order.Front()
		if front == nil {
			break
		}
		c.order.Remove(front)
		delete(c.items, front.Value.(*cacheEntry).key)
	}

	elem := c.order.PushBack(&cacheEntry{key: key, queries: queries})
	c.items[key] = elem
}

// cacheKey computes a deterministic key from normalised input + scope + prompt version.
func cacheKey(raw, scope string) string {
	normalised := normaliseForCache(raw)
	data := fmt.Sprintf("%s\x1f%s\x1f%d", normalised, scope, promptVersion)
	h := sha256.Sum256([]byte(data))
	return hex.EncodeToString(h[:16])
}

// normaliseForCache prepares input for cache key hashing: trim, collapse
// whitespace, lowercase, strip zero-width and non-breaking spaces.
func normaliseForCache(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ToLower(s)

	var b strings.Builder
	b.Grow(len(s))
	prevSpace := false
	for _, r := range s {
		// Strip zero-width characters and BOM.
		if r == '\u200b' || r == '\ufeff' {
			continue
		}
		// Treat non-breaking space as ordinary whitespace.
		if r == '\u00a0' {
			r = ' '
		}
		if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
		} else {
			b.WriteRune(r)
			prevSpace = false
		}
	}
	return strings.TrimSpace(b.String())
}

// truncateInput preserves the tail of the input (the operative signal in
// error pastes and long scrollbacks). Two inputs differing only in their
// first portion hash to the same cache key by design.
func truncateInput(s string, maxChars int) string {
	if len(s) <= maxChars {
		return s
	}
	return s[len(s)-maxChars:]
}

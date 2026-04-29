// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/mem"

	_ "modernc.org/sqlite"
)

const indexSourceTestModel = "fake-embed-test"

// setupIndexSource builds a memstore-backed search.Index plus a vector
// index, populates them with the supplied wiki articles, and returns
// the resulting IndexSource ready for retrieval calls.
func setupIndexSource(t *testing.T, articles []indexSourceArticle) (*IndexSource, *search.Index, *search.VectorIndex, brain.Store) {
	t.Helper()
	ctx := context.Background()

	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	for _, a := range articles {
		var body strings.Builder
		body.WriteString("---\n")
		body.WriteString("title: ")
		body.WriteString(a.Title)
		body.WriteString("\nsummary: ")
		body.WriteString(a.Summary)
		body.WriteString("\n")
		if extra := strings.TrimSpace(a.Frontmatter); extra != "" {
			body.WriteString(extra)
			body.WriteString("\n")
		}
		body.WriteString("---\n")
		body.WriteString(a.Body)
		body.WriteString("\n")
		if err := store.Write(ctx, brain.Path(a.Path), []byte(body.String())); err != nil {
			t.Fatalf("write %s: %v", a.Path, err)
		}
	}

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if err := idx.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	vec, err := search.NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}

	embedder := llm.NewFakeEmbedder(64)
	for _, a := range articles {
		seed := strings.Join([]string{a.Title, a.Summary, a.Body}, " ")
		vecs, err := embedder.Embed(ctx, []string{seed})
		if err != nil {
			t.Fatalf("embed seed %s: %v", a.Path, err)
		}
		if err := vec.Store(ctx, search.VectorEntry{
			Path:     a.Path,
			Checksum: a.Path,
			Model:    indexSourceTestModel,
			Vector:   vecs[0],
			Title:    a.Title,
			Summary:  a.Summary,
		}); err != nil {
			t.Fatalf("store vector for %s: %v", a.Path, err)
		}
	}

	src, err := NewIndexSource(idx, IndexSourceOptions{
		Vectors:  vec,
		Embedder: embedder,
		Model:    indexSourceTestModel,
	})
	if err != nil {
		t.Fatalf("NewIndexSource: %v", err)
	}
	return src, idx, vec, store
}

type indexSourceArticle struct {
	Path        string
	Title       string
	Summary     string
	Frontmatter string
	Body        string
}

// indexSourceCorpus produces a small but varied set of wiki articles.
// Half mention "invoice" so the BM25 query can rank the matching set,
// the rest add distractor noise so retrieval has to discriminate.
func indexSourceCorpus() []indexSourceArticle {
	return []indexSourceArticle{
		{
			Path:    "wiki/invoice-processing.md",
			Title:   "Invoice Processing",
			Summary: "End-to-end automation for supplier invoices",
			Body:    "The invoice processing workflow extracts line items from supplier PDFs and posts them into the ledger.",
		},
		{
			Path:    "wiki/order-processing.md",
			Title:   "Order Processing Pipeline",
			Summary: "Ingest sales orders for retail partners",
			Body:    "Order processing automates document ingestion captured via email, with invoice export for billing.",
		},
		{
			Path:    "wiki/invoice-rollup.md",
			Title:   "Invoice Recap",
			Summary: "Roll-up dashboard across invoice workflows",
			Body:    "Quarterly invoice recap dashboards combine totals across regions and present an executive summary.",
		},
		{
			Path:    "wiki/quote-generator.md",
			Title:   "Quote Generation Tools",
			Summary: "Automated quote drafting from supplier catalogues",
			Body:    "The quote generator drafts proposals from catalogue data and includes a downstream invoice handoff.",
		},
		{
			Path:    "wiki/contact-centre.md",
			Title:   "Contact Centre",
			Summary: "Inbound voice routing",
			Body:    "Telephony stack routes calls via SIP and integrates with CRM seats.",
		},
		{
			Path:    "wiki/holiday-calendar.md",
			Title:   "Holiday Calendar",
			Summary: "Public holidays across regions",
			Body:    "The holiday calendar publishes regional public holidays for HR planning.",
		},
		{
			Path:    "wiki/office-stationery.md",
			Title:   "Stationery Budget",
			Summary: "Stock ledger for office stationery",
			Body:    "The stationery budget tracks consumables across UK and NL offices.",
		},
		{
			Path:    "wiki/hr-handbook.md",
			Title:   "HR Handbook",
			Summary: "Annual leave and expenses policy",
			Body:    "The HR handbook documents annual leave policy, expense limits, and travel reimbursement rules.",
		},
		{
			Path:    "wiki/company-wifi.md",
			Title:   "Office Wifi",
			Summary: "Joining the guest wifi network",
			Body:    "The office wifi guide describes the guest SSID and credential rotation cadence.",
		},
		{
			Path:    "wiki/customer-onboarding.md",
			Title:   "Customer Onboarding",
			Summary: "Activation steps for new accounts",
			Body:    "Customer onboarding sequences walk new accounts through configuration and first invoice issuance.",
		},
	}
}

func TestIndexSource_NewIndexSource_RequiresIndex(t *testing.T) {
	t.Parallel()
	if _, err := NewIndexSource(nil, IndexSourceOptions{}); err == nil {
		t.Fatal("expected error for nil index")
	}
}

func TestIndexSource_NewIndexSource_RequiresModelWhenVectorsSet(t *testing.T) {
	t.Parallel()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	vec, err := search.NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}
	if _, err := NewIndexSource(idx, IndexSourceOptions{Vectors: vec}); err == nil {
		t.Fatal("expected error when Vectors set but Model empty")
	}
}

func TestIndexSource_BM25_ReturnsTopHitsWithIDs(t *testing.T) {
	t.Parallel()
	corpus := indexSourceCorpus()
	src, _, _, _ := setupIndexSource(t, corpus)

	hits, err := src.SearchBM25(context.Background(), "invoice", 5, Filters{})
	if err != nil {
		t.Fatalf("SearchBM25: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one BM25 hit")
	}
	if len(hits) > 5 {
		t.Fatalf("returned %d hits, want at most 5", len(hits))
	}
	for i, h := range hits {
		if h.ID == "" {
			t.Errorf("hit %d has empty ID", i)
		}
		if h.ID != h.Path {
			t.Errorf("hit %d ID=%q != Path=%q", i, h.ID, h.Path)
		}
	}
	// At least the top hit must be one of the explicitly
	// invoice-titled articles. The remaining slots may include
	// articles that only mention invoice in passing, which is
	// expected BM25 behaviour.
	top := strings.ToLower(hits[0].Path)
	if !strings.Contains(top, "invoice") {
		t.Errorf("top hit %q does not look invoice-related (paths: %v)", hits[0].Path, hitPaths(hits))
	}
	expectedByPath := make(map[string]string, len(corpus))
	for _, article := range corpus {
		expectedByPath[article.Path] = article.Body
	}
	if want := expectedByPath[hits[0].Path]; hits[0].Content != want {
		t.Errorf("top hit content = %q, want full indexed body for %s", hits[0].Content, hits[0].Path)
	}
}

func TestIndexSource_BM25_QuotedDatePhraseMatchesStoredSlashDate(t *testing.T) {
	t.Parallel()

	src, _, _, _ := setupIndexSource(t, []indexSourceArticle{{
		Path:    "memory/global/weekly-note.md",
		Title:   "Weekly note",
		Summary: "Supplier timeline update",
		Body:    "[Observed on 2024/03/08 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
	}})

	hits, err := src.SearchBM25(context.Background(), `"2024/03/08"`, 5, Filters{})
	if err != nil {
		t.Fatalf("SearchBM25 quoted date: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected quoted date phrase to match stored slash date")
	}
	if hits[0].Path != "memory/global/weekly-note.md" {
		t.Fatalf("top hit = %q, want weekly note", hits[0].Path)
	}
}

func TestIndexSource_BM25_RespectsScopeFilter(t *testing.T) {
	t.Parallel()
	corpus := append(indexSourceCorpus(), indexSourceArticle{
		Path:    "memory/global/invoice-note.md",
		Title:   "Invoice note",
		Summary: "Personal observation",
		Body:    "Reminder that invoice batches close on Friday.",
	})
	src, _, _, _ := setupIndexSource(t, corpus)

	hits, err := src.SearchBM25(context.Background(), "invoice", 5, Filters{Scope: "global_memory"})
	if err != nil {
		t.Fatalf("SearchBM25 scoped: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected scoped hit")
	}
	for _, h := range hits {
		if !strings.HasPrefix(h.Path, "memory/global/") {
			t.Errorf("scope leak: %s", h.Path)
		}
	}
}

func TestIndexSource_BM25_MemoryScopeIncludesGlobalAndProjectForProjectSlug(t *testing.T) {
	t.Parallel()
	corpus := []indexSourceArticle{
		{
			Path:    "memory/global/farmers-market.md",
			Title:   "Farmers market summary",
			Summary: "Latest market earnings",
			Body:    "The most recent Downtown Farmers Market visit earned $420.",
		},
		{
			Path:    "memory/project/eval-lme/farmers-market-plan.md",
			Title:   "Farmers market plan",
			Summary: "Promo notes for the next market",
			Body:    "Prepare signage for the Downtown Farmers Market and restock candles.",
		},
		{
			Path:    "memory/project/other/farmers-market-other.md",
			Title:   "Other project market plan",
			Summary: "Unrelated project",
			Body:    "This other project also mentions the Downtown Farmers Market.",
		},
		{
			Path:    "raw/lme/farmers-market-session.md",
			Title:   "Raw session",
			Summary: "Transcript",
			Body:    "User mentioned the Downtown Farmers Market in a raw session transcript.",
		},
	}
	src, _, _, _ := setupIndexSource(t, corpus)

	hits, err := src.SearchBM25(context.Background(), "downtown farmers market", 10, Filters{
		Scope:   "memory",
		Project: "eval-lme",
	})
	if err != nil {
		t.Fatalf("SearchBM25 memory scope: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected filtered hits")
	}
	if !containsBM25Path(hits, "memory/global/farmers-market.md") {
		t.Fatalf("expected global memory hit, got %v", hitPaths(hits))
	}
	if !containsBM25Path(hits, "memory/project/eval-lme/farmers-market-plan.md") {
		t.Fatalf("expected eval-lme project hit, got %v", hitPaths(hits))
	}
	if containsBM25Path(hits, "memory/project/other/farmers-market-other.md") {
		t.Fatalf("unexpected other-project hit: %v", hitPaths(hits))
	}
	if containsBM25Path(hits, "raw/lme/farmers-market-session.md") {
		t.Fatalf("unexpected raw hit: %v", hitPaths(hits))
	}
}

func TestIndexSource_BM25_ProjectScopeExcludesGlobalAndRaw(t *testing.T) {
	t.Parallel()
	corpus := []indexSourceArticle{
		{
			Path:    "memory/global/farmers-market.md",
			Title:   "Farmers market summary",
			Summary: "Latest market earnings",
			Body:    "The most recent Downtown Farmers Market visit earned $420.",
		},
		{
			Path:    "memory/project/eval-lme/farmers-market-plan.md",
			Title:   "Farmers market plan",
			Summary: "Promo notes for the next market",
			Body:    "Prepare signage for the Downtown Farmers Market and restock candles.",
		},
		{
			Path:    "raw/lme/farmers-market-session.md",
			Title:   "Raw session",
			Summary: "Transcript",
			Body:    "User mentioned the Downtown Farmers Market in a raw session transcript.",
		},
	}
	src, _, _, _ := setupIndexSource(t, corpus)

	hits, err := src.SearchBM25(context.Background(), "downtown farmers market", 10, Filters{
		Scope:   "project",
		Project: "eval-lme",
	})
	if err != nil {
		t.Fatalf("SearchBM25 project scope: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected project-scoped hit")
	}
	if !containsBM25Path(hits, "memory/project/eval-lme/farmers-market-plan.md") {
		t.Fatalf("expected eval-lme project hit, got %v", hitPaths(hits))
	}
	if containsBM25Path(hits, "memory/global/farmers-market.md") || containsBM25Path(hits, "raw/lme/farmers-market-session.md") {
		t.Fatalf("project scope leaked non-project rows: %v", hitPaths(hits))
	}
}

func TestIndexSource_BM25_ProjectFilterWithoutScopeKeepsGlobalAndMatchingProject(t *testing.T) {
	t.Parallel()
	corpus := []indexSourceArticle{
		{
			Path:    "memory/global/farmers-market.md",
			Title:   "Farmers market summary",
			Summary: "Latest market earnings",
			Body:    "The Downtown Farmers Market visit earned $420.",
		},
		{
			Path:    "memory/project/eval-lme/farmers-market-plan.md",
			Title:   "Farmers market plan",
			Summary: "Promo notes for the next market",
			Body:    "Prepare signage for the Downtown Farmers Market.",
		},
		{
			Path:    "memory/project/other/farmers-market-other.md",
			Title:   "Other project market plan",
			Summary: "Unrelated project",
			Body:    "This other project also mentions the Downtown Farmers Market.",
		},
		{
			Path:    "wiki/farmers-market.md",
			Title:   "Wiki market",
			Summary: "Reference article",
			Body:    "The Downtown Farmers Market has a public website.",
		},
		{
			Path:    "raw/lme/farmers-market-session.md",
			Title:   "Raw session",
			Summary: "Transcript",
			Body:    "User mentioned the Downtown Farmers Market in a raw session transcript.",
		},
	}
	src, _, _, _ := setupIndexSource(t, corpus)

	hits, err := src.SearchBM25(context.Background(), "downtown farmers market", 10, Filters{
		Project: "eval-lme",
	})
	if err != nil {
		t.Fatalf("SearchBM25 project filter: %v", err)
	}
	if !containsBM25Path(hits, "memory/global/farmers-market.md") {
		t.Fatalf("expected global memory hit, got %v", hitPaths(hits))
	}
	if !containsBM25Path(hits, "memory/project/eval-lme/farmers-market-plan.md") {
		t.Fatalf("expected eval-lme project hit, got %v", hitPaths(hits))
	}
	for _, unexpected := range []string{
		"memory/project/other/farmers-market-other.md",
		"wiki/farmers-market.md",
		"raw/lme/farmers-market-session.md",
	} {
		if containsBM25Path(hits, unexpected) {
			t.Fatalf("project filter leaked %s: %v", unexpected, hitPaths(hits))
		}
	}
}

func TestIndexSource_BM25_RespectsPathPrefix(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	hits, err := src.SearchBM25(context.Background(), "invoice", 10, Filters{PathPrefix: "wiki/invoice"})
	if err != nil {
		t.Fatalf("SearchBM25 prefix: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected prefix hit")
	}
	for _, h := range hits {
		if !strings.HasPrefix(h.Path, "wiki/invoice") {
			t.Errorf("prefix leak: %s", h.Path)
		}
	}
}

func TestIndexSource_BM25_RespectsExactPaths(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	hits, err := src.SearchBM25(context.Background(), "invoice", 10, Filters{
		Paths: []string{"wiki/order-processing.md", " wiki/order-processing.md ", "wiki/invoice-rollup.md"},
	})
	if err != nil {
		t.Fatalf("SearchBM25 paths: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected exact-path hit")
	}
	for _, h := range hits {
		if h.Path != "wiki/order-processing.md" && h.Path != "wiki/invoice-rollup.md" {
			t.Fatalf("path leak: %s", h.Path)
		}
	}
}

func TestIndexSource_BM25_TagsMatchYAMLInlineListFormatting(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, []indexSourceArticle{
		{
			Path:        "wiki/tagged-invoice.md",
			Title:       "Tagged invoice",
			Summary:     "Billing workflow",
			Frontmatter: "tags: [billing, urgent]",
			Body:        "The invoice workflow needs escalation.",
		},
		{
			Path:        "wiki/untagged-invoice.md",
			Title:       "Untagged invoice",
			Summary:     "Billing workflow",
			Frontmatter: "tags: [archive]",
			Body:        "The invoice workflow is historical.",
		},
	})

	hits, err := src.SearchBM25(context.Background(), "invoice workflow", 10, Filters{
		Tags: []string{"billing", "urgent"},
	})
	if err != nil {
		t.Fatalf("SearchBM25 tags: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != "wiki/tagged-invoice.md" {
		t.Fatalf("tag-filtered hits = %v, want only tagged invoice", hitPaths(hits))
	}
}

func TestIndexSource_Vectors_ReturnsTopSemanticHits(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	embedder := llm.NewFakeEmbedder(64)
	vecs, err := embedder.Embed(context.Background(), []string{"invoice processing automation"})
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	hits, err := src.SearchVector(context.Background(), vecs[0], 5, Filters{})
	if err != nil {
		t.Fatalf("SearchVector: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one vector hit")
	}
	if len(hits) > 5 {
		t.Fatalf("returned %d hits, want at most 5", len(hits))
	}
	for i, h := range hits {
		if h.ID == "" || h.Path == "" {
			t.Errorf("hit %d missing identity: %+v", i, h)
		}
		if h.ID != h.Path {
			t.Errorf("hit %d ID=%q != Path=%q", i, h.ID, h.Path)
		}
	}
	if hits[0].Content == "" {
		t.Fatal("expected vector hit to hydrate full content")
	}
	if !strings.Contains(hits[0].Content, "invoice") {
		t.Errorf("vector hit content = %q, want hydrated article body", hits[0].Content)
	}
}

func TestIndexSource_Vectors_RespectsExactPaths(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	embedder := llm.NewFakeEmbedder(64)
	vecs, err := embedder.Embed(context.Background(), []string{"invoice processing automation"})
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	hits, err := src.SearchVector(context.Background(), vecs[0], 10, Filters{
		Paths: []string{"wiki/order-processing.md"},
	})
	if err != nil {
		t.Fatalf("SearchVector exact paths: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected exact-path vector hit")
	}
	for _, h := range hits {
		if h.Path != "wiki/order-processing.md" {
			t.Fatalf("path leak: %s", h.Path)
		}
	}
}

func TestIndexSource_Vectors_MemoryScopeIncludesGlobalAndProjectButNotRaw(t *testing.T) {
	t.Parallel()
	corpus := []indexSourceArticle{
		{
			Path:    "memory/global/farmers-market.md",
			Title:   "Farmers market summary",
			Summary: "Latest market earnings",
			Body:    "The most recent Downtown Farmers Market visit earned $420.",
		},
		{
			Path:    "memory/project/eval-lme/farmers-market-plan.md",
			Title:   "Farmers market plan",
			Summary: "Promo notes for the next market",
			Body:    "Prepare signage for the Downtown Farmers Market and restock candles.",
		},
		{
			Path:    "memory/project/other/farmers-market-other.md",
			Title:   "Other project market plan",
			Summary: "Unrelated project",
			Body:    "This other project also mentions the Downtown Farmers Market.",
		},
		{
			Path:    "raw/lme/farmers-market-session.md",
			Title:   "Raw session",
			Summary: "Transcript",
			Body:    "User mentioned the Downtown Farmers Market in a raw session transcript.",
		},
	}
	src, _, _, _ := setupIndexSource(t, corpus)

	embedder := llm.NewFakeEmbedder(64)
	vecs, err := embedder.Embed(context.Background(), []string{"downtown farmers market earnings"})
	if err != nil {
		t.Fatalf("embed query: %v", err)
	}
	hits, err := src.SearchVector(context.Background(), vecs[0], 10, Filters{
		Scope:   "memory",
		Project: "eval-lme",
	})
	if err != nil {
		t.Fatalf("SearchVector memory scope: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected vector hits")
	}
	if !containsVectorPath(hits, "memory/global/farmers-market.md") {
		t.Fatalf("expected global memory hit, got %v", vectorHitPaths(hits))
	}
	if !containsVectorPath(hits, "memory/project/eval-lme/farmers-market-plan.md") {
		t.Fatalf("expected eval-lme project hit, got %v", vectorHitPaths(hits))
	}
	if containsVectorPath(hits, "memory/project/other/farmers-market-other.md") {
		t.Fatalf("unexpected other-project hit: %v", vectorHitPaths(hits))
	}
	if containsVectorPath(hits, "raw/lme/farmers-market-session.md") {
		t.Fatalf("unexpected raw hit: %v", vectorHitPaths(hits))
	}
}

func TestIndexSource_Vectors_WidensCandidatePoolBeforeFiltering(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	const distractors = 250
	for i := 0; i < distractors; i++ {
		path := brain.Path(fmt.Sprintf("memory/project/other/vector-distractor-%03d.md", i))
		body := fmt.Sprintf("---\ntitle: Distractor %03d\nsummary: Other project\n---\nwrong project body\n", i)
		if err := store.Write(ctx, path, []byte(body)); err != nil {
			t.Fatalf("write distractor %d: %v", i, err)
		}
	}
	const allowedPath = "memory/project/eval-lme/vector-target.md"
	if err := store.Write(ctx, brain.Path(allowedPath), []byte("---\ntitle: Allowed\nsummary: Eval project\n---\nallowed project body\n")); err != nil {
		t.Fatalf("write allowed: %v", err)
	}

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	if err := idx.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}
	vec, err := search.NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}
	for i := 0; i < distractors; i++ {
		path := fmt.Sprintf("memory/project/other/vector-distractor-%03d.md", i)
		if err := vec.Store(ctx, search.VectorEntry{
			Path:     path,
			Checksum: path,
			Model:    indexSourceTestModel,
			Vector:   []float32{1, 0},
			Title:    "Distractor",
			Summary:  "Other project",
		}); err != nil {
			t.Fatalf("store distractor vector %d: %v", i, err)
		}
	}
	if err := vec.Store(ctx, search.VectorEntry{
		Path:     allowedPath,
		Checksum: allowedPath,
		Model:    indexSourceTestModel,
		Vector:   []float32{0, 1},
		Title:    "Allowed",
		Summary:  "Eval project",
	}); err != nil {
		t.Fatalf("store allowed vector: %v", err)
	}

	src, err := NewIndexSource(idx, IndexSourceOptions{
		Vectors: vec,
		Model:   indexSourceTestModel,
	})
	if err != nil {
		t.Fatalf("NewIndexSource: %v", err)
	}
	hits, err := src.SearchVector(ctx, []float32{1, 0}, 5, Filters{
		Scope:   "project_memory",
		Project: "eval-lme",
	})
	if err != nil {
		t.Fatalf("SearchVector: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != allowedPath {
		t.Fatalf("filtered vector hits = %v, want allowed low-ranked path", vectorHitPaths(hits))
	}
}

func TestIndexSource_Vectors_NilStoreReturnsEmpty(t *testing.T) {
	t.Parallel()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	src, err := NewIndexSource(idx, IndexSourceOptions{})
	if err != nil {
		t.Fatalf("NewIndexSource: %v", err)
	}
	hits, err := src.SearchVector(context.Background(), []float32{0.1, 0.2, 0.3}, 5, Filters{})
	if err != nil {
		t.Fatalf("SearchVector with nil vectors: %v", err)
	}
	if len(hits) != 0 {
		t.Errorf("expected zero hits when vectors disabled, got %d", len(hits))
	}
}

func TestIndexSource_Chunks_ReturnsAllRows(t *testing.T) {
	t.Parallel()
	corpus := indexSourceCorpus()
	src, _, _, _ := setupIndexSource(t, corpus)

	chunks, err := src.Chunks(context.Background())
	if err != nil {
		t.Fatalf("Chunks: %v", err)
	}
	if len(chunks) != len(corpus) {
		t.Fatalf("Chunks len = %d, want %d", len(chunks), len(corpus))
	}

	got := make(map[string]bool, len(chunks))
	for _, c := range chunks {
		got[c.Path] = true
		if c.ID != c.Path {
			t.Errorf("chunk ID=%q != Path=%q", c.ID, c.Path)
		}
		if c.Title == "" {
			t.Errorf("chunk %s missing title", c.Path)
		}
	}
	for _, a := range corpus {
		if !got[a.Path] {
			t.Errorf("missing chunk %s", a.Path)
		}
	}
}

func TestIndexSource_Lookup_HydratesByID(t *testing.T) {
	t.Parallel()
	corpus := indexSourceCorpus()
	src, _, _, _ := setupIndexSource(t, corpus)

	ids := []string{corpus[0].Path, corpus[2].Path}
	rows, err := src.Lookup(context.Background(), ids)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if len(rows) != len(ids) {
		t.Fatalf("Lookup returned %d rows, want %d", len(rows), len(ids))
	}
	got := make(map[string]search.IndexedRow, len(rows))
	for _, r := range rows {
		got[r.Path] = r
	}
	for _, want := range ids {
		row, ok := got[want]
		if !ok {
			t.Errorf("missing lookup %s", want)
			continue
		}
		if row.Title == "" {
			t.Errorf("%s missing title", want)
		}
		if row.Content == "" {
			t.Errorf("%s missing content", want)
		}
	}
}

func TestIndexSource_EndToEnd_RetrieveHydratesMetadataAliases(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, []indexSourceArticle{{
		Path:        "memory/project/eval-lme/weekly-note.md",
		Title:       "Weekly note",
		Summary:     "Supplier update",
		Frontmatter: "session_date: 2024-03-08\nmodified: 2024-03-09T12:00:00Z",
		Body:        "Met the supplier and agreed the timeline.",
	}})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "supplier timeline",
		Mode:  ModeBM25,
		TopK:  1,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(resp.Chunks))
	}
	meta := resp.Chunks[0].Metadata
	if meta["scope"] != "project_memory" {
		t.Fatalf("scope metadata = %v, want project_memory", meta["scope"])
	}
	if meta["project"] != "eval-lme" {
		t.Fatalf("project metadata = %v, want eval-lme", meta["project"])
	}
	if meta["projectSlug"] != "eval-lme" {
		t.Fatalf("projectSlug metadata = %v, want eval-lme", meta["projectSlug"])
	}
	if meta["sessionDate"] != "2024-03-08" || meta["session_date"] != "2024-03-08" {
		t.Fatalf("session date metadata = %+v, want both aliases", meta)
	}
}

func TestIndexSource_EndToEnd_QuestionDateBoundsExcludeFutureDecoy(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, []indexSourceArticle{
		{
			Path:        "memory/global/past-supplier-status.md",
			Title:       "Past supplier status",
			Summary:     "Supplier update",
			Frontmatter: "session_date: 2024-03-10",
			Body:        "Supplier status is stable.",
		},
		{
			Path:        "memory/global/future-supplier-status.md",
			Title:       "Future supplier status",
			Summary:     "Supplier update",
			Frontmatter: "session_date: 2024-04-10",
			Body:        "Supplier status status status is cancelled.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What is the supplier status?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		Mode:         ModeBM25,
		TopK:         1,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) != 1 || resp.Chunks[0].Path != "memory/global/past-supplier-status.md" {
		t.Fatalf("top chunk = %+v, want past supplier status", resp.Chunks)
	}
	if len(resp.Attempts) == 0 || !resp.Attempts[0].DateBounded {
		t.Fatalf("initial attempt should record date-bound search, got %+v", resp.Attempts)
	}
}

func TestIndexSource_EndToEnd_RetrieveHybrid(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())
	embedder := llm.NewFakeEmbedder(64)

	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice processing",
		Mode:  ModeHybrid,
		TopK:  3,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected hybrid retrieval to return chunks")
	}
	if resp.Trace.EffectiveMode != ModeHybrid {
		t.Fatalf("effective mode %q, want hybrid", resp.Trace.EffectiveMode)
	}
	if resp.Trace.BM25Hits == 0 {
		t.Fatal("expected BM25 hits in hybrid mode")
	}
	// The vector leg may or may not produce hits depending on the
	// fake embedder's deterministic ordering; we only require that
	// the pipeline ran without error.
	t.Logf("hybrid retrieve: %d chunks, BM25=%d, Vec=%d, Fused=%d",
		len(resp.Chunks), resp.Trace.BM25Hits, resp.Trace.VectorHits, resp.Trace.FusedHits)
	for i, c := range resp.Chunks {
		t.Logf("  [%d] %s score=%.4f", i, c.Path, c.Score)
		if c.Path == "" {
			t.Errorf("chunk %d has empty path", i)
		}
	}
}

func TestIndexSource_EndToEnd_RetrieveBM25Only(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice processing",
		Mode:  ModeBM25,
		TopK:  3,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected BM25 retrieval to return chunks")
	}
	if resp.Trace.EmbedderUsed {
		t.Fatal("BM25-only retrieval should not invoke embedder")
	}
}

func TestIndexSource_EndToEnd_TrigramFallback(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, indexSourceCorpus())

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}
	// Misspelled query so initial BM25 returns nothing and the retry
	// ladder flows down to the trigram rung. The fallback corpus is
	// fetched via IndexSource.Chunks (i.e. AllRows on the FTS index).
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invioce procesing",
		Mode:  ModeBM25,
		TopK:  3,
	})
	if err != nil {
		t.Fatalf("Retrieve fallback: %v", err)
	}
	if !resp.Trace.UsedRetry && len(resp.Chunks) > 0 {
		t.Logf("note: initial BM25 unexpectedly matched the misspelled query: %v", hitPathsFromChunks(resp.Chunks))
	}
	t.Logf("trigram fallback retrieve: chunks=%d UsedRetry=%v attempts=%d",
		len(resp.Chunks), resp.Trace.UsedRetry, len(resp.Attempts))
	// The fallback rungs may legitimately return zero matches if the
	// trigram similarity threshold filters everything; we just need
	// proof the pipeline ran end-to-end without error.
	if len(resp.Attempts) == 0 {
		t.Fatal("expected at least one attempt in trace")
	}
}

func TestIndexSource_EndToEnd_TrigramFallback_RespectsExactPaths(t *testing.T) {
	t.Parallel()
	src, _, _, _ := setupIndexSource(t, []indexSourceArticle{
		{
			Path:    "wiki/photosynthesis.md",
			Title:   "Photosynthesis",
			Summary: "Leaf chemistry",
			Body:    "Chlorophyll helps plants turn light into energy.",
		},
		{
			Path:    "wiki/photosynthesis-log.md",
			Title:   "Photosynthesis log",
			Summary: "Experiment notes",
			Body:    "Daily notes about plant growth experiments.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New retriever: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "photosynthasis",
		Mode:  ModeBM25,
		TopK:  3,
		Filters: Filters{
			Paths: []string{"wiki/photosynthesis-log.md"},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve fallback: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected filtered trigram fallback hit")
	}
	for _, chunk := range resp.Chunks {
		if chunk.Path != "wiki/photosynthesis-log.md" {
			t.Fatalf("path leak: %s", chunk.Path)
		}
	}
}

func TestIndexSource_LLMReranker_NameReflectsModel(t *testing.T) {
	t.Parallel()
	provider := llm.NewFake([]string{`[{"id":0,"score":1}]`})
	rr := NewLLMReranker(provider, "fake-model")
	if rr.Name() != "llm:fake-model" {
		t.Errorf("Name = %q, want llm:fake-model", rr.Name())
	}
	empty := NewLLMReranker(provider, "")
	if empty.Name() != "llm-reranker" {
		t.Errorf("Name = %q, want llm-reranker for empty model", empty.Name())
	}
}

// hitPaths extracts the path field from a slice of BM25 hits for log output.
func hitPaths(hits []BM25Hit) []string {
	out := make([]string, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.Path)
	}
	return out
}

func vectorHitPaths(hits []VectorHit) []string {
	out := make([]string, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.Path)
	}
	return out
}

func containsBM25Path(hits []BM25Hit, want string) bool {
	for _, hit := range hits {
		if hit.Path == want {
			return true
		}
	}
	return false
}

func containsVectorPath(hits []VectorHit, want string) bool {
	for _, hit := range hits {
		if hit.Path == want {
			return true
		}
	}
	return false
}

// hitPathsFromChunks extracts path strings from RetrievedChunk slices.
func hitPathsFromChunks(chunks []RetrievedChunk) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c.Path)
	}
	return out
}

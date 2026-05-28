// SPDX-License-Identifier: Apache-2.0

package graph

// DocumentEdgeType enumerates the 9 pre-computed relationship types.
type DocumentEdgeType string

const (
	EdgeSemanticSimilarity DocumentEdgeType = "semantic_similarity"
	EdgeSharedTag          DocumentEdgeType = "shared_tag"
	EdgeSharedFolder       DocumentEdgeType = "shared_folder"
	EdgeSameSession        DocumentEdgeType = "same_session"
	EdgeSupersedes         DocumentEdgeType = "supersedes"
	EdgeSessionEpisode     DocumentEdgeType = "session_episode"
	EdgeEpisodeHeuristic   DocumentEdgeType = "episode_heuristic"
	EdgeDocumentOntology   DocumentEdgeType = "document_ontology"
	EdgeWikilink           DocumentEdgeType = "wikilink"
)

// EntityType classifies documents by their path prefix.
type EntityType string

const (
	EntityMemory     EntityType = "memory"
	EntityArticle    EntityType = "article"
	EntityEpisode    EntityType = "episode"
	EntityHeuristic  EntityType = "heuristic"
	EntityProcedural EntityType = "procedural"
	EntityOntology   EntityType = "ontology"
	EntityIngested   EntityType = "ingested"
)

// EmbeddingDim selects the embedding column.
type EmbeddingDim int

const (
	Dim1024 EmbeddingDim = 1024
	Dim3072 EmbeddingDim = 3072
)

type GraphNode struct {
	ID            string         `json:"id"`
	Path          string         `json:"path"`
	Label         string         `json:"label"`
	EntityType    EntityType     `json:"entityType"`
	Scope         string         `json:"scope"`
	Size          int            `json:"size"`
	ModTime       string         `json:"modTime"`
	ChunkCount    int            `json:"chunkCount"`
	Metadata      map[string]any `json:"metadata"`
	OntologyType  string         `json:"ontologyType,omitempty"`
	OntologyLabel string         `json:"ontologyLabel,omitempty"`
}

type GraphEdge struct {
	Source   string           `json:"source"`
	Target   string           `json:"target"`
	EdgeType DocumentEdgeType `json:"edgeType"`
	Weight   float64          `json:"weight"`
	Label    string           `json:"label,omitempty"`
}

type GraphCommunity struct {
	ID      string   `json:"id"`
	NodeIDs []string `json:"nodeIds"`
	Label   string   `json:"label,omitempty"`
}

type GraphMeta struct {
	TotalNodes       int            `json:"totalNodes"`
	TotalEdges       int            `json:"totalEdges"`
	EntityTypeCounts map[string]int `json:"entityTypeCounts"`
	EdgeTypeCounts   map[string]int `json:"edgeTypeCounts"`
	Truncated        bool           `json:"truncated"`
}

type GraphResponse struct {
	Nodes       []GraphNode      `json:"nodes"`
	Edges       []GraphEdge      `json:"edges"`
	Communities []GraphCommunity `json:"communities"`
	Meta        GraphMeta        `json:"meta"`
}

type GraphQueryOptions struct {
	MinWeight   float64
	EdgeTypes   []DocumentEdgeType
	EntityTypes []EntityType
	MaxNodes    int
	MaxEdges    int
}

type StatsResponse struct {
	DocumentCount    int            `json:"documentCount"`
	ChunkCount       int            `json:"chunkCount"`
	EdgeCount        int            `json:"edgeCount"`
	EntityTypeCounts map[string]int `json:"entityTypeCounts"`
	EdgeTypeCounts   map[string]int `json:"edgeTypeCounts"`
	LastActivityAt   *string        `json:"lastActivityAt"`
}

type UpsertEdge struct {
	SourceDocID string
	TargetDocID string
	EdgeType    DocumentEdgeType
	Weight      float64
	Label       string
}

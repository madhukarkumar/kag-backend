"""Shared Pydantic models for the SingleStore Knowledge Graph Search application."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Entity(BaseModel):
    """Entity model representing items in the knowledge graph."""
    id: int = Field(alias="entity_id")
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., pattern="^(PERSON|ORGANIZATION|LOCATION|TECHNOLOGY|CONCEPT|EVENT|PRODUCT)$")
    description: Optional[str] = Field(
        None,
        min_length=10,
        max_length=2000,
        description="Detailed description of the entity"
    )
    aliases: Optional[List[str]] = Field(
        default_factory=list,
        description="Alternative names for the entity"
    )
    metadata: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Metadata about the entity extraction quality"
    )

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id and self.name == other.name

    @property
    def has_valid_description(self) -> bool:
        """Check if the entity has a valid description."""
        if not self.description:
            return False
        return len(self.description.strip()) >= 10

    def merge_with(self, other: 'Entity') -> 'Entity':
        """Merge this entity with another, keeping the best information."""
        if self != other:
            raise ValueError("Can only merge entities with the same ID and name")
        
        # Keep the longer description if both exist, or the non-None one
        if self.description and other.description:
            description = max([self.description, other.description], key=len)
        else:
            description = self.description or other.description

        # Merge and deduplicate aliases
        aliases = list(set(self.aliases + other.aliases))

        # Merge metadata, taking the higher confidence values
        merged_metadata = {
            k: max(self.metadata.get(k, 0.0), other.metadata.get(k, 0.0))
            for k in set(self.metadata.keys()) | set(other.metadata.keys())
        }

        return Entity(
            entity_id=self.id,
            name=self.name,
            category=self.category,
            description=description,
            aliases=aliases,
            metadata=merged_metadata
        )

class Relationship(BaseModel):
    """Relationship model representing connections between entities."""
    source_entity_id: int
    target_entity_id: int
    relation_type: str
    metadata: Optional[Dict] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Search result model containing document content and metadata."""
    doc_id: int
    content: str
    vector_score: float = Field(default=0.0)
    text_score: float = Field(default=0.0)
    combined_score: float = Field(default=0.0)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

class SearchRequest(BaseModel):
    """Search request parameters."""
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    debug: bool = Field(default=False)

class SearchResponse(BaseModel):
    """Search response containing results and metadata."""
    query: str
    results: List[SearchResult]
    generated_response: Optional[str] = None
    execution_time: float

class DocumentStats(BaseModel):
    """Statistics for a document in the knowledge base."""
    doc_id: int
    title: str
    total_chunks: int
    total_entities: int
    total_relationships: int
    created_at: str
    file_type: str
    status: str = Field(default="processed")

class KBStats(BaseModel):
    """Overall knowledge base statistics."""
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    documents: List[DocumentStats]
    last_updated: str

class KBDataResponse(BaseModel):
    """Response model for KB statistics endpoint."""
    stats: KBStats
    execution_time: float

class GraphNode(BaseModel):
    """Node in the knowledge graph visualization."""
    id: str
    name: str
    category: str
    group: int  # For coloring by category
    val: int = Field(default=1)  # For node size

class GraphLink(BaseModel):
    """Link/relationship in the knowledge graph visualization."""
    source: str
    target: str
    type: str
    value: int = Field(default=1)  # For link thickness

class GraphData(BaseModel):
    """Complete graph data for visualization."""
    nodes: List[GraphNode]
    links: List[GraphLink]

class GraphResponse(BaseModel):
    """Response model for graph data endpoint."""
    data: GraphData
    execution_time: float

class ProcessingStatus(BaseModel):
    """Status of document processing."""
    doc_id: int
    file_name: str
    file_path: str
    file_size: int
    current_step: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class ProcessingStatusResponse(BaseModel):
    """API response for processing status."""
    currentStep: str
    errorMessage: Optional[str] = None
    fileName: str

class Document(BaseModel):
    """Document model representing a processed document."""
    id: int = Field(alias="doc_id")
    file_name: str
    file_path: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    """Document chunk model representing a processed document segment."""
    id: int = Field(alias="chunk_id")
    doc_id: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class TaskResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    doc_id: int
    status: str
    message: Optional[str] = None

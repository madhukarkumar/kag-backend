from fastapi import FastAPI, HTTPException, Query, File, UploadFile, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
import yaml
from typing import List, Dict, Optional, Union
from search.engine import RAGQueryEngine
from db import DatabaseConnection
import time
from core.models import (
    SearchRequest, SearchResponse, SearchResult, Entity, 
    Relationship, KBDataResponse, KBStats, DocumentStats, 
    GraphResponse, GraphData, GraphNode, GraphLink, 
    ProcessingStatusResponse
)
from datetime import datetime
from processors.pdf import (
    save_pdf, create_document_record, get_processing_status, 
    cleanup_processing, PDFProcessingError
)
from tasks.pdf_tasks import process_pdf_task
from celery.result import AsyncResult
from api.auth import get_api_key

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

app = FastAPI(
    title="KagSearch API",
    description="""
    API for document processing, search, and knowledge graph functionality.
    
    Features:
    - PDF document upload and processing
    - Real-time processing status tracking
    - Natural language search
    - Knowledge graph visualization
    - Document analytics
    
    For detailed documentation, see /docs/api.md
    """,
    version="0.2.1"
)

# Database connection
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    global db
    try:
        db = DatabaseConnection()
        db.connect()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    global db
    if db:
        db.disconnect()
        logger.info("Database connection closed")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    doc_id: int
    status: str
    message: Optional[str] = None

# In-memory status cache
processing_status_cache: Dict[int, ProcessingStatusResponse] = {}

async def update_status_cache(doc_id: int, status: ProcessingStatusResponse):
    """Update the in-memory status cache"""
    processing_status_cache[doc_id] = status

@app.get("/processing-status/{doc_id}", response_model=ProcessingStatusResponse)
async def get_status(doc_id: int):
    """Get current processing status"""
    try:
        # First check the cache
        if doc_id in processing_status_cache:
            return processing_status_cache[doc_id]
            
        # If not in cache, get from database
        logger.info(f"Getting processing status for doc_id: {doc_id}")
        try:
            status = get_processing_status(doc_id)
            # Update cache
            await update_status_cache(doc_id, status)
            return status
        except Exception as e:
            # If database query fails, return last known status from cache
            if doc_id in processing_status_cache:
                return processing_status_cache[doc_id]
            raise
            
    except ValueError as e:
        logger.error(f"Processing status not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf", status_code=status.HTTP_202_ACCEPTED, response_model=TaskResponse, dependencies=[Depends(get_api_key)])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    try:
        # Save the uploaded file
        logger.info(f"Saving uploaded file: {file.filename}")
        file_content = await file.read()
        file_path = save_pdf(file_content, file.filename)
        
        # Create document record
        doc_id = create_document_record(file.filename, str(file_path), len(file_content))
        logger.info(f"Created document record with ID: {doc_id}")
        
        # Initialize status in cache
        await update_status_cache(doc_id, ProcessingStatusResponse(
            currentStep="started",
            fileName=file.filename
        ))
        
        # Start celery task
        task = process_pdf_task.delay(doc_id)
        logger.info(f"Started Celery task {task.id} for doc_id {doc_id}")
        
        return TaskResponse(
            task_id=task.id,
            doc_id=doc_id,
            status="pending",
            message="Processing started"
        )
        
    except PDFProcessingError as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task-status/{task_id}", dependencies=[Depends(get_api_key)])
async def get_task_status(task_id: str):
    """Get the status of a Celery task"""
    try:
        task = AsyncResult(task_id)
        
        # Get task state and info
        state = task.state
        info = {}
        
        if state == "SUCCESS":
            info = task.result if isinstance(task.result, dict) else {}
        elif state == "FAILURE":
            # Handle database errors specifically
            error = str(task.result)
            if "OperationalError" in error:
                info = {"error": error}
            else:
                info = {"error": str(task.result)}
        else:
            info = task.info if isinstance(task.info, dict) else {}
        
        # Build response based on state
        response = {
            "task_id": task_id,
            "status": state,
            "message": info.get('status', ''),
            "current": info.get('current', 0),
            "total": info.get('total', 100)
        }
        
        # Add error info if failed
        if state == "FAILURE":
            response["error"] = info.get("error", str(task.result))
            
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting task status: {str(e)}"
        )

@app.post("/kag-search", response_model=SearchResponse, dependencies=[Depends(get_api_key)])
async def search_documents(request: SearchRequest):
    """
    Endpoint for document search using natural language queries
    """
    try:
        # Initialize RAG Query Engine and start timing
        start_time = time.time()
        rag_engine = RAGQueryEngine(debug_output=request.debug)
        
        # Execute query
        response = rag_engine.query(
            query_text=request.query,
            top_k=request.top_k
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create SearchResponse
        search_response = SearchResponse(
            query=request.query,
            results=response.results,  
            generated_response=response.generated_response,
            execution_time=execution_time
        )
        
        return search_response
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kbdata", response_model=KBDataResponse, dependencies=[Depends(get_api_key)])
async def get_kb_data():
    """Get knowledge base statistics and document information."""
    start_time = time.time()
    try:
        with DatabaseConnection() as conn:
            # Get total document count and size
            doc_stats_query = """
                SELECT 
                    COUNT(*) as doc_count,
                    SUM(file_size) as total_size,
                    SUM(CASE WHEN current_step = 'completed' THEN 1 ELSE 0 END) as processed_count,
                    SUM(CASE WHEN current_step IN ('started', 'chunking', 'embeddings', 'entities', 'relationships') THEN 1 ELSE 0 END) as processing_count,
                    SUM(CASE WHEN current_step = 'failed' THEN 1 ELSE 0 END) as error_count
                FROM ProcessingStatus
            """
            doc_stats = conn.execute_query(doc_stats_query)[0]
            
            # Get entity stats
            entity_stats_query = """
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(DISTINCT category) as category_count
                FROM Entities
            """
            entity_stats = conn.execute_query(entity_stats_query)[0]
            
            # Get relationship stats
            rel_stats_query = """
                SELECT 
                    COUNT(*) as total_relationships,
                    COUNT(DISTINCT relation_type) as type_count
                FROM Relationships
            """
            rel_stats = conn.execute_query(rel_stats_query)[0]
            
            # Get recent documents
            recent_docs_query = """
                SELECT 
                    p.doc_id,
                    p.file_name as filename,
                    p.file_size,
                    p.created_at as upload_time,
                    p.current_step as status,
                    p.error_message
                FROM ProcessingStatus p
                ORDER BY p.created_at DESC
                LIMIT 10
            """
            recent_docs = conn.execute_query(recent_docs_query)
            
            # Get category distribution
            category_query = """
                SELECT 
                    category,
                    COUNT(*) as count
                FROM Entities
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
                LIMIT 10
            """
            categories = conn.execute_query(category_query)
            
            documents = []
            for doc in recent_docs:
                doc_id = doc[0]
                # Get document-specific stats
                doc_stats_query = """
                    SELECT 
                        COUNT(DISTINCT de.embedding_id) as chunk_count,
                        COUNT(DISTINCT e.entity_id) as entity_count,
                        COUNT(DISTINCT r.relationship_id) as relationship_count
                    FROM Document_Embeddings de
                    LEFT JOIN Relationships r ON r.doc_id = de.doc_id
                    LEFT JOIN (
                        SELECT DISTINCT e.entity_id, r.doc_id
                        FROM Entities e
                        JOIN Relationships r ON r.source_entity_id = e.entity_id OR r.target_entity_id = e.entity_id
                    ) e ON e.doc_id = de.doc_id
                    WHERE de.doc_id = %s
                """
                doc_detail = conn.execute_query(doc_stats_query, (doc_id,))[0]
                
                # Get file extension as file_type
                file_type = os.path.splitext(doc[1])[1].lstrip('.') if doc[1] else 'unknown'
                
                documents.append(DocumentStats(
                    doc_id=doc_id,
                    title=doc[1],  # Using filename as title
                    total_chunks=doc_detail[0],
                    total_entities=doc_detail[1],
                    total_relationships=doc_detail[2],
                    created_at=doc[3].isoformat() if doc[3] else None,
                    file_type=file_type,
                    status=doc[4]
                ))
            
            # Create KBStats
            kb_stats = KBStats(
                total_documents=doc_stats[0],
                total_chunks=sum(d.total_chunks for d in documents),
                total_entities=entity_stats[0],
                total_relationships=rel_stats[0],
                documents=documents,
                last_updated=datetime.now().isoformat()
            )
            
            return KBDataResponse(
                stats=kb_stats,
                execution_time=time.time() - start_time
            )

    except Exception as e:
        logger.error(f"Error getting KB stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve knowledge base statistics: {str(e)}"
        )

@app.get("/graph-data", response_model=GraphResponse, dependencies=[Depends(get_api_key)])
async def get_graph_data():
    """Get knowledge graph visualization data."""
    start_time = time.time()
    try:
        with DatabaseConnection() as conn:
            # Get unique categories and assign group numbers
            category_query = """
                SELECT DISTINCT category 
                FROM Entities 
                WHERE category IS NOT NULL
                ORDER BY category
            """
            categories = conn.execute_query(category_query)
            category_to_group = {cat[0]: idx for idx, cat in enumerate(categories, 1)}
            
            # Get all entities
            entity_query = """
                SELECT 
                    entity_id,
                    name,
                    COALESCE(category, 'unknown') as category,
                    COUNT(DISTINCT r1.relationship_id) + COUNT(DISTINCT r2.relationship_id) as connection_count
                FROM Entities e
                LEFT JOIN Relationships r1 ON e.entity_id = r1.source_entity_id
                LEFT JOIN Relationships r2 ON e.entity_id = r2.target_entity_id
                GROUP BY entity_id, name, category
            """
            entities = conn.execute_query(entity_query)
            
            # Get all relationships
            relationship_query = """
                SELECT 
                    r.source_entity_id,
                    r.target_entity_id,
                    r.relation_type,
                    COUNT(*) as weight
                FROM Relationships r
                GROUP BY source_entity_id, target_entity_id, relation_type
            """
            relationships = conn.execute_query(relationship_query)
            
            # Create nodes
            nodes = [
                GraphNode(
                    id=str(entity[0]),
                    name=entity[1],
                    category=entity[2],
                    group=category_to_group.get(entity[2], 0),
                    val=max(1, int(entity[3]))  # Node size based on connections
                )
                for entity in entities
            ]
            
            # Create links
            links = [
                GraphLink(
                    source=str(rel[0]),
                    target=str(rel[1]),
                    type=rel[2],
                    value=int(rel[3])  # Link thickness based on relationship count
                )
                for rel in relationships
            ]
            
            return GraphResponse(
                data=GraphData(nodes=nodes, links=links),
                execution_time=time.time() - start_time
            )

    except Exception as e:
        logger.error(f"Error getting graph data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving graph data: {str(e)}"
        )

@app.delete("/cancel-processing/{doc_id}", dependencies=[Depends(get_api_key)])
async def cancel_processing(doc_id: int):
    try:
        cleanup_processing(doc_id)
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Models
class ChunkingConfig(BaseModel):
    semantic_rules: List[str]
    overlap_size: int = Field(ge=0)
    min_chunk_size: int = Field(ge=0)
    max_chunk_size: int = Field(ge=0)

class EntityExtractionConfig(BaseModel):
    model: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    min_description_length: int = Field(ge=0)
    max_description_length: int = Field(ge=0)
    description_required: bool
    system_prompt: str
    extraction_prompt_template: str

class SearchConfig(BaseModel):
    top_k: int = Field(ge=1)
    vector_weight: float = Field(ge=0.0, le=1.0)
    text_weight: float = Field(ge=0.0, le=1.0)
    exact_phrase_weight: float = Field(ge=0.0)
    single_term_weight: float = Field(ge=0.0)
    proximity_distance: int = Field(ge=0)
    min_score_threshold: float = Field(ge=0.0, le=1.0)
    min_similarity_score: float = Field(ge=0.0, le=1.0)
    context_window_size: int = Field(ge=0)

class ResponseGenerationConfig(BaseModel):
    temperature: float = Field(ge=0.0, le=1.0)
    max_tokens: int = Field(ge=0)
    citation_style: str
    include_confidence: bool
    prompt_template: str

class RetrievalConfig(BaseModel):
    search: SearchConfig
    response_generation: ResponseGenerationConfig

class KnowledgeCreationConfig(BaseModel):
    chunking: ChunkingConfig
    entity_extraction: EntityExtractionConfig

class FullConfig(BaseModel):
    knowledge_creation: KnowledgeCreationConfig
    retrieval: RetrievalConfig

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

@app.get("/config", response_model=FullConfig, tags=["config"])
async def get_config():
    try:
        logger.info(f"Loading config from {CONFIG_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise HTTPException(status_code=404, detail=f"Config file not found at {CONFIG_PATH}")
        
        with open(CONFIG_PATH, 'r') as f:
            try:
                config = yaml.safe_load(f)
                logger.info("YAML loaded successfully")
                logger.info(f"Config structure: {config.keys()}")
            except yaml.YAMLError as e:
                logger.error(f"YAML parsing error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error parsing config YAML: {str(e)}")
            
            try:
                config_model = FullConfig(**config)
                logger.info("Config validated successfully")
                return config_model
            except Exception as e:
                logger.error(f"Config validation error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Config validation error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/config", tags=["config"])
async def update_config(config: FullConfig):
    try:
        logger.info(f"Updating config at {CONFIG_PATH}")
        # Convert to dict and save to yaml
        config_dict = config.dict()
        with open(CONFIG_PATH, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        logger.info("Config updated successfully")
        return {"status": "success", "message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import re
import fitz  # PyMuPDF
from openai import OpenAI
from processors.knowledge import KnowledgeGraphGenerator
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
import time

from db import DatabaseConnection
from core.config import config
from core.models import Document, DocumentChunk

from utils.status_cache import update_status_cache

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure LlamaParse
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not LLAMA_CLOUD_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

def validate_pdf(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate PDF file for size and password protection.
    Returns (is_valid, error_message)
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            return False, "File size exceeds 50MB limit"

        doc = fitz.open(file_path)
        if doc.needs_pass:
            doc.close()
            return False, "Password-protected PDFs are not supported"
            
        doc.close()
        return True, None
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"

def save_pdf(file_data: bytes, filename: str) -> str:
    """
    Save PDF file to documents directory
    """
    base_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    name, ext = os.path.splitext(base_filename)
    file_path = os.path.join(DOCUMENTS_DIR, base_filename)
    counter = 1
    
    # If file exists, append a number to the filename
    while os.path.exists(file_path):
        new_filename = f"{name}_{counter}{ext}"
        file_path = os.path.join(DOCUMENTS_DIR, new_filename)
        counter += 1
    
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    return file_path

def create_document_record(filename: str, file_path: str, file_size: int) -> int:
    """
    Create initial document record and return doc_id
    """
    conn = DatabaseConnection()
    try:
        conn.connect()
        logger.info(f"Creating document record for {filename}")
        
        # Create document record
        query = """
            INSERT INTO Documents (title, source)
            VALUES (%s, %s)
        """
        conn.execute_query(query, (filename, file_path))
        logger.info("Document record created")
        
        # Get the last inserted ID
        doc_id = conn.execute_query("SELECT LAST_INSERT_ID()")[0][0]
        logger.info(f"Got document ID: {doc_id}")
        
        # Create processing status record
        query = """
            INSERT INTO ProcessingStatus 
            (doc_id, file_name, file_path, file_size, current_step)
            VALUES (%s, %s, %s, %s, 'started')
        """
        conn.execute_query(query, (doc_id, filename, file_path, file_size))
        logger.info(f"Processing status record created for doc_id {doc_id}")
        
        # Verify the record was created
        verify_query = "SELECT file_path FROM ProcessingStatus WHERE doc_id = %s"
        result = conn.execute_query(verify_query, (doc_id,))
        if not result:
            logger.error(f"Failed to find ProcessingStatus record for doc_id {doc_id}")
        else:
            logger.info(f"Verified ProcessingStatus record exists for doc_id {doc_id}")
        
        return doc_id
    except Exception as e:
        logger.error(f"Error creating document record: {str(e)}", exc_info=True)
        raise
    finally:
        conn.disconnect()

def update_processing_status(doc_id: int, step: str, error_message: Optional[str] = None):
    """
    Update processing status for a document
    """
    conn = DatabaseConnection()
    try:
        conn.connect()
        query = """
            UPDATE ProcessingStatus 
            SET current_step = %s, error_message = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s
        """
        conn.execute_query(query, (step, error_message, doc_id))
        logger.info(f"Updated processing status for doc_id {doc_id} to {step}")
        
        # Get the updated status to sync with cache
        status = get_processing_status(doc_id)
        
        # Update the cache with the new module
        update_status_cache(doc_id, status)
            
    finally:
        conn.disconnect()

def get_processing_status(doc_id: int) -> Dict[str, Any]:
    """
    Get current processing status
    """
    conn = DatabaseConnection()
    try:
        conn.connect()
        query = """
            SELECT current_step, error_message, file_name, 
                   TIMESTAMPDIFF(SECOND, updated_at, NOW()) as seconds_since_update
            FROM ProcessingStatus
            WHERE doc_id = %s
        """
        result = conn.execute_query(query, (doc_id,))
        if not result:
            raise ValueError(f"No processing status found for doc_id {doc_id}")
            
        current_step, error_message, file_name, seconds_since_update = result[0]
        
        # If the operation is taking too long (over 5 minutes), mark it as failed
        if seconds_since_update > 300 and current_step not in ['completed', 'failed']:
            error_message = "Operation timed out after 5 minutes"
            update_processing_status(doc_id, 'failed', error_message)
            current_step = 'failed'
            
        return {
            "currentStep": current_step,
            "errorMessage": error_message,
            "fileName": file_name
        }
    finally:
        conn.disconnect()

def cleanup_processing(doc_id: int):
    """
    Clean up processing data for cancelled/failed jobs
    """
    conn = DatabaseConnection()
    try:
        conn.connect()
        # Delete document record
        conn.execute_query("DELETE FROM Documents WHERE doc_id = %s", (doc_id,))
        # Delete processing status
        conn.execute_query("DELETE FROM ProcessingStatus WHERE doc_id = %s", (doc_id,))
        # Delete any chunks
        conn.execute_query("DELETE FROM Document_Embeddings WHERE doc_id = %s", (doc_id,))
    finally:
        conn.disconnect()


def analyze_document_structure(doc: fitz.Document) -> Dict[str, Any]:
    """
    Analyze document structure to extract hierarchy and sections.
    (Used for optional structural context.)
    
    Args:
        doc: PyMuPDF document object
        
    Returns:
        Dict containing document structure information
    """
    structure = {
        "sections": [],
        "hierarchy": {},
        "toc": []
    }
    
    # Extract table of contents if available
    toc = doc.get_toc()
    if toc:
        structure["toc"] = toc
        
    current_section = None
    current_level = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            # Check for headings based on font size and style
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = span["size"]
                        text = span["text"].strip()
                        
                        # Identify potential headings
                        if size > 12 and text:  # Adjust threshold as needed
                            level = 1 if size > 16 else 2
                            if current_level < level or current_section is None:
                                section = {
                                    "title": text,
                                    "level": level,
                                    "start_page": page_num,
                                    "subsections": []
                                }
                                
                                if level == 1:
                                    structure["sections"].append(section)
                                    current_section = section
                                elif current_section is not None:
                                    current_section["subsections"].append(section)
                                
                                current_level = level
    
    return structure

def detect_semantic_unit(content: str) -> str:
    """
    Detect the semantic unit type of the content (optional heuristic).
    """
    content_lower = content.lower()
    
    if any(marker in content_lower for marker in ["example:", "e.g.", "example "]):
        return "example"
    elif any(marker in content_lower for marker in ["definition:", "is defined as", "refers to"]):
        return "definition"
    elif any(marker in content_lower for marker in ["step ", "first", "second", "finally"]):
        return "procedure"
    elif "?" in content and len(content.split()) < 50:
        return "question"
    elif any(marker in content_lower for marker in ["note:", "important:", "warning:"]):
        return "note"
    else:
        return "general"

def create_chunk_metadata(
    doc_id: int,
    position: int,
    structure: Dict[str, Any],
    content: str,
    prev_chunk_id: Optional[int] = None,
    next_chunk_id: Optional[int] = None,
    overlap_start_id: Optional[int] = None,
    overlap_end_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create metadata for a chunk based on its position and document structure.
    """
    current_section = None
    section_path = []
    
    # Try to find if any known section title is in the content
    for section in structure["sections"]:
        if section["title"] in content:
            current_section = section["title"]
            section_path.append(section["title"])
            for subsection in section["subsections"]:
                if subsection["title"] in content:
                    section_path.append(subsection["title"])
                    break
            break
    
    return {
        "doc_id": doc_id,
        "position": position,
        "section_path": "/".join(section_path) if section_path else None,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "overlap_start_id": overlap_start_id,
        "overlap_end_id": overlap_end_id,
        "semantic_unit": detect_semantic_unit(content),
        "structural_context": json.dumps(section_path)
    }

def chunk_by_paragraphs(
    text: str, 
    min_size: int, 
    max_size: int
) -> List[str]:
    """
    Splits text into paragraphs, then merges them to ensure
    each chunk is within [min_size, max_size].
    """
    # Split roughly by double newlines for paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    final_chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        # If adding this paragraph won't exceed max_size, append
        if sum(len(p) for p in current_chunk) + len(paragraph) <= max_size:
            current_chunk.append(paragraph)
        else:
            # finalize the current chunk
            if current_chunk:
                combined = "\n".join(current_chunk)
                final_chunks.append(combined)
            current_chunk = [paragraph]
    # add any leftover
    if current_chunk:
        combined = "\n".join(current_chunk)
        final_chunks.append(combined)

    # Now ensure that each chunk meets min_size by merging if needed
    merged_chunks = []
    buffer = None
    for c in final_chunks:
        if buffer is None:
            buffer = c
        else:
            if len(buffer) < min_size:
                # merge with current chunk
                buffer += "\n" + c
            else:
                merged_chunks.append(buffer)
                buffer = c
    if buffer:
        merged_chunks.append(buffer)

    return merged_chunks

def apply_overlap(
    chunks: List[str], 
    overlap_size: int
) -> List[Dict[str, Any]]:
    """
    Applies overlap from the previous chunk's tail and
    the next chunk's head, returning a list of dicts 
    with {'content': str, 'position': int}.
    """
    enhanced = []
    for i, ctext in enumerate(chunks):
        with_prev = ctext
        if i > 0:
            overlap_part = chunks[i-1][-overlap_size:]
            with_prev = overlap_part + "\n" + with_prev

        if i < len(chunks) - 1:
            overlap_part = chunks[i+1][:overlap_size]
            with_prev += "\n" + overlap_part

        enhanced.append({"content": with_prev, "position": i})
    return enhanced


def llamaparse_pdf(file_path: str, max_retries: int = 3) -> str:
    """
    Parse PDF using LlamaParse API and return markdown text.
    
    Args:
        file_path: Path to the PDF file
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        str: Parsed markdown text from the PDF
        
    Raises:
        PDFProcessingError: If parsing fails
    """
    start_time = datetime.now()
    
    try:
        logger.info("=" * 80)
        logger.info("Starting LlamaParse PDF processing")
        logger.info(f"Input file: {file_path}")
        logger.info("=" * 80)
        
        # Initialize LlamaParse with markdown output
        logger.info("Step 1: Initializing LlamaParse parser...")
        parser = LlamaParse(
            result_type="markdown",  # Get markdown formatted output
            num_workers=1,  # Use single worker for better stability
            check_interval=2.0  # Increase check interval to 2 seconds
        )
        logger.info("Parser initialized successfully")
        
        # Set up file extractor for PDF
        logger.info("Step 2: Setting up file extractor...")
        file_extractor = {".pdf": parser}
        logger.info("File extractor configured")
        
        # Use SimpleDirectoryReader to parse the file with retries
        logger.info("Step 3: Starting document parsing...")
        parse_start_time = datetime.now()
        
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Parsing attempt {attempt + 1}/{max_retries}")
                documents = SimpleDirectoryReader(
                    input_files=[file_path], 
                    file_extractor=file_extractor
                ).load_data()
                
                if documents and len(documents) > 0 and documents[0].text.strip():
                    content = documents[0].text
                    parse_duration = (datetime.now() - parse_start_time).total_seconds()
                    total_duration = (datetime.now() - start_time).total_seconds()
                    
                    logger.info("=" * 80)
                    logger.info("LlamaParse processing completed successfully:")
                    logger.info(f"Parsing attempt: {attempt + 1}/{max_retries}")
                    logger.info(f"Parsing time: {parse_duration:.2f} seconds")
                    logger.info(f"Total processing time: {total_duration:.2f} seconds")
                    logger.info(f"Content length: {len(content)} characters")
                    logger.info("=" * 80)
                    
                    return content
                else:
                    last_error = "No content extracted from PDF"
                    logger.warning(f"Attempt {attempt + 1} failed: Empty content received")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
            # Wait before retry if not the last attempt
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        total_duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"All {max_retries} parsing attempts failed. Last error: {last_error}"
        logger.error("=" * 80)
        logger.error(f"Error in LlamaParse processing after {total_duration:.2f} seconds:")
        logger.error(error_msg)
        logger.error("=" * 80)
        raise PDFProcessingError(error_msg)
            
    except Exception as e:
        total_duration = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 80)
        logger.error(f"Error in LlamaParse processing after {total_duration:.2f} seconds:")
        logger.error(str(e))
        logger.error("=" * 80)
        raise PDFProcessingError(f"Failed to parse PDF with LlamaParse: {str(e)}")

def process_pdf(doc_id: int, task=None):
    """
    Main PDF processing pipeline:
    1) Parse PDF -> text
    2) Analyze document structure (optional)
    3) Chunk text by paragraphs + enforce size + overlap
    4) Insert chunk metadata / embeddings
    5) Extract knowledge from each chunk (entities, relationships)
    6) Update status
    """
    try:
        conn = DatabaseConnection()
        try:
            conn.connect()
            
            # Get file path
            query = "SELECT file_path FROM ProcessingStatus WHERE doc_id = %s"
            result = conn.execute_query(query, (doc_id,))
            if not result:
                raise PDFProcessingError("Document not found")
            
            file_path = result[0][0]
            logger.info(f"Processing PDF: {file_path}")
            
            # Update status to processing
            update_processing_status(doc_id, "processing")
            
            # 1) Parse PDF to text
            text = llamaparse_pdf(file_path)
            
            # 2) Optional structure analysis
            doc = fitz.open(file_path)
            structure = analyze_document_structure(doc)
            
            # 3) Chunk text by paragraphs
            chunking_config = config.knowledge_creation['chunking']
            overlap_size = chunking_config['overlap_size']
            min_chunk_size = chunking_config['min_chunk_size']
            max_chunk_size = chunking_config['max_chunk_size']
            
            paragraph_chunks = chunk_by_paragraphs(text, min_chunk_size, max_chunk_size)
            enhanced_chunks = apply_overlap(paragraph_chunks, overlap_size)
            
            # 4) Store each chunk and create embeddings
            for chunk_dict in enhanced_chunks:
                chunk_text = chunk_dict["content"]
                position = chunk_dict["position"]
                
                # Insert chunk metadata
                metadata_query = """
                    INSERT INTO Chunk_Metadata 
                    (doc_id, position, section_path, prev_chunk_id, 
                     next_chunk_id, overlap_start_id, overlap_end_id, 
                     semantic_unit, structural_context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # We can figure out the IDs for prev/next if needed, or just store None
                chunk_meta = create_chunk_metadata(
                    doc_id=doc_id,
                    position=position,
                    structure=structure,
                    content=chunk_text,
                    prev_chunk_id=None,
                    next_chunk_id=None,
                    overlap_start_id=None,
                    overlap_end_id=None
                )
                
                conn.execute_query(
                    metadata_query,
                    (
                        chunk_meta["doc_id"],
                        chunk_meta["position"],
                        chunk_meta["section_path"],
                        chunk_meta["prev_chunk_id"],
                        chunk_meta["next_chunk_id"],
                        chunk_meta["overlap_start_id"],
                        chunk_meta["overlap_end_id"],
                        chunk_meta["semantic_unit"],
                        chunk_meta["structural_context"]
                    )
                )
                chunk_metadata_id = conn.execute_query("SELECT LAST_INSERT_ID()")[0][0]
                
                # Generate embedding for chunk content
                client = OpenAI()
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk_text
                )
                embedding = response.data[0].embedding
                
                # Store chunk and embedding in Document_Embeddings
                conn.execute_query(
                    """
                    INSERT INTO Document_Embeddings (doc_id, content, embedding) 
                    VALUES (%s, %s, JSON_ARRAY_PACK(%s))
                    """,
                    (doc_id, chunk_text, json.dumps(embedding))
                )
            
            # 5) Extract knowledge from each chunk
            #    (We do a second pass so we can store the knowledge in Entities/Relationships)
            kg = KnowledgeGraphGenerator(debug_output=True)
            for i, chunk_dict in enumerate(enhanced_chunks):
                chunk_text = chunk_dict["content"]
                logger.debug(f"Processing knowledge for chunk {i}, text length: {len(chunk_text)}")
                try:
                    knowledge = kg.extract_knowledge_sync(chunk_text)
                    if knowledge:
                        kg.store_knowledge(knowledge, conn)
                except Exception as e:
                    logger.error(f"Error extracting knowledge for chunk {i}: {str(e)}")
                    continue
            
            # 6) Update status to completed
            update_processing_status(doc_id, "completed")
            logger.info(f"Completed processing PDF: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            update_processing_status(doc_id, "failed", str(e))
            raise
        finally:
            conn.disconnect()
            
    except Exception as e:
        logger.error(f"Error in process_pdf: {str(e)}", exc_info=True)
        raise

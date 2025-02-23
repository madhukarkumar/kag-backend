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

# New imports for semantic chunking
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Initialize Sentence Transformer model for semantic chunking
model = SentenceTransformer('all-MiniLM-L6-v2')

class PDFProcessingError(Exception):
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
    """Save PDF file to documents directory"""
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
    """Create initial document record and return doc_id"""
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
    """Update processing status for a document"""
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
    """Get current processing status"""
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
    """Clean up processing data for cancelled/failed jobs"""
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

# New function for extracting section text
def extract_section_text(text: str, section: Dict[str, Any]) -> str:
    """Extract text for a specific section based on its title."""
    title = section['title']
    start_idx = text.find(title)
    if start_idx == -1:
        return text
    
    end_idx = len(text)
    for next_section in section.get('subsections', []):
        next_idx = text.find(next_section['title'], start_idx + len(title))
        if next_idx != -1 and next_idx < end_idx:
            end_idx = next_idx
    
    return text[start_idx:end_idx].strip()

# New function for paragraph-based chunking (fallback)
def paragraph_chunking(text: str, min_size: int = 200, max_size: int = 1500) -> List[Dict[str, Any]]:
    """Split text into paragraphs as a fallback."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) <= max_size:
            current_chunk += '\n\n' + para
        else:
            if len(current_chunk) >= min_size:
                chunks.append({"content": current_chunk.strip(), "metadata": {"section": "Paragraph"}})
            current_chunk = para
    
    if len(current_chunk) >= min_size:
        chunks.append({"content": current_chunk.strip(), "metadata": {"section": "Paragraph"}})
    
    return chunks

# Updated get_semantic_chunks function with sentence embeddings and document structure
def get_semantic_chunks(text: str, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split text into semantic chunks using embeddings and document structure."""
    try:
        chunking_config = config.knowledge_creation['chunking']
        min_size, max_size = chunking_config['min_chunk_size'], chunking_config['max_chunk_size']
        
        # Split text into sections based on structure
        sections = []
        for section in structure['sections']:
            section_text = extract_section_text(text, section)
            sections.append({
                "title": section['title'],
                "content": section_text,
                "level": section['level']
            })
        
        if not sections:
            sections = [{"title": "Full Document", "content": text, "level": 1}]
        
        # Process each section into semantic chunks
        all_chunks = []
        for section in sections:
            sentences = re.split(r'(?<=[.!?])\s+', section['content'].strip())
            if not sentences:
                continue
            
            embeddings = model.encode(sentences)
            chunks = []
            current_chunk = sentences[0]
            section_chunks = [{"content": current_chunk, "metadata": {"section": section['title']}}]
            
            for i in range(1, len(sentences)):
                similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                if similarity > 0.7 and len(current_chunk) + len(sentences[i]) <= max_size:
                    current_chunk += '. ' + sentences[i]
                    section_chunks[-1]["content"] = current_chunk
                else:
                    if len(current_chunk) >= min_size:
                        chunks.append(section_chunks[-1])
                    current_chunk = sentences[i]
                    section_chunks.append({"content": current_chunk, "metadata": {"section": section['title']}})
            
            if len(current_chunk) >= min_size:
                chunks.append(section_chunks[-1])
            
            all_chunks.extend(chunks)
        
        # Validate chunks
        all_chunks = validate_chunks(all_chunks, min_size, max_size)
        
        logger.info(f"Generated {len(all_chunks)} semantic chunks")
        return all_chunks
    
    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        return paragraph_chunking(text)

# New function to validate chunks
def validate_chunks(chunks: List[Dict[str, Any]], min_size: int, max_size: int) -> List[Dict[str, Any]]:
    """Validate and refine chunks to meet size constraints."""
    valid_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        if min_size <= len(content) <= max_size:
            valid_chunks.append(chunk)
        elif len(content) < min_size and valid_chunks:
            # Merge small chunks with previous
            valid_chunks[-1]["content"] += '\n' + content
        elif len(content) > max_size:
            # Split large chunks using paragraph chunking
            sub_chunks = paragraph_chunking(content, min_size, max_size)
            valid_chunks.extend(sub_chunks)
    return valid_chunks

# Existing analyze_document_structure function (unchanged)
def analyze_document_structure(doc: fitz.Document) -> Dict[str, Any]:
    """
    Analyze document structure to extract hierarchy and sections.
    
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

# Existing create_chunk_metadata function (unchanged)
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
    """Create metadata for a chunk based on its position and document structure."""
    
    # Find the current section based on content
    current_section = None
    section_path = []
    
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

# Existing detect_semantic_unit function (unchanged)
def detect_semantic_unit(content: str) -> str:
    """Detect the semantic unit type of the content."""
    # Simple heuristic-based detection
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

# Existing process_chunks_with_overlap function (unchanged)
def process_chunks_with_overlap(
    chunks: List[str],
    doc_id: int,
    structure: Dict[str, Any],
    overlap_size: int = None
) -> List[Dict[str, Any]]:
    """
    Process chunks adding overlap and metadata.
    
    Args:
        chunks: List of semantic chunks from Gemini
        doc_id: Document ID
        structure: Document structure information
        overlap_size: Number of characters to overlap
        
    Returns:
        List of enhanced chunks with metadata
    """
    if overlap_size is None:
        overlap_size = config.knowledge_creation['chunking']['overlap_size']
    
    enhanced_chunks = []
    chunk_ids = {}  # Store chunk IDs for linking
    
    for i, chunk in enumerate(chunks):
        # Create base chunk with content
        enhanced_chunk = {
            "content": chunk,
            "position": i
        }
        
        # Add overlap with previous chunk
        if i > 0:
            overlap_start = chunks[i-1][-overlap_size:]
            enhanced_chunk["content"] = overlap_start + "\n" + chunk
            enhanced_chunk["overlap_start_id"] = i-1
        
        # Add overlap with next chunk
        if i < len(chunks) - 1:
            overlap_end = chunks[i+1][:overlap_size]
            enhanced_chunk["content"] = enhanced_chunk["content"] + "\n" + overlap_end
            enhanced_chunk["overlap_end_id"] = i+1
        
        # Add metadata
        enhanced_chunk["metadata"] = create_chunk_metadata(
            doc_id=doc_id,
            position=i,
            structure=structure,
            content=enhanced_chunk["content"],
            prev_chunk_id=i-1 if i > 0 else None,
            next_chunk_id=i+1 if i < len(chunks) - 1 else None,
            overlap_start_id=enhanced_chunk.get("overlap_start_id"),
            overlap_end_id=enhanced_chunk.get("overlap_end_id")
        )
        
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

# Existing llamaparse_pdf function (unchanged)
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

# Updated process_pdf function
def process_pdf(doc_id: int, task=None):
    """Process PDF file through all steps"""
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
            
            # Open PDF
            doc = fitz.open(file_path)
            
            # Update status to processing
            update_processing_status(doc_id, "processing")
            
            # Extract text from PDF using LlamaParse
            text = llamaparse_pdf(file_path)
            
            # Analyze document structure
            structure = analyze_document_structure(doc)
            
            # Get semantic chunks using the new method
            semantic_chunks = get_semantic_chunks(text, structure)
            
            # Process chunks with overlap
            enhanced_chunks = process_chunks_with_overlap(
                chunks=[chunk["content"] for chunk in semantic_chunks],
                doc_id=doc_id,
                structure=structure
            )
            
            # Attach section metadata to enhanced chunks
            for i, chunk in enumerate(enhanced_chunks):
                chunk["metadata"]["section"] = semantic_chunks[i]["metadata"]["section"]
            
            # Store chunks and metadata
            for chunk in enhanced_chunks:
                metadata = chunk["metadata"]
                metadata_query = """
                    INSERT INTO Chunk_Metadata 
                    (doc_id, position, section_path, prev_chunk_id, 
                     next_chunk_id, overlap_start_id, overlap_end_id, 
                     semantic_unit, structural_context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                conn.execute_query(
                    metadata_query,
                    (
                        metadata["doc_id"],
                        metadata["position"],
                        metadata["section"],
                        metadata["prev_chunk_id"],
                        metadata["next_chunk_id"],
                        metadata["overlap_start_id"],
                        metadata["overlap_end_id"],
                        metadata["semantic_unit"],
                        metadata["structural_context"]
                    )
                )
                chunk_metadata_id = conn.execute_query("SELECT LAST_INSERT_ID()")[0][0]
                
                # Get embedding for chunk content
                client = OpenAI()
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk["content"]
                )
                embedding = response.data[0].embedding
                
                # Store chunk and embedding
                conn.execute_query(
                    """
                    INSERT INTO Document_Embeddings (doc_id, content, embedding, chunk_metadata_id) 
                    VALUES (%s, %s, JSON_ARRAY_PACK(%s), %s)
                    """,
                    (doc_id, chunk['content'], json.dumps(embedding), chunk_metadata_id)
                )
            
            # Extract and store knowledge
            kg = KnowledgeGraphGenerator(debug_output=True)
            for chunk in enhanced_chunks:
                knowledge = kg.extract_knowledge_sync(chunk['content'])
                kg.store_knowledge(knowledge, conn)
            
            # Update status to completed
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
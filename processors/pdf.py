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

from db import DatabaseConnection
from core.config import config
from core.models import Document, DocumentChunk

# Set up logging
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

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
            SET current_step = %s,
                error_message = %s,
                updated_at = NOW()
            WHERE doc_id = %s
        """
        conn.execute_query(query, (step, error_message, doc_id))
    finally:
        conn.disconnect()

def get_processing_status(doc_id: int) -> Dict[str, Any]:
    """Get current processing status"""
    conn = DatabaseConnection()
    try:
        conn.connect()
        query = """
            SELECT current_step, error_message, file_name, 
                   TIMESTAMPDIFF(SECOND, last_updated, NOW()) as seconds_since_update
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

def get_semantic_chunks(text: str) -> List[str]:
    """Use Gemini to get semantic chunks from text."""
    try:
        # Get chunking configuration
        chunking_config = config.knowledge_creation['chunking']
        
        prompt = f"""Split the following text into semantic chunks. Each chunk should be a coherent unit of information.
        Follow these rules strictly:
        {config.get_chunking_rules()}

        Return only the chunks, one per line, with '---' as separator.
        
        Text to split:
        {text}
        """
        
        logger.info("Sending request to Gemini for semantic chunking")
        logger.info(f"Input text length: {len(text)} characters")
        logger.info(f"Input text preview: {text[:200]}...")
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini")
        
        if not response.text:
            logger.warning("Gemini returned empty response, falling back to basic chunking")
            return [text]
            
        logger.info(f"Raw Gemini response: {response.text}")
        
        # Primary chunking method: Split using '---' separator
        chunks = [chunk.strip() for chunk in response.text.split('---') if chunk.strip()]
        
        # Secondary method: If Gemini didn't provide expected separators, split into sentence-based chunks
        if not chunks or len(chunks) < 2:
            logger.warning("No valid '---' chunk markers found, falling back to sentence-based chunking.")
            sentences = re.split(r'(?<=[.!?])\s+', text)  # Splits at sentence endings
            chunks = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]  # Group sentences
        
        # Apply size constraints from config
        valid_chunks = []
        for chunk in chunks:
            if chunking_config['min_chunk_size'] <= len(chunk) <= chunking_config['max_chunk_size']:
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Chunk size {len(chunk)} outside configured bounds, skipping")
        
        # If all chunks were filtered out, revert to the entire text as a single chunk
        if not valid_chunks:
            logger.warning("All chunks filtered out, falling back to basic chunking")
            return [text]
        
        # Log each chunk
        logger.info(f"Generated {len(valid_chunks)} semantic chunks:")
        for i, chunk in enumerate(valid_chunks, 1):
            logger.info(f"Chunk {i}/{len(valid_chunks)}:")
            logger.info(f"Length: {len(chunk)} characters")
            logger.info(f"Content: {chunk[:200]}...")  # Log only first 200 characters for brevity
            logger.info("-" * 80)
        
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        logger.warning("Falling back to basic chunking")
        return [text]

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
            
            # Extract text from PDF
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Analyze document structure
            structure = analyze_document_structure(doc)
            
            # Get semantic chunks using Gemini
            semantic_chunks = get_semantic_chunks(text)
            
            # Process chunks with overlap and metadata
            enhanced_chunks = process_chunks_with_overlap(
                chunks=semantic_chunks,
                doc_id=doc_id,
                structure=structure
            )
            
            # Store chunks and metadata
            for chunk in enhanced_chunks:
                # Store chunk metadata
                metadata_query = """
                    INSERT INTO Chunk_Metadata 
                    (doc_id, position, section_path, prev_chunk_id, 
                     next_chunk_id, overlap_start_id, overlap_end_id, 
                     semantic_unit, structural_context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                metadata = chunk["metadata"]
                conn.execute_query(
                    metadata_query,
                    (
                        metadata["doc_id"],
                        metadata["position"],
                        metadata["section_path"],
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
                    INSERT INTO Document_Embeddings (doc_id, content, embedding) 
                    VALUES (%s, %s, JSON_ARRAY_PACK(%s))
                    """,
                    (doc_id, chunk['content'], json.dumps(embedding))
                )
            
            # Extract and store knowledge
            kg = KnowledgeGraphGenerator(debug_output=True)
            for i, chunk in enumerate(enhanced_chunks):
                chunk_text = chunk['content']
                logger.debug(f"Processing chunk {i}, content: {repr(chunk_text)}")  # Debug log
                try:
                    knowledge = kg.extract_knowledge_sync(chunk_text)
                    if knowledge:
                        kg.store_knowledge(knowledge, conn)
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    logger.debug(f"Problematic chunk content: {repr(chunk_text)}")
                    continue  # Skip failed chunk and continue with others
            
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

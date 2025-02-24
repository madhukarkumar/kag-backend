import os
from pathlib import Path
from openai import OpenAI
import logging
from datetime import datetime
import sys
import time
import json
from typing import Optional, Tuple, Dict, List, Any
from pydub import AudioSegment
import math
from moviepy.editor import VideoFileClip

# Add the parent directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import DatabaseConnection
from processors.knowledge import KnowledgeGraphGenerator
from utils.status_cache import update_status_cache
from dotenv import load_dotenv

load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('video_processing.log')  # File output
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_BYTES = 24 * 1024 * 1024  # 24MB to be safe
CHUNK_OVERLAP_MS = 10000  # 10 seconds overlap between chunks
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv']
DOCUMENTS_DIR = os.path.join(os.getcwd(), "documents")
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

class VideoProcessingError(Exception):
    pass

def save_video(file_data: bytes, filename: str) -> str:
    """Save video file to documents directory"""
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
        
        # Get the last inserted ID
        doc_id = conn.execute_query("SELECT LAST_INSERT_ID()")[0][0]
        logger.info(f"Document record created with ID: {doc_id}")
        
        # Create processing status record
        query = """
            INSERT INTO ProcessingStatus 
            (doc_id, file_name, file_path, file_size, current_step)
            VALUES (%s, %s, %s, %s, 'started')
        """
        conn.execute_query(query, (doc_id, filename, file_path, file_size))
        logger.info(f"Processing status record created for doc_id {doc_id}")
        
        return doc_id
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

def store_transcript_chunks(doc_id: int, transcript: str) -> List[Dict]:
    """Store transcript chunks in the database"""
    conn = DatabaseConnection()
    try:
        conn.connect()
        
        # Create a single chunk for now (can be enhanced with semantic chunking later)
        chunk_query = """
            INSERT INTO Chunk_Metadata 
            (doc_id, position, semantic_unit)
            VALUES (%s, %s, %s)
        """
        conn.execute_query(chunk_query, (doc_id, 0, 'transcript'))
        chunk_id = conn.execute_query("SELECT LAST_INSERT_ID()")[0][0]
        
        # Generate and store embedding
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=transcript
        )
        embedding = response.data[0].embedding
        
        # Store embedding
        embedding_query = """
            INSERT INTO Document_Embeddings 
            (doc_id, content, embedding, chunk_metadata_id) 
            VALUES (%s, %s, JSON_ARRAY_PACK(%s), %s)
        """
        conn.execute_query(
            embedding_query,
            (doc_id, transcript, json.dumps(embedding), chunk_id)
        )
        
        return [{"chunk_id": chunk_id, "content": transcript}]
    finally:
        conn.disconnect()

def extract_audio(video_path: Path) -> Path:
    """Extract audio from video file and save it temporarily"""
    logger.info(f"Extracting audio from {video_path}")
    video = VideoFileClip(str(video_path))
    audio_path = video_path.with_suffix('.wav')
    logger.info(f"Saving temporary audio file to {audio_path}")
    video.audio.write_audiofile(str(audio_path), logger='bar')
    video.close()
    logger.info("Audio extraction completed")
    return audio_path

def split_audio_file(audio_path: Path) -> List[Path]:
    """Split large audio file into chunks smaller than 25MB with overlap"""
    logger.info(f"Checking if audio file needs to be split")
    file_size = audio_path.stat().st_size
    
    if file_size <= MAX_FILE_SIZE_BYTES:
        logger.info("Audio file is small enough, no splitting needed")
        return [audio_path]
        
    logger.info(f"Audio file is too large ({file_size} bytes), splitting into chunks")
    audio = AudioSegment.from_wav(str(audio_path))
    total_duration_ms = len(audio)
    
    num_chunks = math.ceil(file_size / MAX_FILE_SIZE_BYTES)
    base_chunk_duration = total_duration_ms // num_chunks
    
    logger.info(f"Splitting into {num_chunks} chunks of approximately {base_chunk_duration/1000:.2f} seconds each")
    
    chunks = []
    start = 0
    chunk_number = 1
    
    while start < total_duration_ms:
        if chunk_number == num_chunks:
            end = total_duration_ms
        else:
            end = min(start + base_chunk_duration + CHUNK_OVERLAP_MS, total_duration_ms)
        
        chunk_audio = audio[start:end]
        chunk_path = audio_path.with_stem(f"{audio_path.stem}_chunk_{chunk_number}")
        chunk_audio.export(str(chunk_path), format="wav")
        
        chunk_size = chunk_path.stat().st_size
        logger.info(f"Created chunk {chunk_number}/{num_chunks}: {chunk_path} ({chunk_size/1024/1024:.2f}MB, duration: {(end-start)/1000:.2f}s)")
        
        chunks.append(chunk_path)
        start = end - CHUNK_OVERLAP_MS if end < total_duration_ms else end
        chunk_number += 1
    
    return chunks

def transcribe_with_openai(audio_path: Path) -> Optional[str]:
    """Transcribe audio using OpenAI's Whisper model"""
    try:
        logger.info("Attempting transcription with OpenAI")
        client = OpenAI()
        
        audio_chunks = split_audio_file(audio_path)
        all_transcripts = []
        
        for chunk_path in audio_chunks:
            logger.info(f"Processing chunk: {chunk_path}")
            with open(chunk_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                all_transcripts.append(response)
            
            if chunk_path != audio_path:
                chunk_path.unlink()
                logger.info(f"Cleaned up chunk: {chunk_path}")
        
        full_transcript = " ".join(all_transcripts)
        logger.info("OpenAI transcription successful")
        return full_transcript
        
    except Exception as e:
        error_msg = str(e)
        logger.error("OpenAI transcription failed:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_msg}")
        
        error_path = audio_path.with_suffix('.openai.error.log')
        error_path.write_text(f"Error occurred at {datetime.now()}\n\n{error_msg}")
        logger.error(f"OpenAI error details saved to {error_path}")
        
        return None

def save_transcript_markdown(video_path: Path, transcript: str, metadata: Dict = None) -> Path:
    """Save transcript and processing metadata as a markdown file"""
    # Create documents directory if it doesn't exist
    docs_dir = Path('documents')
    docs_dir.mkdir(exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = docs_dir / f"{video_path.stem}_{timestamp}_transcript.md"
    
    logger.info(f"Saving transcript to {markdown_path}")
    
    # Build markdown content
    content = [
        f"# Video Transcription: {video_path.stem}",
        "",
        "## Processing Metadata",
        f"- **Source File:** {video_path.name}",
        f"- **Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **File Type:** {video_path.suffix[1:].upper()}",
        f"- **File Size:** {video_path.stat().st_size / (1024*1024):.2f} MB",
    ]
    
    # Add any additional metadata
    if metadata:
        content.extend([
            "",
            "## Processing Results",
            *[f"- **{k}:** {v}" for k, v in metadata.items() if k != 'transcript']
        ])
    
    content.extend([
        "",
        "## Transcript",
        transcript
    ])
    
    # Write the file
    markdown_path.write_text('\n'.join(content))
    logger.info(f"Transcript saved successfully to {markdown_path}")
    return markdown_path

def process_video(video_path: Path) -> Dict:
    """Process video through the pipeline"""
    try:
        # Create document record
        file_size = video_path.stat().st_size
        doc_id = create_document_record(
            filename=video_path.name,
            file_path=str(video_path),
            file_size=file_size
        )
        
        update_processing_status(doc_id, "extracting_audio")
        audio_path = extract_audio(video_path)
        
        try:
            # Transcribe audio
            update_processing_status(doc_id, "transcribing")
            transcript = transcribe_with_openai(audio_path)
            
            if not transcript:
                raise VideoProcessingError("Failed to get transcript")
            
            # Save transcript to markdown for reference
            save_transcript_markdown(video_path, transcript, {"doc_id": doc_id})
            
            # Store chunks and generate embeddings
            update_processing_status(doc_id, "processing")
            chunks = store_transcript_chunks(doc_id, transcript)
            
            # Extract knowledge
            update_processing_status(doc_id, "extracting_knowledge")
            kg = KnowledgeGraphGenerator(debug_output=True)
            conn = DatabaseConnection()
            try:
                conn.connect()
                for chunk in chunks:
                    try:
                        knowledge = kg.extract_knowledge_sync(chunk['content'])
                        if knowledge:
                            kg.store_knowledge(knowledge, conn)
                    except Exception as e:
                        logger.error(f"Error extracting knowledge from chunk: {str(e)}")
            finally:
                conn.disconnect()
            
            # Mark as completed
            update_processing_status(doc_id, "completed")
            
            return {
                "doc_id": doc_id,
                "status": "completed",
                "chunks": len(chunks)
            }
            
        except Exception as e:
            update_processing_status(doc_id, "failed", str(e))
            raise
        finally:
            if audio_path.exists():
                audio_path.unlink()
                
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise VideoProcessingError(f"Failed to process video: {str(e)}")

def main():
    if len(sys.argv) != 2:
        logger.error("Missing video file argument")
        print("Usage: python video_processor.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    video_path = Path(DOCUMENTS_DIR) / video_file

    if not video_path.exists():
        logger.error(f"File not found: {video_path}")
        sys.exit(1)

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        logger.error(f"Unsupported file format: {video_path.suffix}")
        sys.exit(1)

    try:
        result = process_video(video_path)
        logger.info("Video processing complete")
        logger.info("Processing results:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Video processor started")
    main()
    logger.info("Video processor finished")
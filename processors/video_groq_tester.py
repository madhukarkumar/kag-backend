import os
from pathlib import Path
from groq import Groq
from openai import OpenAI
import logging
from datetime import datetime
import sys
import time
from typing import Optional, Tuple
from pydub import AudioSegment
import math

from moviepy.editor import VideoFileClip

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_BYTES = 24 * 1024 * 1024  # 24MB to be safe
CHUNK_OVERLAP_MS = 10000  # 10 seconds overlap between chunks

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    logger.info("Successfully imported VideoFileClip using direct import")
except ImportError as e:
    logger.error("Failed to import VideoFileClip. Detailed error information:")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error message: {str(e)}")
    logger.error("\nTroubleshooting steps:")
    logger.error("1. Try installing an older version: pip install 'moviepy<2.0.0'")
    logger.error("2. Or try installing the latest development version:")
    logger.error("   pip install git+https://github.com/Zulko/moviepy.git")
    sys.exit(1)

def extract_audio(video_path):
    """
    Extract audio from video file and save it temporarily
    """
    logger.info(f"Extracting audio from {video_path}")
    video = VideoFileClip(str(video_path))
    audio_path = video_path.with_suffix('.wav')
    logger.info(f"Saving temporary audio file to {audio_path}")
    video.audio.write_audiofile(str(audio_path), logger='bar')
    video.close()
    logger.info("Audio extraction completed")
    return audio_path

def split_audio_file(audio_path: Path) -> list[Path]:
    """
    Split large audio file into chunks smaller than 25MB with overlap
    """
    logger.info(f"Checking if audio file needs to be split")
    file_size = audio_path.stat().st_size
    
    if file_size <= MAX_FILE_SIZE_BYTES:
        logger.info("Audio file is small enough, no splitting needed")
        return [audio_path]
        
    logger.info(f"Audio file is too large ({file_size} bytes), splitting into chunks")
    audio = AudioSegment.from_wav(str(audio_path))
    total_duration_ms = len(audio)
    
    # Calculate number of chunks needed (rounded up)
    num_chunks = math.ceil(file_size / MAX_FILE_SIZE_BYTES)
    # Calculate base chunk duration (without overlap)
    base_chunk_duration = total_duration_ms // num_chunks
    
    logger.info(f"Splitting into {num_chunks} chunks of approximately {base_chunk_duration/1000:.2f} seconds each")
    
    chunks = []
    start = 0
    chunk_number = 1
    
    while start < total_duration_ms:
        # For the last chunk, make sure we include all remaining audio
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

def transcribe_with_groq(client: Groq, audio_path: Path, max_retries: int = 2) -> Tuple[Optional[str], bool]:
    """
    Attempt transcription with Groq API
    Returns: (transcript, success)
    """
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                
            logger.info(f"Groq transcription attempt {attempt + 1}/{max_retries}")
            response = client.audio.transcriptions.create(
                file=(str(audio_path), audio_content),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )
            logger.info("Groq transcription successful")
            return response.text, True
            
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            error_msg = str(e)
            
            logger.error(f"Groq attempt {attempt + 1} failed:")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {error_msg}")
            
            if attempt == max_retries - 1:
                logger.error("All Groq retry attempts failed")
                error_path = audio_path.with_suffix('.groq.error.log')
                error_path.write_text(f"Error occurred at {datetime.now()}\n\n{error_msg}")
                logger.error(f"Groq error details saved to {error_path}")
                return None, False
                
            logger.info(f"Waiting {delay} seconds before retry...")
            time.sleep(delay)
    
    return None, False

def transcribe_with_openai(audio_path: Path) -> Optional[str]:
    """
    Attempt transcription with OpenAI API
    """
    try:
        logger.info("Attempting transcription with OpenAI as fallback")
        client = OpenAI()
        
        # Split audio if needed
        audio_chunks = split_audio_file(audio_path)
        all_transcripts = []
        
        # Process each chunk
        for chunk_path in audio_chunks:
            logger.info(f"Processing chunk: {chunk_path}")
            with open(chunk_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                all_transcripts.append(response)  # response is already a string
            
            # Clean up chunk if it's not the original file
            if chunk_path != audio_path:
                chunk_path.unlink()
                logger.info(f"Cleaned up chunk: {chunk_path}")
        
        # Combine all transcripts
        full_transcript = " ".join(all_transcripts)
        logger.info("OpenAI transcription successful")
        return full_transcript
        
    except Exception as e:
        error_msg = str(e)
        logger.error("OpenAI transcription failed:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_msg}")
        
        # Save error details
        error_path = audio_path.with_suffix('.openai.error.log')
        error_path.write_text(f"Error occurred at {datetime.now()}\n\n{error_msg}")
        logger.error(f"OpenAI error details saved to {error_path}")
        
        return None

def transcribe_audio(audio_path, api_key):
    """
    Send audio to Groq API for transcription, fallback to OpenAI if Groq fails
    """
    logger.info(f"Starting transcription process")
    
    # Try Groq first
    logger.info("Attempting transcription with Groq")
    groq_client = Groq()
    transcript, success = transcribe_with_groq(groq_client, audio_path)
    
    # If Groq fails, try OpenAI
    if not success:
        logger.info("Groq transcription failed, falling back to OpenAI")
        transcript = transcribe_with_openai(audio_path)
    
    if transcript:
        logger.info("Transcription completed successfully")
        return transcript
    else:
        logger.error("All transcription attempts failed")
        return None

def save_markdown(video_path, transcript):
    """
    Save transcript as a markdown file with metadata
    """
    markdown_path = video_path.with_suffix('.md')
    logger.info(f"Saving markdown file to {markdown_path}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"""# Video Transcription: {video_path.stem}

## Metadata
- **Source File:** {video_path.name}
- **Transcription Date:** {timestamp}
- **File Type:** {video_path.suffix[1:].upper()}

## Transcript
{transcript}
"""
    
    markdown_path.write_text(markdown_content)
    logger.info("Markdown file saved successfully")
    return markdown_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python video_groq_tester.py <video_file>")
        sys.exit(1)

    # Path to documents folder
    docs_path = Path('documents')
    if not docs_path.exists():
        logger.error("Documents folder not found")
        sys.exit(1)

    video_file = sys.argv[1]
    video_path = docs_path / video_file  # Create path to video file

    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist")
        sys.exit(1)

    try:
        # Extract audio from video
        audio_path = extract_audio(video_path)
        transcript = None

        # Temporarily skip Groq transcription
        """
        # Try Groq first
        logger.info("Attempting transcription with Groq")
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        transcript, success = transcribe_with_groq(groq_client, audio_path)
        """
        
        # Go directly to OpenAI
        transcript = transcribe_with_openai(audio_path)

        if transcript:
            # Save transcript
            transcript_path = video_path.with_suffix('.md')
            transcript_path.write_text(transcript)
            logger.info(f"Transcript saved to {transcript_path}")
        else:
            logger.error("All transcription attempts failed")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary audio file
        logger.info(f"Cleaning up temporary audio file {audio_path}")
        if audio_path.exists():
            audio_path.unlink()
        logger.info("Video transcription process completed")

if __name__ == "__main__":
    main()
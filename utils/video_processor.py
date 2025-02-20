from pathlib import Path
from processors.video_groq_tester import extract_audio, transcribe_with_openai
from utils.logger import get_logger

logger = get_logger(__name__)

def process_video(file_path: Path) -> dict:
    """Process video files and return transcript"""
    try:
        # Extract audio and get transcript
        audio_path = extract_audio(file_path)
        transcript = transcribe_with_openai(audio_path)
        
        if transcript:
            # Clean up temp audio file
            audio_path.unlink()
            return {
                'success': True,
                'transcript': transcript,
                'file_path': str(file_path)
            }
            
        return {
            'success': False,
            'error': 'Transcription failed'
        }
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

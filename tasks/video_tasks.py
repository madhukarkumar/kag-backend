from celery import shared_task
from processors.video_processor import process_video
from pathlib import Path
from db import DatabaseConnection

@shared_task(name='tasks.process_video_task')
def process_video_task(doc_id: int):
    """Process a video document asynchronously"""
    # Get file path from ProcessingStatus
    conn = DatabaseConnection()
    try:
        conn.connect()
        query = "SELECT file_path FROM ProcessingStatus WHERE doc_id = %s"
        result = conn.execute_query(query, (doc_id,))
        if not result:
            raise ValueError("Document not found")
        file_path = result[0][0]
        return process_video(Path(file_path))
    finally:
        conn.disconnect() 
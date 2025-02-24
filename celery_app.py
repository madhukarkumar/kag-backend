from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis URL from environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
app = Celery(
    'singlestore_kag',
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_connection_retry_on_startup=True
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    worker_max_memory_per_child=200000,  # 200MB
    imports=['tasks.pdf_tasks', 'tasks.video_tasks']  # Import both PDF and video tasks modules
)

# Example task
@app.task
def process_document(document_id: str, input_file: str):
    """Process a document asynchronously"""
    from main import DocumentProcessor
    processor = DocumentProcessor()
    return processor.process_document(document_id, input_file)

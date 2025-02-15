from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis URL from environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
app = Celery('tasks',
             broker=REDIS_URL,
             backend=REDIS_URL,
             broker_connection_retry_on_startup=True)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    enable_utc=True,
    task_track_started=True,
    task_ignore_result=False,
    task_acks_late=True,
    worker_prefetch_multiplier=1
)

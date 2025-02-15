from celery import Celery

# Create Celery app
app = Celery('tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0')

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

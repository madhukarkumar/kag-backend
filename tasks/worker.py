from celery import signals
from celery.utils.log import get_task_logger
import os
import sys

from core.config import config
from processors import process_pdf_document
from processors import generate_knowledge_graph

# Import the Celery app from celeryconfig
from .celeryconfig import app as celery

# Set up logging
logger = get_task_logger(__name__)

@celery.task(bind=True, name='tasks.process_pdf_task')
def process_pdf_task(self, doc_id: int):
    """
    Celery task to process a PDF file.
    
    Args:
        doc_id: Document ID to process
    """
    try:
        # Update task state to STARTED
        self.update_state(state='STARTED', meta={
            'status': 'Starting PDF processing...',
            'current': 0,
            'total': 100
        })
        
        # Process PDF and create embeddings
        process_pdf_document(doc_id)
        
        # Generate knowledge graph
        generate_knowledge_graph(doc_id)
        
        # Update task state to SUCCESS
        self.update_state(state='SUCCESS', meta={
            'status': 'PDF processed successfully',
            'current': 100,
            'total': 100
        })
        
        return {
            'status': 'success',
            'message': f'Successfully processed document {doc_id}'
        }
        
    except Exception as exc:
        # Log the error
        logger.error(f"Error processing PDF: {str(exc)}", exc_info=True)
        
        # Update task state to FAILURE
        self.update_state(state='FAILURE', meta={
            'status': 'Failed to process PDF',
            'error': str(exc),
            'exc_type': type(exc).__name__
        })
        
        # Re-raise the exception for Celery to handle
        raise exc

@signals.task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Log when task starts"""
    logger.info(f"Starting task {task_id}")

@signals.task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Log when task succeeds"""
    logger.info(f"Task {sender.request.id} completed successfully: {result}")

@signals.task_failure.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    """Log when task fails"""
    logger.error(f"Task {sender.request.id} failed: {str(exception)}")

from celery import shared_task
from processors.pdf import process_pdf

@shared_task(name='tasks.process_pdf_task')
def process_pdf_task(doc_id: int):
    """Process a PDF document asynchronously"""
    return process_pdf(doc_id)

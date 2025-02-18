import os
import sys
import logging
from datetime import datetime
from processors.pdf import llamaparse_pdf

# Set up logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_llamaparse(filename: str):
    """
    Test the llamaparse_pdf function with a given file.
    
    Args:
        filename: Name of the PDF file in the documents directory
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting LlamaParse PDF Test")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Construct full file path
        documents_dir = os.path.join(os.getcwd(), "documents")
        file_path = os.path.join(documents_dir, filename)
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Input file size: {file_size:.2f} MB")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        # Call llamaparse_pdf
        logger.info("Calling llamaparse_pdf function...")
        markdown_text = llamaparse_pdf(file_path)
        
        # Save output to a markdown file
        output_filename = f"{os.path.splitext(filename)[0]}_parsed.md"
        output_path = os.path.join(documents_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
            
        logger.info(f"Parsed content saved to: {output_path}")
        
        # Calculate statistics
        total_duration = (datetime.now() - start_time).total_seconds()
        chars_per_second = len(markdown_text) / total_duration if total_duration > 0 else 0
        
        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info("=" * 80)
        logger.info(f"Total characters extracted: {len(markdown_text):,}")
        logger.info(f"Processing speed: {chars_per_second:.2f} chars/second")
        logger.info(f"Output file size: {os.path.getsize(output_path) / 1024:.2f} KB")
        logger.info("=" * 80)
        
        # Print content preview
        preview_length = 500
        preview = markdown_text[:preview_length] + "..." if len(markdown_text) > preview_length else markdown_text
        logger.info("\nContent Preview:")
        logger.info("=" * 80)
        print(preview)
        logger.info("=" * 80)
        
        # Print full content
        logger.info("\nFull Markdown Content:")
        logger.info("=" * 80)
        print(markdown_text)
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Test failed with error:")
        logger.error(str(e))
        logger.error("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_llamaparse.py <filename>")
        print("Example: python test_llamaparse.py document.pdf")
        sys.exit(1)
        
    test_llamaparse(sys.argv[1])

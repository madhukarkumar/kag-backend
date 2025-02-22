from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Cache entry class to track timestamps
class CacheEntry:
    def __init__(self, status: Any, timestamp: datetime):
        self.status = status
        self.timestamp = timestamp

# Global cache dictionary
processing_status_cache: Dict[int, CacheEntry] = {}

def update_status_cache(doc_id: int, status: Any):
    """Update the in-memory status cache with timestamp"""
    try:
        processing_status_cache[doc_id] = CacheEntry(status, datetime.now())
        logger.debug(f"Updated cache for doc_id {doc_id}")
    except Exception as e:
        logger.error(f"Failed to update cache for doc_id {doc_id}: {str(e)}")

def get_cached_status(doc_id: int) -> Optional[Any]:
    """Get status from cache if valid"""
    try:
        if doc_id in processing_status_cache:
            entry = processing_status_cache[doc_id]
            # Cache entries expire after 30 seconds
            if datetime.now() - entry.timestamp < timedelta(seconds=30):
                logger.debug(f"Cache hit for doc_id {doc_id}")
                return entry.status
            else:
                # Remove expired entry
                logger.debug(f"Removing expired cache entry for doc_id {doc_id}")
                del processing_status_cache[doc_id]
    except Exception as e:
        logger.error(f"Error accessing cache for doc_id {doc_id}: {str(e)}")
    return None

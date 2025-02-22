"""Status cache management utilities."""
import logging
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# In-memory cache for processing status
# Format: {doc_id: (status_dict, expiry_timestamp)}
_status_cache: Dict[int, Tuple[Dict[str, Any], datetime]] = {}

# Cache expiration time in seconds
CACHE_TTL = 2  # Short TTL to ensure fresh status

def update_status_cache(doc_id: int, status: Dict[str, Any]) -> None:
    """Update the in-memory status cache for a document.
    
    Args:
        doc_id: Document ID
        status: Status dictionary to cache
    """
    expiry = datetime.now() + timedelta(seconds=CACHE_TTL)
    _status_cache[doc_id] = (status, expiry)
    logger.debug(f"Updated status cache for doc_id {doc_id}: {status}")

def get_status_from_cache(doc_id: int) -> Dict[str, Any]:
    """Get status from cache if available and not expired.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Cached status dictionary or None if not found or expired
    """
    if doc_id not in _status_cache:
        return None
        
    status, expiry = _status_cache[doc_id]
    if datetime.now() > expiry:
        del _status_cache[doc_id]
        return None
        
    return status

def clear_status_cache(doc_id: int) -> None:
    """Clear cached status for a document.
    
    Args:
        doc_id: Document ID
    """
    if doc_id in _status_cache:
        del _status_cache[doc_id]
        logger.debug(f"Cleared status cache for doc_id {doc_id}")

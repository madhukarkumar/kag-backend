from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from os import getenv
import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Validate the API key from the request header.
    Raises 403 if the key is invalid or missing.
    """
    api_key = getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY environment variable not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key configuration error"
        )
    
    if api_key_header == api_key:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key"
    )

import os
import logging
from typing import Any, List, Optional, Dict
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.connection = self._create_connection()
    
    def _create_connection(self):
        """Create a connection to SingleStore"""
        try:
            connection = mysql.connector.connect(
                host=os.getenv('SINGLESTORE_HOST', 'localhost'),
                user=os.getenv('SINGLESTORE_USER', 'root'),
                password=os.getenv('SINGLESTORE_PASSWORD', ''),
                database=os.getenv('SINGLESTORE_DATABASE', 'kag'),
                port=int(os.getenv('SINGLESTORE_PORT', '3306'))
            )
            
            if connection.is_connected():
                logger.info('Successfully connected to SingleStore database')
                return connection
                
        except Error as e:
            logger.error(f"Error connecting to SingleStore: {e}")
            raise
    
    def execute(self, query: str, values: tuple = None) -> Any:
        """Execute a query and return the cursor"""
        try:
            cursor = self.connection.cursor()
            if values:
                cursor.execute(query, values)
            else:
                cursor.execute(query)
            
            self.connection.commit()
            return cursor
            
        except Error as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Values: {values}")
            self.connection.rollback()
            raise
    
    def fetch_one(self, query: str, values: tuple = None) -> Optional[tuple]:
        """Execute a query and fetch one result"""
        cursor = self.execute(query, values)
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def fetch_all(self, query: str, values: tuple = None) -> List[tuple]:
        """Execute a query and fetch all results"""
        cursor = self.execute(query, values)
        results = cursor.fetchall()
        cursor.close()
        return results
    
    def close(self):
        """Close the database connection"""
        if self.connection.is_connected():
            self.connection.close()
            logger.info('Database connection closed')
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Get or create a database connection"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance 
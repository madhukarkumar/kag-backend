"""
SingleStore database connection and operations module.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import mysql.connector
from mysql.connector import Error
import numpy as np
import json
from datetime import datetime
import logging

from core.config import config
from core.models import Document, DocumentChunk, Entity, Relationship

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    logger.warning("python-dotenv not found, using environment variables directly")

# Database configuration
DB_HOST = os.getenv("SINGLESTORE_HOST", "localhost")
DB_PORT = int(os.getenv("SINGLESTORE_PORT", "3306"))  # Default SingleStore port
DB_USER = os.getenv("SINGLESTORE_USER", "root")
DB_PASSWORD = os.getenv("SINGLESTORE_PASSWORD", "")
DB_DATABASE = os.getenv("SINGLESTORE_DATABASE", "knowledge_graph")

class DatabaseConnection:
    """Manages database connections and operations for SingleStore."""
    
    def __init__(self):
        """Initialize the database connection."""
        self.conn = None
        self.cursor = None
        
    def connect(self) -> None:
        """
        Establish connection to SingleStore database.
        
        Raises:
            Exception: If connection fails or required environment variables are missing.
        """
        if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_DATABASE]):
            raise ValueError(
                "Missing required database configuration. Please ensure all required "
                "environment variables are set in .env file:\n"
                "- SINGLESTORE_HOST\n"
                "- SINGLESTORE_USER\n"
                "- SINGLESTORE_PASSWORD\n"
                "- SINGLESTORE_DATABASE"
            )
        
        try:
            self.conn = mysql.connector.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_DATABASE
            )
            self.cursor = self.conn.cursor(buffered=True)
            logger.info("Successfully connected to SingleStore database")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close database connection and cursor."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception:
                pass
            self.cursor = None
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            logger.info("Database connection closed")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[Tuple[Any, ...]]]:
        """
        Execute a SQL query and return results if any.
        
        Args:
            query: SQL query to execute
            params: Optional tuple of parameters for the query
            
        Returns:
            List of tuples containing the query results, or None for non-SELECT queries
            
        Raises:
            Exception: If query execution fails
        """
        try:
            if not self.conn or not self.cursor:
                raise Exception("No active database connection")
            
            self.cursor.execute(query, params)
            
            # For SELECT queries, return results
            if query.strip().upper().startswith('SELECT'):
                return self.cursor.fetchall()
            
            # For non-SELECT queries, commit the transaction
            self.conn.commit()
            return None
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def is_connected(self) -> bool:
        """
        Check if the database connection is open and valid.
        
        Returns:
            True if the connection is open and valid, False otherwise
        """
        if self.conn is None or self.cursor is None:
            return False
        try:
            # Check if connection is open
            if getattr(self.conn, '_closed', True):
                return False
            # Test connection with a simple query
            self.cursor.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if the table exists, False otherwise
        """
        try:
            # Use case-sensitive table name comparison for SingleStore
            result = self.execute_query(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = BINARY %s",
                (table_name,)
            )
            return result[0][0] > 0
        except Exception as e:
            logger.error("Error checking table existence: %s", str(e))
            return False

    def create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        try:
            # Create Document_Embeddings table
            if not self.table_exists("Document_Embeddings"):
                create_embeddings_table = """
                CREATE TABLE Document_Embeddings (
                    embedding_id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    doc_id BIGINT NOT NULL,
                    content TEXT,
                    embedding VECTOR(1536)
                )
                """
                self.execute_query(create_embeddings_table)
                
                # Add vector index
                vector_index = """
                ALTER TABLE Document_Embeddings
                    ADD VECTOR INDEX idx_vec (embedding) 
                    INDEX_OPTIONS '{"index_type": "HNSW_FLAT", "metric_type": "EUCLIDEAN_DISTANCE"}'
                """
                self.execute_query(vector_index)
                
                # Add fulltext index
                fulltext_index = """
                ALTER TABLE Document_Embeddings
                    ADD FULLTEXT INDEX content_ft_idx (content) USING VERSION 2
                """
                self.execute_query(fulltext_index)
                logger.info("Created Document_Embeddings table with indexes")
            
            # Create Documents table
            if not self.table_exists("Documents"):
                create_documents = """
                CREATE TABLE Documents (
                    doc_id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    filename VARCHAR(255) NOT NULL,
                    file_size BIGINT,
                    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status ENUM('processing', 'processed', 'error') DEFAULT 'processing',
                    error_message TEXT,
                    title VARCHAR(255),
                    author VARCHAR(100),
                    publish_date DATE,
                    source VARCHAR(255)
                )
                """
                self.execute_query(create_documents)
                logger.info("Created Documents table")
            
            # Create Relationships table
            if not self.table_exists("Relationships"):
                create_relationships = """
                CREATE TABLE Relationships (
                    relationship_id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    source_entity_id BIGINT NOT NULL,
                    target_entity_id BIGINT NOT NULL,
                    relation_type VARCHAR(100),
                    doc_id BIGINT,
                    KEY (source_entity_id) USING HASH,
                    KEY (target_entity_id) USING HASH,
                    KEY (doc_id)
                )
                """
                self.execute_query(create_relationships)
                logger.info("Created Relationships table")
            
            # Create Entities table
            if not self.table_exists("Entities"):
                create_entities = """
                CREATE TABLE Entities (
                    entity_id BIGINT NOT NULL AUTO_INCREMENT,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    aliases JSON,
                    category VARCHAR(100),
                    PRIMARY KEY (entity_id, name),
                    SHARD KEY (entity_id, name)
                )
                """
                self.execute_query(create_entities)
                logger.info("Created Entities table")
                
        except Exception as e:
            logger.error("Failed to create tables: %s", str(e))
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # An exception occurred, rollback any changes
            if self.conn:
                self.conn.rollback()
        else:
            # No exception, commit any pending changes
            if self.conn and not getattr(self.conn, '_closed', True):
                self.conn.commit()
        self.disconnect()


def test_connection():
    """
    Test the database connection and run a simple query.
    This function will:
    1. Try to establish a connection
    2. Run a simple SELECT query to verify query execution
    3. Print the database version and current time
    """
    try:
        with DatabaseConnection() as db:
            # Test connection status
            is_connected = db.is_connected()
            print(f"Connection Status: {'✅ Connected' if is_connected else '❌ Not Connected'}")
            
            # Get database version
            version = db.execute_query("SELECT VERSION()")
            print(f"Database Version: {version[0][0] if version else 'Unknown'}")
            
            # Get current timestamp
            time_result = db.execute_query("SELECT NOW()")
            print(f"Current Time: {time_result[0][0] if time_result else 'Unknown'}")
            
            # Get user and database info
            info = db.execute_query("""
                SELECT USER() AS user, DATABASE() AS db
            """)
            if info:
                print(f"Connected User: {info[0][0]}")
                print(f"Current Database: {info[0][1]}")
            
            print("✅ Connection test successful!")
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")


if __name__ == "__main__":
    test_connection()

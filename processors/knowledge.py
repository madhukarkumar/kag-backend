"""
Knowledge Graph Generator for SingleStore Document Processing.

This module extracts entities and relationships from document chunks using OpenAI's GPT models
and stores them in SingleStore for graph-based querying.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from db import DatabaseConnection
from core.config import config
from core.models import Entity, Relationship

# Set up logging
logger = logging.getLogger(__name__)

# Load entity extraction prompt
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'search', 'prompts', 'entity_extraction_prompt.txt')
with open(PROMPT_PATH, 'r') as f:
    ENTITY_EXTRACTION_PROMPT = f.read()

class KnowledgeGraphGenerator:
    """Generates knowledge graphs from document chunks using OpenAI."""
    
    def __init__(self, debug_output: bool = False):
        """
        Initialize the KnowledgeGraphGenerator with necessary configurations.
        
        Args:
            debug_output: If True, save extracted knowledge to local JSON files
        """
        # Load environment variables
        load_dotenv(override=True)
        
        # Set up OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Get entity extraction config
        self.extraction_config = config.knowledge_creation['entity_extraction']
        
        # Debug configuration
        self.debug_output = debug_output
        self.debug_dir = "debug_output"
        if debug_output:
            os.makedirs(self.debug_dir, exist_ok=True)
        
    def extract_knowledge_sync(self, text: str) -> Dict[str, Any]:
        """Extract knowledge from text synchronously."""
        logger.info("Extracting knowledge from text")
        logger.info(f"Text length: {len(text)} characters")
        
        # Build the prompt
        prompt = config.get_extraction_prompt(text)
        
        # Create API parameters with only supported parameters
        api_params = {
            "model": self.extraction_config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": self.extraction_config['system_prompt']
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "medium"
        }
        
        # Make synchronous API call
        try:
            response = self.openai_client.chat.completions.create(**api_params)
            
            if not response.choices:
                logger.error("No response from OpenAI")
                return {"entities": [], "relationships": []}
            
            # Get the response content and ensure it's clean
            content = response.choices[0].message.content.strip()
            logger.debug(f"Raw response: {content}")
            
            try:
                # Parse the JSON response
                knowledge = json.loads(content)
                
                # Ensure required structure
                if not isinstance(knowledge, dict):
                    logger.error("Response is not a dictionary")
                    return {"entities": [], "relationships": []}
                
                knowledge.setdefault("entities", [])
                knowledge.setdefault("relationships", [])
                
                # Validate entities and relationships with enhanced checks
                valid_entities = []
                for entity in knowledge["entities"]:
                    if isinstance(entity, dict) and "name" in entity and "type" in entity:
                        # Convert type to category for database
                        entity["category"] = entity["type"]
                        
                        # Ensure description meets minimum requirements
                        description = entity.get("description", "").strip()
                        if description and len(description) >= self.extraction_config.get('min_description_length', 50):
                            entity["description"] = description
                        else:
                            # Generate a basic description from context if missing
                            entity["description"] = self._generate_basic_description(entity, text)
                        
                        # Enhanced metadata
                        metadata = entity.get("metadata", {})
                        metadata.update({
                            "confidence": metadata.get("confidence", 0.7),
                            "context_relevance": metadata.get("context_relevance", 0.7),
                            "description_quality": self._calculate_description_quality(entity["description"])
                        })
                        entity["metadata"] = metadata
                        
                        # Handle aliases
                        entity.setdefault("aliases", [])
                        
                        valid_entities.append(entity)
                
                valid_relationships = []
                for rel in knowledge["relationships"]:
                    if isinstance(rel, dict) and all(k in rel for k in ["source", "target", "type"]):
                        # Convert type to relation_type for database
                        rel["relation_type"] = rel["type"]
                        
                        # Add relationship description if missing
                        if not rel.get("description"):
                            rel["description"] = self._generate_relationship_description(rel)
                        
                        # Enhanced metadata
                        metadata = rel.get("metadata", {})
                        metadata.update({
                            "confidence": metadata.get("confidence", 0.7),
                            "context_relevance": metadata.get("context_relevance", 0.7)
                        })
                        rel["metadata"] = metadata
                        
                        valid_relationships.append(rel)
                
                knowledge["entities"] = valid_entities
                knowledge["relationships"] = valid_relationships
                
                return knowledge
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Problematic content: {content}")
                return {"entities": [], "relationships": []}
            
        except Exception as e:
            logger.error(f"Error in knowledge extraction: {str(e)}")
            return {"entities": [], "relationships": []}

    def _calculate_description_quality(self, description: str) -> float:
        """Calculate a quality score for an entity description."""
        if not description:
            return 0.0
            
        # Basic quality metrics
        length_score = min(len(description) / 200, 1.0)  # Normalize by ideal length
        has_context = any(word in description.lower() for word in ["is", "was", "are", "means", "refers"])
        has_details = len(description.split()) >= 10
        
        # Combine scores
        quality_score = (length_score + float(has_context) + float(has_details)) / 3
        return round(quality_score, 2)

    def _generate_basic_description(self, entity: Dict, context: str) -> str:
        """Generate a basic description for an entity from context."""
        name = entity["name"]
        category = entity["category"]
        
        # Find the sentence containing the entity name
        sentences = context.split('.')
        relevant_sentences = [s for s in sentences if name.lower() in s.lower()]
        
        if relevant_sentences:
            # Use the most relevant sentence as description
            description = relevant_sentences[0].strip()
        else:
            # Fallback to a basic description
            description = f"A {category.lower()} mentioned in the document"
            
        return description

    def _generate_relationship_description(self, relationship: Dict) -> str:
        """Generate a description for a relationship."""
        source = relationship["source"]
        target = relationship["target"]
        rel_type = relationship["relation_type"]
        
        return f"Describes how {source} {rel_type.lower()} {target}"

    def merge_entity_info(self, existing_entity: Dict, new_entity: Dict) -> Dict:
        """
        Merge information from multiple occurrences of the same entity.
        
        Args:
            existing_entity: Currently stored entity information
            new_entity: New entity information to merge
            
        Returns:
            Dict containing merged entity information
        """
        # Get descriptions
        existing_desc = existing_entity.get("description", "").strip()
        new_desc = new_entity.get("description", "").strip()
        
        # Merge descriptions intelligently
        if existing_desc and new_desc:
            # Use the higher quality description
            existing_quality = self._calculate_description_quality(existing_desc)
            new_quality = self._calculate_description_quality(new_desc)
            
            if new_quality > existing_quality:
                merged_description = new_desc
            else:
                merged_description = existing_desc
        else:
            merged_description = existing_desc or new_desc
        
        # Merge and deduplicate aliases
        merged_aliases = list(set(
            existing_entity.get("aliases", []) + 
            new_entity.get("aliases", [])
        ))
        
        # Keep original category unless it's empty
        category = existing_entity.get("category") or new_entity.get("category", "")
        
        # Merge metadata
        existing_meta = existing_entity.get("metadata", {})
        new_meta = new_entity.get("metadata", {})
        merged_metadata = {
            "confidence": max(
                existing_meta.get("confidence", 0.0),
                new_meta.get("confidence", 0.0)
            ),
            "context_relevance": max(
                existing_meta.get("context_relevance", 0.0),
                new_meta.get("context_relevance", 0.0)
            ),
            "description_quality": self._calculate_description_quality(merged_description)
        }
        
        return {
            "name": existing_entity["name"],
            "category": category,
            "description": merged_description,
            "aliases": merged_aliases,
            "metadata": merged_metadata
        }

    def save_debug_output(self, data: Dict, doc_id: int, chunk_id: Optional[int] = None) -> None:
        """
        Save extracted knowledge to a JSON file for debugging.
        
        Args:
            data: Dictionary containing extracted knowledge
            doc_id: Document identifier
            chunk_id: Optional chunk identifier
        """
        if not self.debug_output:
            return
            
        try:
            # Create a filename based on doc_id and chunk_id
            filename = f"doc_{doc_id}"
            if chunk_id is not None:
                filename += f"_chunk_{chunk_id}"
            filename += f"_{int(datetime.now().timestamp())}.json"
            
            filepath = os.path.join(self.debug_dir, filename)
            
            # Add metadata to the output
            debug_data = {
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "timestamp": datetime.now().isoformat(),
                },
                "extracted_data": data
            }
            
            # Save to file with pretty printing
            with open(filepath, 'w') as f:
                json.dump(debug_data, f, indent=2)
            
            logger.debug(f"Debug output saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug output: {str(e)}")
    
    def store_knowledge(self, knowledge: Dict, db: DatabaseConnection) -> None:
        """
        Store extracted knowledge in SingleStore.
        
        Args:
            knowledge: Dict containing entities and relationships
            db: Database connection instance
        """
        try:
            # Start transaction
            db.execute_query("START TRANSACTION")
            
            try:
                # Process and deduplicate entities first
                unique_entities = {}
                for entity in knowledge["entities"]:
                    name = entity["name"]
                    if name in unique_entities:
                        # Merge with existing entity
                        existing = unique_entities[name]
                        merged = self.merge_entity_info(existing, entity)
                        unique_entities[name] = merged
                    else:
                        unique_entities[name] = entity
                
                # Now store the unique entities
                for entity in unique_entities.values():
                    # Check if entity exists in database
                    check_entity_query = """
                    SELECT entity_id, name, description, aliases, category 
                    FROM Entities 
                    WHERE name = %s
                    """
                    existing = db.execute_query(check_entity_query, (entity["name"],))
                    
                    if existing:
                        # Merge with existing database entity
                        existing_entity = {
                            "entity_id": existing[0][0],
                            "name": existing[0][1],
                            "description": existing[0][2],
                            "aliases": existing[0][3] if isinstance(existing[0][3], list) else json.loads(existing[0][3]) if existing[0][3] else [],
                            "category": existing[0][4]
                        }
                        merged = self.merge_entity_info(existing_entity, entity)
                        
                        # Update existing entity
                        update_entity_query = """
                        UPDATE Entities 
                        SET description = %s,
                            aliases = %s,
                            category = %s
                        WHERE entity_id = %s
                        """
                        db.execute_query(
                            update_entity_query,
                            (
                                merged["description"],
                                json.dumps(merged["aliases"]),
                                merged["category"],
                                existing_entity["entity_id"]
                            )
                        )
                        logger.info(f"Updated existing entity: {merged['name']} (ID: {existing_entity['entity_id']})")
                    else:
                        # Insert new entity
                        insert_entity_query = """
                        INSERT INTO Entities (name, description, aliases, category)
                        VALUES (%s, %s, %s, %s)
                        """
                        db.execute_query(
                            insert_entity_query,
                            (
                                entity["name"],
                                entity.get("description", ""),  # Use empty string as default
                                json.dumps(entity.get("aliases", [])),
                                entity["category"]
                            )
                        )
                        logger.info(f"Inserted new entity: {entity['name']}")
                
                # Store relationships
                for rel in knowledge["relationships"]:
                    insert_rel_query = """
                    INSERT INTO Relationships 
                    (source_entity_id, target_entity_id, relation_type)
                    SELECT DISTINCT
                        s.entity_id, 
                        t.entity_id, 
                        %s
                    FROM Entities s
                    JOIN Entities t
                    WHERE s.name = %s AND t.name = %s
                    """
                    db.execute_query(
                        insert_rel_query,
                        (
                            rel["relation_type"],
                            rel["source"],
                            rel["target"]
                        )
                    )
                    logger.info(f"Stored relationship: {rel['source']} -> {rel['target']}")
                
                # Commit transaction
                db.execute_query("COMMIT")
                logger.info("Transaction committed successfully")
                
            except Exception as e:
                # Rollback transaction on error
                db.execute_query("ROLLBACK")
                logger.error(f"Transaction rolled back due to error: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}", exc_info=True)
            raise

    def process_document(self, doc_id: int) -> None:
        """
        Process all chunks from a document to extract and store knowledge.
        
        Args:
            doc_id: Document identifier to process
        """
        try:
            with DatabaseConnection() as db:
                # Get all chunks for the document
                query = """
                SELECT content, embedding_id 
                FROM Document_Embeddings 
                WHERE doc_id = %s
                """
                chunks = db.execute_query(query, (doc_id,))
                
                # Process each chunk
                for chunk in chunks:
                    chunk_text = chunk[0]
                    chunk_id = chunk[1]
                    
                    # Extract knowledge
                    logger.info(f"Processing chunk {chunk_id} from document {doc_id}")
                    knowledge = self.extract_knowledge_sync(chunk_text)
                    
                    # Save debug output if enabled
                    self.save_debug_output(knowledge, doc_id, chunk_id)
                    
                    # Store knowledge
                    self.store_knowledge(knowledge, db)
                    
                logger.info(f"Successfully processed document {doc_id}")
                
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise

def generate_knowledge_graph(doc_id: int, debug_output: bool = False) -> None:
    """
    Generate knowledge graph for a document.
    
    Args:
        doc_id: Document ID to process
        debug_output: Whether to save debug output
    """
    generator = KnowledgeGraphGenerator(debug_output=debug_output)
    generator.process_document(doc_id)

def main():
    """Command line interface for knowledge graph generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate knowledge graph from documents")
    parser.add_argument("--doc_id", type=int, required=True, help="Document ID to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug output to JSON files")
    
    args = parser.parse_args()
    
    try:
        generate_knowledge_graph(args.doc_id, args.debug)
        logger.info("Knowledge graph generation completed successfully")
        
    except Exception as e:
        logger.error(f"Knowledge graph generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

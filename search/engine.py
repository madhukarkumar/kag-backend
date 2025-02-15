"""
RAG Query System for SingleStore Knowledge Graph.

This module implements a hybrid search system that combines:
1. Vector similarity search
2. Full-text search
3. Knowledge graph traversal
to answer natural language queries with citations.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from db import DatabaseConnection
from core.models import Entity, Relationship, SearchResult, SearchResponse
from core.config import config
import re
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """Implements hybrid search combining vector similarity, text search, and knowledge graph."""
    
    def __init__(self, debug_output: bool = False):
        """
        Initialize the RAG Query Engine.
        
        Args:
            debug_output: If True, enable debug output mode
        """
        # Load environment variables
        load_dotenv(override=True)
        logger.info("Environment variables loaded")
        
        # Get configuration
        self.search_config = config.retrieval['search']
        self.response_config = config.retrieval['response_generation']
        
        # Set up OpenAI client for embeddings (required)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"OpenAI API Key present: {bool(self.openai_api_key)}")
        
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is missing")
            raise ValueError("OPENAI_API_KEY environment variable is required for embeddings")
            
        self.embedding_client = OpenAI()  # Will use OPENAI_API_KEY from environment
        
        # Set up response generation client (OpenAI or Groq)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        groq_base_url = self.response_config.get('groq_base_url')
        
        # Use Groq for response generation if configured
        if self.groq_api_key and groq_base_url:
            logger.info("Using Groq API for response generation")
            self.response_client = OpenAI(
                base_url=groq_base_url,
                api_key=self.groq_api_key
            )
        else:
            logger.info("Using OpenAI API for response generation")
            self.response_client = self.embedding_client  # Use same OpenAI client
        
        # Debug configuration
        self.debug_output = debug_output
        self.debug_dir = "debug_output"
        if self.debug_output:
            os.makedirs(self.debug_dir, exist_ok=True)

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the query text."""
        try:
            response = self.embedding_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            raise

    def vector_search(self, db: DatabaseConnection, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Perform vector similarity search."""
        try:
            # Format vector for SQL
            vector_param = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
            
            # Set vector parameter
            db.execute_query("SET @qvec = %s :> VECTOR(1536);", (vector_param,))
            
            # Execute search
            vector_search_sql = """
                SELECT doc_id, content, (embedding <*> @qvec) AS score
                FROM Document_Embeddings
                ORDER BY score DESC
                LIMIT %s;
            """
            
            results = db.execute_query(vector_search_sql, (limit,))
            
            return [
                {"doc_id": r[0], "content": r[1], "score": r[2]}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise

    def text_search(self, db: DatabaseConnection, query: str, limit: int = 10) -> List[Dict]:
        """Perform full-text keyword search using Full-Text Search Version 2."""
        try:
            # Limit query length to prevent parser errors
            max_query_length = self.search_config.get('max_query_length', 500)
            if len(query) > max_query_length:
                # Take first N chars of original query + important keywords
                words = query.split()
                base_query = ' '.join(words[:10])  # First 10 words
                important_words = [w for w in words[10:] if len(w) > 3][:20]  # Up to 20 important keywords
                query = f"{base_query} {' '.join(important_words)}"
                logger.info(f"Query truncated to: {query}")

            # Replace hyphens with spaces for better matching
            query = query.replace('-', ' ')
            
            # Extract key phrases (quoted terms)
            key_phrases = re.findall(r'"([^"]*)"', query)
            remaining_text = re.sub(r'"[^"]*"', '', query)
            
            # Split remaining text into terms and clean them
            terms = set()  # Use set to deduplicate
            multi_word_terms = set()  # For terms with spaces
            
            for t in remaining_text.split():
                t = t.strip()
                if t:
                    # Escape special characters that could break the parser
                    t = re.sub(r'[+\-=&|><!(){}[\]^"~*?:/\\]', ' ', t)
                    t = t.strip()
                    if t:
                        if ' ' in t:
                            multi_word_terms.add(t)
                        elif len(t) > 2:  # Only add single words if longer than 2 chars
                            terms.add(t.lower())  # Normalize to lowercase
            
            # Build search expression with semantic operators
            search_parts = []
            
            # Add exact phrases with high weight
            for phrase in key_phrases:
                if phrase:
                    # Escape special characters in phrases
                    phrase = re.sub(r'[+\-=&|><!(){}[\]^"~*?:/\\]', ' ', phrase)
                    phrase = phrase.strip()
                    if phrase:
                        search_parts.append(f'content:"{phrase}">>{self.search_config.get("exact_phrase_weight", 2.0)}')
            
            # Add multi-word terms as phrases
            for term in multi_word_terms:
                search_parts.append(f'content:"{term}">>{self.search_config.get("exact_phrase_weight", 2.0)}')
            
            # Add individual terms with proximity search
            if terms:
                # Group terms for proximity search (limited to 5 terms to prevent complexity)
                proximity_terms = list(terms)[:5]
                terms_str = ' '.join(proximity_terms)
                if terms_str:
                    search_parts.append(f'content:"{terms_str}"~{self.search_config.get("proximity_distance", 5)}')
                
                # Add individual terms with lower weight (avoiding duplicates)
                for term in terms:
                    search_parts.append(f'content:{term}>>{self.search_config.get("single_term_weight", 1.5)}')
            
            # Return empty results if no search terms found
            if not search_parts:
                logger.info("No valid search terms found, returning empty results")
                return []
            
            # Combine all parts with OR (limit number of clauses)
            search_parts = search_parts[:50]  # Limit to prevent too complex queries
            formatted_query = ' OR '.join(search_parts)
            logger.info(f"Text search query: {formatted_query}")
            
            sql = """
                SELECT 
                    doc_id,
                    content,
                    MATCH(TABLE Document_Embeddings) AGAINST(%s) as text_score
                FROM Document_Embeddings 
                WHERE MATCH(TABLE Document_Embeddings) AGAINST(%s)
                ORDER BY text_score DESC
                LIMIT %s;
            """
            
            results = db.execute_query(sql, (formatted_query, formatted_query, limit))
            
            return [
                {
                    "doc_id": r[0],
                    "content": r[1],
                    "text_score": float(r[2])
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []

    def merge_search_results(
            self, 
            vector_results: List[Dict], 
            text_results: List[Dict],
            vector_weight: float = None
        ) -> List[Dict]:
        """Merge and rank results from vector and text searches."""
        try:
            # Use config weight if not specified
            if vector_weight is None:
                vector_weight = self.search_config.get('vector_weight', 0.7)
            text_weight = 1 - vector_weight
            
            logger.info(f"Merging with weights - vector: {vector_weight}, text: {text_weight}")
            logger.info(f"Input results - vector: {len(vector_results)}, text: {len(text_results)}")
            
            # Normalize scores
            vec_max = max([r.get('score', 0) for r in vector_results]) if vector_results else 1.0
            txt_max = max([r.get('text_score', 0) for r in text_results]) if text_results else 1.0
            logger.info(f"Max scores - vector: {vec_max}, text: {txt_max}")
            
            # Create a map of doc_id to result for both result sets
            vector_map = {r['doc_id']: {
                **r,
                'vector_score': r.get('score', 0) / vec_max if vec_max else 0
            } for r in vector_results}
            
            text_map = {r['doc_id']: {
                **r,
                'text_score': r.get('text_score', 0) / txt_max if txt_max else 0
            } for r in text_results}
            
            logger.info(f"Unique docs - vector: {len(vector_map)}, text: {len(text_map)}")
            
            # Get all unique doc_ids
            all_doc_ids = set(vector_map.keys()) | set(text_map.keys())
            logger.info(f"Total unique docs before merging: {len(all_doc_ids)}")
            
            # Combine scores
            merged = []
            for doc_id in all_doc_ids:
                vector_result = vector_map.get(doc_id, {'vector_score': 0})
                text_result = text_map.get(doc_id, {'text_score': 0})
                
                combined_score = (
                    vector_weight * vector_result.get('vector_score', 0) +
                    text_weight * text_result.get('text_score', 0)
                )
                
                # Only include results that meet the minimum score threshold
                min_score = self.search_config.get('min_score_threshold', 0.15)
                if combined_score >= min_score:
                    merged.append({
                        'doc_id': doc_id,
                        'content': vector_result.get('content') or text_result.get('content'),
                        'vector_score': vector_result.get('vector_score', 0),
                        'text_score': text_result.get('text_score', 0),
                        'combined_score': combined_score
                    })
            
            # Sort by combined score
            merged.sort(key=lambda x: x['combined_score'], reverse=True)
            logger.info(f"Total results after merging and filtering: {len(merged)}")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            raise

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve search accuracy:
        1. Remove special characters but keep important punctuation
        2. Normalize whitespace
        3. Extract key concepts and expand with synonyms
        """
        # Clean and normalize
        query = re.sub(r'[^\w\s?.!,]', ' ', query)
        query = ' '.join(query.split())
        
        # Extract key concepts using OpenAI or Groq
        try:
            # Get model based on configuration
            if self.groq_api_key and self.response_config.get('groq_base_url'):
                model = self.response_config.get('query_expansion', {}).get('groq_model', 'mixtral-8x7b-32768')
                logger.info(f"Using Groq model for query expansion: {model}")
                response = self.response_client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "system",
                        "content": "Extract and expand key concepts from the query. Format: concept1 | synonym1, synonym2 | concept2 | synonym1, synonym2. Limit your answers to less than 300 words"
                    }, {
                        "role": "user",
                        "content": query
                    }],
                    temperature=0.0
                )
            else:
                # Use OpenAI for query expansion
                model = self.response_config.get('query_expansion', {}).get('openai_model', 'gpt-4o')
                logger.info(f"Using OpenAI model for query expansion: {model}")
                response = self.embedding_client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "system",
                        "content": "Extract and expand key concepts from the query. Format: concept1 | synonym1, synonym2 | concept2 | synonym1, synonym2"
                    }, {
                        "role": "user",
                        "content": query
                    }],
                    temperature=0.0
                )
            
            # Parse expanded concepts
            expanded = response.choices[0].message.content
            expanded_terms = []
            for concept_group in expanded.split('|'):
                expanded_terms.extend(t.strip() for t in concept_group.split(','))
            
            # Combine original query with expanded terms
            enhanced_query = f"{query} {' '.join(expanded_terms)}"
            return enhanced_query.strip()
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return query

    def query(self, query_text: str, top_k: int = 5) -> SearchResponse:
        """Execute a hybrid search query combining vector and text search."""
        try:
            # Preprocess and enhance query
            enhanced_query = self.preprocess_query(query_text)
            logger.info(f"Enhanced query: {enhanced_query}")
            
            with DatabaseConnection() as db:
                # Get results from both search methods
                config_top_k = self.search_config.get('top_k', 20)  # Use config value, default to 20
                logger.info(f"Using config top_k: {config_top_k}")
                
                vector_results = self.vector_search(db, self.get_query_embedding(enhanced_query), limit=config_top_k)
                logger.info(f"Vector search returned {len(vector_results)} results")
                
                text_results = self.text_search(db, enhanced_query, limit=config_top_k)
                logger.info(f"Text search returned {len(text_results)} results")
                
                # Merge results
                merged_results = self.merge_search_results(vector_results, text_results)
                logger.info(f"After merging: {len(merged_results)} results")
                
                # Sort by combined score and limit to top_k
                merged_results.sort(key=lambda x: x['combined_score'], reverse=True)
                merged_results = merged_results[:top_k]
                logger.info(f"After limiting to top_k: {len(merged_results)} results")
                
                # Build context with SearchResult objects
                formatted_results = []
                for doc in merged_results:
                    # Get entities for this content
                    entities = self.get_entities_for_content(db, doc["content"])
                    logger.info(f"Found {len(entities)} entities for doc {doc['doc_id']}")
                    
                    # Get relationships for these entities
                    relationships = self.get_relationships(db, [e.id for e in entities])
                    logger.info(f"Found {len(relationships)} relationships for doc {doc['doc_id']}")
                    
                    # Create SearchResult object
                    search_result = SearchResult(
                        doc_id=doc["doc_id"],
                        content=doc["content"],
                        vector_score=doc.get("vector_score", 0.0),
                        text_score=doc.get("text_score", 0.0),
                        combined_score=doc["combined_score"],
                        entities=entities,
                        relationships=relationships
                    )
                    formatted_results.append(search_result)
                
                # Create SearchResponse
                response = SearchResponse(
                    query=query_text,
                    results=formatted_results,
                    generated_response=self.generate_response(query_text, {"results": formatted_results}),
                    execution_time=0.0  # We'll set this in the API layer
                )
                
                logger.info(f"Final response has {len(response.results)} results")
                return response
                
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}", exc_info=True)
            raise  # Let the API layer handle the error

    def _build_prompt(self, query: str, context: Dict) -> str:
        """Build prompt for the LLM using retrieved context."""
        PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'prompts', 'rag.md')
        with open(PROMPT_PATH, 'r') as f:
            RAG_PROMPT_TEMPLATE = f.read()
        return config.get_response_prompt(query, context, RAG_PROMPT_TEMPLATE)
        
    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a response using the language model."""
        try:
            # Get model configuration
            model = self.response_config.get('model', 'gpt-4o')
            model_config = self.response_config.get('model_config', {}).get(model, {})
            max_tokens = model_config.get('max_tokens', self.response_config.get('max_tokens', 1000))
            temperature = model_config.get('temperature', self.response_config.get('temperature', 0.3))
            
            # If using Groq, ensure we use a supported model
            if self.groq_api_key and self.response_config.get('groq_base_url'):
                if not any(model.startswith(prefix) for prefix in ['mixtral-']):
                    logger.warning(f"Model {model} not supported by Groq, falling back to mixtral-8x7b-32768")
                    model = 'mixtral-8x7b-32768'
            
            response = self.response_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Query: {query}\n\nContext: {context}"}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def get_entities_for_content(self, db: DatabaseConnection, content: str) -> List[Entity]:
        """Find entities mentioned in the content."""
        try:
            # Extract potential entity names using simple word-based approach
            # Remove special characters and split into words
            words = re.sub(r'[^\w\s]', ' ', content).split()
            # Get unique words, filter out common words and very short terms
            unique_terms = set(word.lower() for word in words if len(word) > 2)
            
            # Format terms for SQL query
            terms_str = ', '.join(f"'{term}'" for term in unique_terms)
            
            if not terms_str:
                return []
            
            # Query using schema-defined columns
            sql = """
                SELECT DISTINCT
                    entity_id,
                    name,
                    category,
                    COALESCE(description, '') as description,
                    COALESCE(aliases, '[]') as aliases
                FROM Entities
                WHERE LOWER(name) IN (%s)
                LIMIT 10;
            """ % terms_str
            
            logger.debug(f"Executing entity search SQL: {sql}")
            results = db.execute_query(sql)
            
            entities = []
            for r in results:
                try:
                    # Parse the JSON string into a Python list
                    aliases = json.loads(r[4]) if r[4] else []
                    
                    entity = Entity(
                        entity_id=r[0],  # Use entity_id to match the Field alias
                        name=r[1],
                        category=r[2],
                        description=r[3],
                        aliases=aliases
                    )
                    entities.append(entity)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse aliases JSON for entity {r[0]}: {e}")
                    # Continue with empty aliases if JSON parsing fails
                    entity = Entity(
                        entity_id=r[0],  # Use entity_id to match the Field alias
                        name=r[1],
                        category=r[2],
                        description=r[3],
                        aliases=[]
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error finding entities: {str(e)}", exc_info=True)
            return []

    def get_relationships(self, db: DatabaseConnection, entity_ids: List[int]) -> List[Relationship]:
        """Get relationships for the given entities."""
        try:
            if not entity_ids:
                return []
            
            # Format entity IDs for SQL
            ids_str = ', '.join(str(id) for id in entity_ids)
            
            # Query using schema-defined columns
            sql = """
                SELECT DISTINCT
                    source_entity_id,
                    target_entity_id,
                    relation_type,
                    doc_id
                FROM Relationships
                WHERE source_entity_id IN (%s)
                OR target_entity_id IN (%s)
                LIMIT 20;
            """ % (ids_str, ids_str)
            
            results = db.execute_query(sql)
            
            return [
                Relationship(
                    source_entity_id=r[0],
                    target_entity_id=r[1],
                    relation_type=r[2],
                    metadata={"doc_id": r[3]} if r[3] else {}
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}", exc_info=True)
            return []

    def save_debug_output(self, stage: str, data: Dict) -> None:
        """Save intermediate results for debugging."""
        if not self.debug_output:
            return
            
        try:
            filename = f"rag_query_{stage}_{int(time.time())}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Debug output for {stage} saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug output: {str(e)}")

    def hybrid_search(self, db: DatabaseConnection, query: str, top_k: int = 5) -> List[Dict]:
        # Add early exit conditions
        vector_results = self.vector_search(db, self.get_query_embedding(query), limit=top_k*3)
        if len(vector_results) >= top_k and \
           vector_results[0]['score'] > 0.9:  # High confidence match
            return vector_results[:top_k]
        
        # Otherwise proceed with hybrid
        text_results = self.text_search(db, query, limit=top_k*2)
        return self.merge_search_results(vector_results, text_results)

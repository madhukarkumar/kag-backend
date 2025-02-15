CREATE TABLE Document_Embeddings (
  embedding_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  doc_id       BIGINT NOT NULL,
  content      TEXT,
  embedding    VECTOR(1536),
  chunk_metadata_id BIGINT,
  SORT KEY(),  -- Ensure this is a columnstore table&#8203;:contentReference[oaicite:11]{index=11}
  FULLTEXT USING VERSION 2 content_ft_idx (content),  -- Full-Text index (v2) on content&#8203;:contentReference[oaicite:12]{index=12}
  VECTOR INDEX embedding_vec_idx (embedding)          -- Vector index on embedding column&#8203;:contentReference[oaicite:13]{index=13}
    INDEX_OPTIONS '{ "index_type": "HNSW_FLAT", "metric_type": "DOT_PRODUCT" }',
  FOREIGN KEY (chunk_metadata_id) REFERENCES Chunk_Metadata(chunk_id)
);

ALTER TABLE Entities
  ADD FULLTEXT USING VERSION 2 ft_idx_name (name);

ALTER TABLE Entities ADD UNIQUE INDEX idx_entity_name (name);


CREATE TABLE Documents (
  doc_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(255),
  author VARCHAR(100),
  publish_date DATE,
  source VARCHAR(255)
  -- Other metadata fields (e.g. summary, URL) can be added as needed
);

CREATE TABLE Relationships (
  relationship_id BIGINT PRIMARY KEY AUTO_INCREMENT,
  source_entity_id BIGINT NOT NULL,
  target_entity_id BIGINT NOT NULL,
  relation_type VARCHAR(100),
  doc_id BIGINT,   -- reference to Documents.doc_id (not an enforced foreign key)
  KEY (source_entity_id) USING HASH,  -- index for quickly finding relationships by source
  KEY (target_entity_id) USING HASH,  -- index for quickly finding relationships by target
  KEY (doc_id)                      -- index for querying relationships by document
);

CREATE TABLE Entities (
    entity_id BIGINT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    aliases JSON,
    category VARCHAR(100),
    PRIMARY KEY (entity_id, name),
    SHARD KEY (entity_id, name),
    FULLTEXT USING VERSION 2 name_ft_idx (name)
);

CREATE TABLE Chunk_Metadata (
    chunk_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    doc_id BIGINT NOT NULL,
    position INT NOT NULL,
    section_path TEXT,
    prev_chunk_id BIGINT,
    next_chunk_id BIGINT,
    overlap_start_id BIGINT,
    overlap_end_id BIGINT,
    semantic_unit VARCHAR(255),
    structural_context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES Documents(doc_id),
    FOREIGN KEY (prev_chunk_id) REFERENCES Chunk_Metadata(chunk_id),
    FOREIGN KEY (next_chunk_id) REFERENCES Chunk_Metadata(chunk_id),
    FOREIGN KEY (overlap_start_id) REFERENCES Chunk_Metadata(chunk_id),
    FOREIGN KEY (overlap_end_id) REFERENCES Chunk_Metadata(chunk_id),
    SHARD KEY (doc_id)
);

CREATE TABLE ProcessingStatus (
    doc_id BIGINT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL UNIQUE,
    file_path VARCHAR(512) NOT NULL,
    file_size BIGINT NOT NULL,
    current_step ENUM('started', 'chunking', 'embeddings', 'entities', 'relationships', 'completed', 'failed') NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES Documents(doc_id) ON DELETE CASCADE
);

SHOW INDEXES FROM Document_Embeddings;

OPTIMIZE TABLE Document_Embeddings FLUSH;  -- Ensure recent data is indexed

SELECT
    doc_id,
    content,
    MATCH (TABLE Document_Embeddings) AGAINST ('How does SingleStore support hybrid search in RAG?') AS score
FROM Document_Embeddings
WHERE MATCH (TABLE Document_Embeddings) AGAINST ('How does SingleStore support hybrid search in RAG?')
ORDER BY score DESC
LIMIT 10;



----- Chunk related changes for better accuracy
-- Create Chunk_Metadata table
-- Create Chunk_Metadata table
CREATE TABLE Chunk_Metadata (
    chunk_id BIGINT NOT NULL AUTO_INCREMENT,
    doc_id BIGINT NOT NULL,
    position INT NOT NULL,
    section_path TEXT,
    prev_chunk_id BIGINT,
    next_chunk_id BIGINT,
    overlap_start_id BIGINT,
    overlap_end_id BIGINT,
    semantic_unit VARCHAR(255),
    structural_context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, chunk_id),  -- Include doc_id from SHARD KEY
    SHARD KEY (doc_id),
    SORT KEY(),  -- Columnstore table
    KEY (prev_chunk_id) USING HASH,
    KEY (next_chunk_id) USING HASH,
    KEY (overlap_start_id) USING HASH,
    KEY (overlap_end_id) USING HASH
);

-- Add chunk_metadata_id to Document_Embeddings
ALTER TABLE Document_Embeddings 
ADD COLUMN chunk_metadata_id BIGINT,
ADD KEY (chunk_metadata_id) USING HASH;

-- Create index on section_path
ALTER TABLE Chunk_Metadata 
ADD FULLTEXT USING VERSION 2 section_path_ft_idx (section_path);


-- First drop existing index
DROP INDEX embedding_vec_idx ON Document_Embeddings;

-- Then recreate with proper syntax
-- Then recreate with proper syntax
-- Recreate the vector index with supported index options only
   ALTER TABLE Document_Embeddings 
   ADD VECTOR INDEX embedding_vec_idx (embedding)
   INDEX_OPTIONS '{"index_type": "HNSW_FLAT", "metric_type": "DOT_PRODUCT", "M": 32, "efConstruction": 200}';
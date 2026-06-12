-- =============================================================
-- Cattle Biometric Identification Database Schema
-- PostgreSQL 15+ with pgvector extension
-- =============================================================

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================
-- CATTLE TABLE  – core metadata for each registered animal
-- =============================================================
CREATE TABLE IF NOT EXISTS cattle (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tag_id        VARCHAR(64) UNIQUE NOT NULL,         -- ear tag / RFID
    name          VARCHAR(128),                         -- optional name
    breed         VARCHAR(64),
    sex           VARCHAR(16) CHECK (sex IN ('Male', 'Female', 'Unknown')),
    date_of_birth DATE,
    farm_name     VARCHAR(128),
    farm_location VARCHAR(256),
    owner_name    VARCHAR(128),
    owner_contact VARCHAR(64),
    weight_kg     FLOAT,
    notes         TEXT,
    photo_path    TEXT,                                 -- stored relative path
    status        VARCHAR(32) DEFAULT 'active' CHECK (status IN ('active', 'sold', 'deceased')),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================
-- CATTLE EMBEDDINGS TABLE  – 256-d GNN embedding per image
-- =============================================================
CREATE TABLE IF NOT EXISTS cattle_embeddings (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cattle_id      UUID NOT NULL REFERENCES cattle(id) ON DELETE CASCADE,
    embedding      vector(256) NOT NULL,                -- L2-normalised GNN embedding
    image_path     TEXT,                                -- which muzzle image produced this embedding
    model_version  VARCHAR(64) DEFAULT 'CattleGNN-v1',
    extractor      VARCHAR(32) DEFAULT 'superpoint',    -- superpoint | sift
    num_keypoints  INT,
    confidence     FLOAT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate cosine similarity search
CREATE INDEX IF NOT EXISTS idx_cattle_embeddings_hnsw
    ON cattle_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- B-tree index for fast cattle_id lookups
CREATE INDEX IF NOT EXISTS idx_cattle_embeddings_cattle_id
    ON cattle_embeddings (cattle_id);

-- =============================================================
-- IDENTIFICATION LOGS TABLE  – every identification attempt
-- =============================================================
CREATE TABLE IF NOT EXISTS identification_logs (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_image    TEXT,                                -- uploaded image path
    matched_cattle UUID REFERENCES cattle(id),          -- NULL if no match
    similarity     FLOAT,                               -- top-1 cosine similarity
    threshold      FLOAT,                               -- threshold used
    accepted       BOOLEAN,                             -- similarity >= threshold
    top_k_results  JSONB,                               -- [{cattle_id, similarity, tag_id, name}]
    extractor      VARCHAR(32),
    model_version  VARCHAR(64),
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================
-- HELPER: update updated_at automatically
-- =============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_cattle_updated_at
    BEFORE UPDATE ON cattle
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================
-- VIEWS
-- =============================================================

-- Summary view: cattle + their embedding count
CREATE OR REPLACE VIEW cattle_summary AS
SELECT
    c.id,
    c.tag_id,
    c.name,
    c.breed,
    c.sex,
    c.farm_name,
    c.status,
    c.photo_path,
    COUNT(ce.id)::INT AS embedding_count,
    MAX(ce.created_at) AS last_enrolled,
    c.created_at
FROM cattle c
LEFT JOIN cattle_embeddings ce ON c.id = ce.cattle_id
GROUP BY c.id;

-- Stats view
CREATE OR REPLACE VIEW system_stats AS
SELECT
    (SELECT COUNT(*) FROM cattle WHERE status = 'active')::INT AS active_cattle,
    (SELECT COUNT(*) FROM cattle)::INT AS total_cattle,
    (SELECT COUNT(*) FROM cattle_embeddings)::INT AS total_embeddings,
    (SELECT COUNT(*) FROM identification_logs)::INT AS total_identifications,
    (SELECT COUNT(*) FROM identification_logs WHERE accepted = true)::INT AS successful_identifications;

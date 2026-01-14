-- Lenny's Podcast Knowledge Base Schema
-- 4 core tables: episodes, guests, episode_guests, transcript_chunks

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. GUESTS: Normalized guest information
CREATE TABLE IF NOT EXISTS guests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. EPISODES: Core episode metadata
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    youtube_url TEXT,
    video_id TEXT UNIQUE,
    description TEXT,
    duration_seconds INTEGER,
    duration_display TEXT,
    view_count INTEGER,
    transcript_raw TEXT,
    transcript_word_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. EPISODE_GUESTS: Many-to-many relationship
CREATE TABLE IF NOT EXISTS episode_guests (
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    guest_id UUID REFERENCES guests(id) ON DELETE CASCADE,
    PRIMARY KEY (episode_id, guest_id)
);

-- 4. TRANSCRIPT_CHUNKS: For RAG/semantic search
-- Gemini text-embedding-004 produces 768-dimensional vectors
-- Natural key: (episode_id, chunk_index) for idempotent upserts
CREATE TABLE IF NOT EXISTS transcript_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    speaker TEXT,
    timestamp_start TEXT,
    timestamp_seconds INTEGER,
    content TEXT NOT NULL,
    word_count INTEGER,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (episode_id, chunk_index)  -- Natural key for upsert
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transcript_chunks_episode ON transcript_chunks(episode_id);
CREATE INDEX IF NOT EXISTS idx_transcript_chunks_embedding ON transcript_chunks 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_episodes_view_count ON episodes(view_count DESC);
CREATE INDEX IF NOT EXISTS idx_guests_slug ON guests(slug);
CREATE INDEX IF NOT EXISTS idx_episodes_slug ON episodes(slug);

-- Function to search chunks by similarity
CREATE OR REPLACE FUNCTION search_chunks(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    episode_id UUID,
    episode_title TEXT,
    guest_name TEXT,
    speaker TEXT,
    content TEXT,
    timestamp_start TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tc.id AS chunk_id,
        tc.episode_id,
        e.title AS episode_title,
        g.name AS guest_name,
        tc.speaker,
        tc.content,
        tc.timestamp_start,
        1 - (tc.embedding <=> query_embedding) AS similarity
    FROM transcript_chunks tc
    JOIN episodes e ON tc.episode_id = e.id
    LEFT JOIN episode_guests eg ON e.id = eg.episode_id
    LEFT JOIN guests g ON eg.guest_id = g.id
    WHERE 1 - (tc.embedding <=> query_embedding) > match_threshold
    ORDER BY tc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Add missing unique constraint for idempotent upserts
-- Run this if you already created tables from 001_create_schema.sql before the constraint was added

-- Add unique constraint on transcript_chunks (episode_id, chunk_index)
-- This allows upserts without duplicates
ALTER TABLE transcript_chunks 
ADD CONSTRAINT transcript_chunks_episode_chunk_unique 
UNIQUE (episode_id, chunk_index);

"""
Ingest Lenny's Podcast transcripts into Supabase.

Parses markdown files, chunks transcripts by speaker turns,
generates Gemini embeddings, and pushes to Supabase.

Usage:
    python ingest_transcripts.py
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
from tqdm import tqdm

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")
GEMINI_API_KEY = os.getenv("GEMIMI_API_KEY")  # Note: matches user's spelling

EPISODES_PATH = Path(__file__).parent.parent / "lennys-podcast-transcripts" / "episodes"
EMBEDDING_MODEL = "models/text-embedding-004"
CHUNK_TARGET_WORDS = 400  # Target words per chunk
CHUNK_MAX_WORDS = 600     # Max words before forcing split
DEFAULT_LIMIT = 10        # Limit episodes for testing (set to None for all)


@dataclass
class TranscriptChunk:
    """A chunk of transcript with speaker and timestamp."""
    speaker: str
    timestamp_start: str
    timestamp_seconds: int
    content: str
    word_count: int


@dataclass
class EpisodeData:
    """Parsed episode data from markdown file."""
    slug: str
    title: str
    guest: str
    youtube_url: str
    video_id: str
    description: str
    duration_seconds: int
    duration_display: str
    view_count: int
    transcript_raw: str
    chunks: list[TranscriptChunk]


def parse_timestamp(ts: str) -> int:
    """Convert timestamp string (HH:MM:SS or MM:SS) to seconds."""
    parts = ts.strip("()").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def parse_transcript_file(filepath: Path) -> Optional[EpisodeData]:
    """Parse a transcript markdown file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Split frontmatter and content
    parts = content.split("---", 2)
    if len(parts) < 3:
        print(f"No frontmatter found in {filepath}")
        return None

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {filepath}: {e}")
        return None

    transcript_content = parts[2].strip()
    
    # Extract raw transcript (everything after "## Transcript")
    transcript_match = re.search(r"## Transcript\s*\n(.+)", transcript_content, re.DOTALL)
    transcript_raw = transcript_match.group(1).strip() if transcript_match else transcript_content

    # Parse speaker turns with timestamps
    # Pattern: "Speaker Name (HH:MM:SS):" or "(HH:MM:SS):"
    turn_pattern = re.compile(
        r"(?:^|\n)(?:([A-Za-z][A-Za-z\s\.]+?)\s*)?\((\d{1,2}:\d{2}:\d{2})\):\s*",
        re.MULTILINE
    )
    
    chunks = []
    current_speaker = "Unknown"
    matches = list(turn_pattern.finditer(transcript_raw))
    
    for i, match in enumerate(matches):
        speaker = match.group(1) or current_speaker
        speaker = speaker.strip()
        current_speaker = speaker
        timestamp = match.group(2)
        
        # Get content until next speaker or end
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(transcript_raw)
        content = transcript_raw[start_pos:end_pos].strip()
        
        if content:
            word_count = count_words(content)
            
            # If chunk is too large, split it
            if word_count > CHUNK_MAX_WORDS:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = []
                current_words = 0
                
                for sentence in sentences:
                    sentence_words = count_words(sentence)
                    if current_words + sentence_words > CHUNK_TARGET_WORDS and current_chunk:
                        chunk_content = " ".join(current_chunk)
                        chunks.append(TranscriptChunk(
                            speaker=speaker,
                            timestamp_start=timestamp,
                            timestamp_seconds=parse_timestamp(timestamp),
                            content=chunk_content,
                            word_count=count_words(chunk_content)
                        ))
                        current_chunk = [sentence]
                        current_words = sentence_words
                    else:
                        current_chunk.append(sentence)
                        current_words += sentence_words
                
                if current_chunk:
                    chunk_content = " ".join(current_chunk)
                    chunks.append(TranscriptChunk(
                        speaker=speaker,
                        timestamp_start=timestamp,
                        timestamp_seconds=parse_timestamp(timestamp),
                        content=chunk_content,
                        word_count=count_words(chunk_content)
                    ))
            else:
                chunks.append(TranscriptChunk(
                    speaker=speaker,
                    timestamp_start=timestamp,
                    timestamp_seconds=parse_timestamp(timestamp),
                    content=content,
                    word_count=word_count
                ))

    slug = filepath.parent.name
    
    return EpisodeData(
        slug=slug,
        title=frontmatter.get("title", ""),
        guest=frontmatter.get("guest", ""),
        youtube_url=frontmatter.get("youtube_url", ""),
        video_id=frontmatter.get("video_id", ""),
        description=frontmatter.get("description", ""),
        duration_seconds=int(frontmatter.get("duration_seconds", 0)),
        duration_display=frontmatter.get("duration", ""),
        view_count=int(frontmatter.get("view_count", 0)),
        transcript_raw=transcript_raw,
        chunks=chunks
    )


def parse_guest_names(guest_str: str) -> list[str]:
    """Parse guest names from string (handles multiple guests)."""
    # Split by common separators
    separators = [" and ", " & ", ", ", " with "]
    names = [guest_str]
    
    for sep in separators:
        new_names = []
        for name in names:
            new_names.extend(name.split(sep))
        names = new_names
    
    return [name.strip() for name in names if name.strip()]


def slugify(name: str) -> str:
    """Convert name to slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    return slug.strip("-")


class TranscriptIngester:
    """Handles ingestion of transcripts into Supabase."""
    
    def __init__(self):
        if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
            raise ValueError("Missing required environment variables")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        self.embedding_model = genai.GenerativeModel(EMBEDDING_MODEL)
        
        # Cache for guest IDs
        self.guest_cache: dict[str, str] = {}
    
    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Gemini."""
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    
    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def upsert_guest(self, name: str) -> str:
        """Upsert a guest and return their ID.
        
        Uses slug as natural key for idempotent upserts.
        """
        slug = slugify(name)
        
        if slug in self.guest_cache:
            return self.guest_cache[slug]
        
        # Upsert: insert or update on conflict with slug
        result = self.supabase.table("guests").upsert(
            {"name": name, "slug": slug},
            on_conflict="slug"
        ).execute()
        
        guest_id = result.data[0]["id"]
        self.guest_cache[slug] = guest_id
        return guest_id
    
    def upsert_episode(self, episode: EpisodeData) -> str:
        """Upsert an episode and return its ID.
        
        Uses slug as natural key for idempotent upserts.
        """
        episode_data = {
            "slug": episode.slug,
            "title": episode.title,
            "youtube_url": episode.youtube_url,
            "video_id": episode.video_id,
            "description": episode.description,
            "duration_seconds": episode.duration_seconds,
            "duration_display": episode.duration_display,
            "view_count": episode.view_count,
            "transcript_raw": episode.transcript_raw,
            "transcript_word_count": count_words(episode.transcript_raw)
        }
        
        # Upsert: insert or update on conflict with slug
        result = self.supabase.table("episodes").upsert(
            episode_data,
            on_conflict="slug"
        ).execute()
        
        return result.data[0]["id"]
    
    def link_episode_guests(self, episode_id: str, guest_ids: list[str]) -> None:
        """Link episode to guests (many-to-many).
        
        Uses (episode_id, guest_id) composite key for idempotent upserts.
        """
        for guest_id in guest_ids:
            # Upsert: insert or ignore on conflict (composite primary key)
            self.supabase.table("episode_guests").upsert(
                {"episode_id": episode_id, "guest_id": guest_id},
                on_conflict="episode_id,guest_id"
            ).execute()
    
    def upsert_chunks(self, episode_id: str, chunks: list[TranscriptChunk]) -> None:
        """Upsert transcript chunks with embeddings.
        
        Uses (episode_id, chunk_index) as natural key for idempotent upserts.
        """
        if not chunks:
            return
        
        # Generate embeddings in batches
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self.get_embeddings_batch(texts)
            
            # Upsert chunks with embeddings (using natural key: episode_id + chunk_index)
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_data = {
                    "episode_id": episode_id,
                    "chunk_index": i + j,
                    "speaker": chunk.speaker,
                    "timestamp_start": chunk.timestamp_start,
                    "timestamp_seconds": chunk.timestamp_seconds,
                    "content": chunk.content,
                    "word_count": chunk.word_count,
                    "embedding": embedding
                }
                # Upsert: insert or update on conflict with natural key
                self.supabase.table("transcript_chunks").upsert(
                    chunk_data,
                    on_conflict="episode_id,chunk_index"
                ).execute()
    
    def ingest_episode(self, episode: EpisodeData) -> None:
        """Ingest a single episode into Supabase."""
        # 1. Upsert guests
        guest_names = parse_guest_names(episode.guest)
        guest_ids = [self.upsert_guest(name) for name in guest_names]
        
        # 2. Upsert episode
        episode_id = self.upsert_episode(episode)
        
        # 3. Link episode to guests
        self.link_episode_guests(episode_id, guest_ids)
        
        # 4. Upsert chunks with embeddings
        self.upsert_chunks(episode_id, episode.chunks)
    
    def ingest_all(self, limit: int = None) -> None:
        """Ingest transcripts from the episodes directory.
        
        Args:
            limit: Max number of episodes to process (None for all)
        """
        episode_dirs = sorted([d for d in EPISODES_PATH.iterdir() if d.is_dir()])
        
        if limit:
            episode_dirs = episode_dirs[:limit]
            print(f"Processing {limit} of {len(list(EPISODES_PATH.iterdir()))} episode directories")
        else:
            print(f"Processing all {len(episode_dirs)} episode directories")
        
        for episode_dir in tqdm(episode_dirs, desc="Ingesting episodes"):
            transcript_path = episode_dir / "transcript.md"
            
            if not transcript_path.exists():
                print(f"No transcript found in {episode_dir.name}")
                continue
            
            episode = parse_transcript_file(transcript_path)
            if episode:
                try:
                    self.ingest_episode(episode)
                except Exception as e:
                    print(f"Error ingesting {episode_dir.name}: {e}")
                    continue


def main():
    """Main entry point."""
    print("Starting transcript ingestion...")
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Episodes path: {EPISODES_PATH}")
    print(f"Limit: {DEFAULT_LIMIT if DEFAULT_LIMIT else 'None (all episodes)'}")
    
    ingester = TranscriptIngester()
    ingester.ingest_all(limit=DEFAULT_LIMIT)
    
    print("Ingestion complete!")


if __name__ == "__main__":
    main()

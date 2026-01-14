# Lenny's Podcast Knowledge Base

A RAG-powered knowledge base built from 269 Lenny's Podcast transcripts, featuring top operators like Brian Chesky, Marty Cagan, Elena Verna, Shreyas Doshi, and 265+ more.

**Quick video on how to use the MCP**
https://www.tella.tv/video/vladimirs-video-aad7

## Architecture

```
chat-with-lenny/
├── lennys-podcast-transcripts/   # Source transcripts (269 episodes)
│   └── episodes/
│       └── {guest-slug}/
│           └── transcript.md
├── supabase/
│   └── migrations/
│       └── 001_create_schema.sql # Database schema
├── scripts/
│   ├── requirements.txt
│   └── ingest_transcripts.py     # One-off ingestion script
├── mcp_server/
│   ├── requirements.txt
│   ├── server.py                 # MCP server implementation
│   └── mcp.json                  # MCP configuration
└── .env                          # Environment variables
```

## Quick Start

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=your_anon_key
GEMIMI_API_KEY=your_gemini_api_key
```

**Get your keys:**
- **Supabase**: Project Settings → API → Project URL and `anon` `public` key
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Supabase Setup

1. Create a new Supabase project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** → **New Query**
3. Copy and run the entire contents of `supabase/migrations/001_create_schema.sql`
4. This creates 4 tables and enables pgvector for semantic search

### 3. Ingest Transcripts

```bash
cd scripts
pip install -r requirements.txt
python ingest_transcripts.py
```

**What this does:**
- Parses 269 transcript markdown files
- Chunks by speaker turns (~400-600 words)
- Generates Gemini embeddings (768-dim) for each chunk
- Uploads to Supabase

**Time:** ~13,000 embeddings to generate (30-60 minutes depending on API rate limits)

### 4. Test the Server

```bash
cd mcp_server
pip install -r requirements.txt
python server.py
```

The server should start without errors. Press `Ctrl+C` to stop.

### 5. Add to Cursor

Edit `~/.cursor/mcp.json` (or `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json` on Windows):

```json
{
  "mcpServers": {
    "lenny-wisdom": {
      "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
      "args": ["/Users/vladimirdeziegler/chat-with-lenny/mcp_server/server.py"]
    }
  }
}
```

**Note:** 
- Update the Python path (use `which python3` to find it) and project path to match your system
- You can use `python3` instead of the full path if it's in your PATH
- **Restart Cursor** after making changes

### 6. Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "lenny-wisdom": {
      "command": "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
      "args": ["/Users/vladimirdeziegler/chat-with-lenny/mcp_server/server.py"]
    }
  }
}
```

**Note:**
- Update the Python path (use `which python3` to find it) and project path to match your system
- You can use `python3` instead of the full path if it's in your PATH
- **Restart Claude Desktop** after making changes

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_wisdom` | Semantic search across all transcripts |
| `get_advice` | Synthesized C-level advice from multiple experts |
| `compare_experts` | Compare different viewpoints on a topic |
| `generate_playbook` | Create actionable playbooks from expert advice |
| `find_metrics` | Find KPIs and benchmarks mentioned by experts |
| `list_episodes` | Browse and filter episodes |

## Example Queries

```
"How should I structure my product team?"
"What do experts say about product-market fit?"
"Compare Brian Chesky and Marty Cagan on product management"
"Generate a playbook for launching a new product"
"What metrics should I track for B2B SaaS growth?"
```

## Database Schema

```sql
-- 4 tables, ~13,000 chunks, 768-dim Gemini embeddings
guests (id, name, slug)
episodes (id, title, slug, youtube_url, video_id, description, duration_seconds, view_count, transcript_raw)
episode_guests (episode_id, guest_id)
transcript_chunks (id, episode_id, chunk_index, speaker, timestamp_start, content, embedding)
```

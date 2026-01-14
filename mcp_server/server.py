"""
Lenny's Wisdom MCP Server

C-Level Advisory Copilot powered by 269 podcast transcripts.
Provides semantic search, advice synthesis, and expert comparison.
"""

import os
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from supabase import create_client, Client
import google.generativeai as genai

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")
GEMINI_API_KEY = os.getenv("GEMIMI_API_KEY")

EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(LLM_MODEL)

# Initialize MCP server
server = Server("lenny-wisdom")


def get_embedding(text: str) -> list[float]:
    """Generate embedding for text using Gemini."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def search_similar_chunks(query: str, limit: int = 10, min_similarity: float = 0.5) -> list[dict]:
    """Search for similar chunks using vector similarity."""
    query_embedding = get_embedding(query)
    
    # Call the search function we created in Supabase
    result = supabase.rpc(
        "search_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": min_similarity,
            "match_count": limit
        }
    ).execute()
    
    return result.data or []


def synthesize_with_llm(prompt: str, context: str) -> str:
    """Generate a response using Gemini LLM with context."""
    full_prompt = f"""You are a C-level advisor with access to wisdom from 269 podcast episodes 
featuring top operators like Brian Chesky, Marty Cagan, Elena Verna, Shreyas Doshi, and more.

Based on the following expert insights, provide helpful, actionable advice.
Always attribute specific insights to the speaker who said them.

CONTEXT FROM EXPERT INTERVIEWS:
{context}

USER QUESTION:
{prompt}

Provide a thoughtful, well-structured response that synthesizes the expert perspectives.
Include specific quotes and attributions where relevant."""

    response = llm.generate_content(full_prompt)
    return response.text


# Define MCP Tools
TOOLS = [
    Tool(
        name="search_wisdom",
        description="Semantic search across 269 podcast transcripts from top operators (Brian Chesky, Marty Cagan, Elena Verna, etc.). Use for finding expert opinions on specific topics.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question or topic to search for"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="get_advice",
        description="Get synthesized C-level advice on a business challenge from multiple expert perspectives. Best for strategic questions about product, growth, leadership, etc.",
        inputSchema={
            "type": "object",
            "properties": {
                "challenge": {
                    "type": "string",
                    "description": "Business challenge or strategic question"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about your situation (company stage, role, industry)"
                }
            },
            "required": ["challenge"]
        }
    ),
    Tool(
        name="compare_experts",
        description="Compare different expert viewpoints on a topic. Useful for understanding different schools of thought or approaches.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to compare viewpoints on"
                },
                "experts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: specific experts to compare (e.g., ['Brian Chesky', 'Marty Cagan'])"
                }
            },
            "required": ["topic"]
        }
    ),
    Tool(
        name="generate_playbook",
        description="Generate an actionable playbook based on expert advice for a specific goal.",
        inputSchema={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal you want to achieve (e.g., 'launch a new product', 'build a growth team')"
                },
                "constraints": {
                    "type": "string",
                    "description": "Optional constraints (timeline, budget, team size, etc.)"
                }
            },
            "required": ["goal"]
        }
    ),
    Tool(
        name="find_metrics",
        description="Find KPIs, benchmarks, and metrics recommended by experts for a given context.",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category of metrics (growth, retention, engagement, revenue, team)"
                },
                "context": {
                    "type": "string",
                    "description": "Context for the metrics (e.g., 'B2B SaaS Series A', 'consumer app')"
                }
            },
            "required": ["category"]
        }
    ),
    Tool(
        name="list_episodes",
        description="Browse and filter episodes by guest or topic. Returns episode metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "guest": {
                    "type": "string",
                    "description": "Filter by guest name"
                },
                "search": {
                    "type": "string",
                    "description": "Search in episode titles and descriptions"
                },
                "sort": {
                    "type": "string",
                    "enum": ["views", "duration", "recent"],
                    "description": "Sort order (default: views)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum results (default: 10)",
                    "default": 10
                }
            }
        }
    )
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "search_wisdom":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 5)
        
        results = search_similar_chunks(query, limit=limit)
        
        if not results:
            return [TextContent(type="text", text="No relevant results found.")]
        
        # Format results
        formatted = []
        for r in results:
            formatted.append(
                f"**{r['guest_name']}** in *{r['episode_title']}* ({r['timestamp_start']}):\n"
                f"> {r['content'][:500]}{'...' if len(r['content']) > 500 else ''}\n"
                f"(Similarity: {r['similarity']:.2f})"
            )
        
        return [TextContent(type="text", text="\n\n---\n\n".join(formatted))]
    
    elif name == "get_advice":
        challenge = arguments.get("challenge", "")
        context = arguments.get("context", "")
        
        # Search for relevant chunks
        search_query = f"{challenge} {context}".strip()
        results = search_similar_chunks(search_query, limit=8)
        
        if not results:
            return [TextContent(type="text", text="No relevant expert insights found for this challenge.")]
        
        # Build context from results
        expert_context = "\n\n".join([
            f"**{r['guest_name']}** ({r['episode_title']}):\n{r['content']}"
            for r in results
        ])
        
        # Synthesize advice
        advice = synthesize_with_llm(challenge, expert_context)
        
        return [TextContent(type="text", text=advice)]
    
    elif name == "compare_experts":
        topic = arguments.get("topic", "")
        experts = arguments.get("experts", [])
        
        # Search for relevant chunks
        results = search_similar_chunks(topic, limit=15)
        
        if experts:
            # Filter to specific experts
            results = [r for r in results if any(
                expert.lower() in (r.get("guest_name") or "").lower() 
                for expert in experts
            )]
        
        if not results:
            return [TextContent(type="text", text="No relevant expert viewpoints found.")]
        
        # Build context
        expert_context = "\n\n".join([
            f"**{r['guest_name']}** ({r['episode_title']}):\n{r['content']}"
            for r in results
        ])
        
        # Generate comparison
        comparison_prompt = f"Compare the different expert viewpoints on: {topic}"
        comparison = synthesize_with_llm(comparison_prompt, expert_context)
        
        return [TextContent(type="text", text=comparison)]
    
    elif name == "generate_playbook":
        goal = arguments.get("goal", "")
        constraints = arguments.get("constraints", "")
        
        # Search for relevant chunks
        search_query = f"how to {goal} best practices steps"
        results = search_similar_chunks(search_query, limit=10)
        
        if not results:
            return [TextContent(type="text", text="No relevant expert insights found for this goal.")]
        
        # Build context
        expert_context = "\n\n".join([
            f"**{r['guest_name']}** ({r['episode_title']}):\n{r['content']}"
            for r in results
        ])
        
        # Generate playbook
        playbook_prompt = f"""Generate a step-by-step playbook for: {goal}
        
Constraints: {constraints if constraints else 'None specified'}

Structure the playbook with:
1. Key principles from experts
2. Step-by-step actions
3. Common pitfalls to avoid
4. Success metrics to track"""
        
        playbook = synthesize_with_llm(playbook_prompt, expert_context)
        
        return [TextContent(type="text", text=playbook)]
    
    elif name == "find_metrics":
        category = arguments.get("category", "")
        context = arguments.get("context", "")
        
        # Search for metric-related chunks
        search_query = f"{category} metrics KPIs benchmarks {context}"
        results = search_similar_chunks(search_query, limit=8)
        
        if not results:
            return [TextContent(type="text", text="No relevant metrics or benchmarks found.")]
        
        # Build context
        expert_context = "\n\n".join([
            f"**{r['guest_name']}** ({r['episode_title']}):\n{r['content']}"
            for r in results
        ])
        
        # Generate metrics summary
        metrics_prompt = f"""Extract and summarize the key metrics, KPIs, and benchmarks mentioned for:
Category: {category}
Context: {context if context else 'General'}

Include specific numbers and targets where mentioned."""
        
        metrics = synthesize_with_llm(metrics_prompt, expert_context)
        
        return [TextContent(type="text", text=metrics)]
    
    elif name == "list_episodes":
        guest = arguments.get("guest")
        search = arguments.get("search")
        sort = arguments.get("sort", "views")
        limit = arguments.get("limit", 10)
        
        # Build query
        query = supabase.table("episodes").select(
            "title, slug, youtube_url, duration_display, view_count, description"
        )
        
        if search:
            query = query.or_(f"title.ilike.%{search}%,description.ilike.%{search}%")
        
        # Sort
        if sort == "views":
            query = query.order("view_count", desc=True)
        elif sort == "duration":
            query = query.order("duration_seconds", desc=True)
        
        query = query.limit(limit)
        result = query.execute()
        
        if not result.data:
            return [TextContent(type="text", text="No episodes found.")]
        
        # If filtering by guest, we need to join
        if guest:
            guest_result = supabase.table("guests").select("id").ilike("name", f"%{guest}%").execute()
            if guest_result.data:
                guest_ids = [g["id"] for g in guest_result.data]
                episode_guests = supabase.table("episode_guests").select("episode_id").in_("guest_id", guest_ids).execute()
                episode_ids = [eg["episode_id"] for eg in episode_guests.data]
                result.data = [e for e in result.data if e.get("id") in episode_ids]
        
        # Format results
        formatted = []
        for e in result.data:
            formatted.append(
                f"**{e['title']}**\n"
                f"Duration: {e['duration_display']} | Views: {e['view_count']:,}\n"
                f"URL: {e['youtube_url']}"
            )
        
        return [TextContent(type="text", text="\n\n".join(formatted))]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

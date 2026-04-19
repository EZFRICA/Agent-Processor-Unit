"""
APU Tools — Managed by the TEU (Tool Execution Unit).
Contains all external capabilities with built-in caching and monitoring.

Architecture:
  _raw_X_logic() → Pure implementation (sync or async), no caching.
  @tool X()      → Thin wrapper: delegates entirely to teu.execute_tool().
  TEUController  → Handles L1 IO Cache (15 min TTL) and performance monitoring.
"""

from langchain_core.tools import tool
from apu.teu.controller import teu
from config import GEMINI_API_KEY
from logger import get_logger
import asyncio

logger = get_logger(__name__)

# Dedicated search model — more powerful than the main conversational LLM.
# gemini-flash-lite handles conversation; gemini-flash handles real-time retrieval.
_SEARCH_MODEL = "gemini-flash-latest"


def _sync_grounded_search(query: str) -> str:
    """
    Synchronous grounded search using the Google Gen AI SDK.
    Uses Google Search as a retrieval tool for real-time data.
    Wrapped in asyncio.to_thread() by the caller to stay non-blocking.
    """
    from google.genai import Client as GeminiClient
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

    client = GeminiClient(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=_SEARCH_MODEL,
        contents=query,
        config=GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())],
            response_modalities=["TEXT"],
        ),
    )
    return response.text or "No results found."


async def _raw_search_logic(query: str) -> str:
    """
    Async wrapper around the synchronous grounded search.
    Offloads the blocking SDK call to a thread to preserve the event loop.
    This is the implementation that TEU executes and caches.
    """
    logger.info("TEU: Dispatching grounded search via %s — '%s'", _SEARCH_MODEL, query)
    result = await asyncio.to_thread(_sync_grounded_search, query)
    logger.info("TEU: Search completed (%d chars).", len(result))
    return result


@tool
async def google_search(query: str) -> str:
    """Search Google for real-time travel information like flight prices, hotel costs, weather, or local events."""
    return await teu.execute_tool(
        tool_name="google_search",
        tool_func=_raw_search_logic,
        query=query
    )

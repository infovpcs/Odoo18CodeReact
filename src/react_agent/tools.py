"""This module provides tools for web scraping, search functionality, and Odoo code utilities.

It includes:
- A basic Tavily search function (as an example)
- Odoo 18 Code Search/Load Utility tools for searching, loading, and validating Odoo code

These tools support both Main and Critic agents in the LangGraph workflow.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration
from langchain_core.tools import tool
from react_agent.odoo_code_utils import ODOO_CODE_TOOLS


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


TOOLS: List[Callable[..., Any]] = [search] + ODOO_CODE_TOOLS

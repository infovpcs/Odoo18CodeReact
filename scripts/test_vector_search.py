#!/usr/bin/env python
"""
Test script for the Supabase vector search functionality.
This script verifies that the vector search is working correctly with the indexed Odoo 18 codebase.
"""

import asyncio
import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.react_agent.vector_store import SupabaseVectorStore, get_embedding, VectorSearchResult

# Load environment variables
load_dotenv()

async def test_vector_search(query: str, filter_metadata: Dict[str, Any] = None, limit: int = 5):
    """
    Test the vector search functionality with a query.
    
    Args:
        query: The search query
        filter_metadata: Optional metadata filters
        limit: Maximum number of results to return
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    if filter_metadata:
        print(f"FILTERS: {filter_metadata}")
    print(f"{'='*80}")
    
    try:
        # Initialize the vector store
        vector_store = SupabaseVectorStore()
        
        # Generate embedding for the query
        query_embedding = await get_embedding(query)
        
        # Search for similar content
        results = await vector_store.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=0.6,  # Lower threshold for testing
            filter_metadata=filter_metadata
        )
        
        # Display results
        if not results:
            print("No results found.")
            return
        
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results):
            print(f"RESULT {i+1} (Similarity: {result.similarity:.4f}):")
            print(f"File: {result.metadata.get('file_path', 'Unknown')}")
            print(f"Module: {result.metadata.get('module', 'Unknown')}")
            print(f"File Type: {result.metadata.get('file_type', 'Unknown')}")
            
            # Print a snippet of the content (first 200 chars)
            content_preview = result.content[:200].replace('\n', ' ').strip()
            if len(result.content) > 200:
                content_preview += "..."
            print(f"Content: {content_preview}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error during vector search: {e}")

async def run_test_queries():
    """Run a series of test queries to verify the vector search functionality."""
    
    # Test 1: Basic Odoo model definition query
    await test_vector_search("How to define an Odoo model with fields")
    
    # Test 2: Query with module filter
    await test_vector_search(
        "How to create a sale order programmatically",
        filter_metadata={"module": "sale"}
    )
    
    # Test 3: Query for specific file type
    await test_vector_search(
        "How to define XML views in Odoo",
        filter_metadata={"file_type": "view"}
    )
    
    # Test 4: Query for OWL components
    await test_vector_search("How to create an OWL component in Odoo 18")
    
    # Test 5: Query for Odoo API usage
    await test_vector_search("How to use the Odoo ORM API for CRUD operations")
    
    # Test 6: Query for Odoo controllers
    await test_vector_search(
        "How to create a controller in Odoo",
        filter_metadata={"file_type": "controller"}
    )
    
    # Test 7: Query for JavaScript code
    await test_vector_search(
        "How to use JavaScript in Odoo",
        filter_metadata={"file_extension": "js"}
    )

if __name__ == "__main__":
    asyncio.run(run_test_queries())

"""
Vector store module for Odoo React Agent.
Provides vector search capabilities using Supabase.
"""

from typing import List, Dict, Any, Optional
import os
import re
import asyncio
import aiohttp
from pydantic import BaseModel, Field
from supabase import create_client, Client
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration constants
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai, google, or mock
ENABLE_EMBEDDING_FALLBACK = os.getenv("ENABLE_EMBEDDING_FALLBACK", "True").lower() in ("true", "1", "yes")

class VectorSearchResult(BaseModel):
    """Result from a vector search operation."""
    content: str = Field(..., description="The content of the search result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the search result")
    similarity: float = Field(..., description="Similarity score (0-1)")

class SupabaseVectorStore:
    """
    Supabase Vector Store for Odoo code and documentation embeddings.
    """
    
    def __init__(self):
        """Initialize the Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL and key must be set as environment variables "
                "(SUPABASE_URL, SUPABASE_KEY)"
            )
            
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = "odoo_embeddings"
        
    async def store_embedding(
        self, 
        content: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store content with its embedding vector and metadata.
        
        Args:
            content: The text content to store
            embedding: The vector embedding of the content
            metadata: Additional information about the content
            
        Returns:
            Dict: The stored record
        """
        record = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }
        
        response = self.client.table(self.table_name).insert(record).execute()
        return response.data[0] if response.data else {}
        
    async def batch_store_embeddings(
        self,
        contents: List[str],
        metadata_list: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Store multiple content items with their embeddings and metadata in batch.
        
        Args:
            contents: List of text contents to store
            metadata_list: List of metadata dictionaries for each content
            embeddings: Optional pre-computed embeddings (will be generated if not provided)
            
        Returns:
            List[Dict]: The stored records
        """
        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = await batch_get_embeddings(contents)
            
        # Prepare records for batch insert
        records = []
        for i, (content, embedding, metadata) in enumerate(zip(contents, embeddings, metadata_list)):
            records.append({
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            })
            
        # Insert records in batch
        response = self.client.table(self.table_name).insert(records).execute()
        return response.data if response.data else []
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        similarity_threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar content based on embedding similarity.
        
        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filters
            
        Returns:
            List[VectorSearchResult]: Matching results with similarity scores
        """
        # Build the query
        query = self.client.rpc(
            "match_embeddings",
            {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": limit
            }
        )
        
        # Apply metadata filters if provided
        if filter_metadata:
            for key, value in filter_metadata.items():
                query = query.eq(f"metadata->{key}", value)
        
        # Execute the query
        response = query.execute()
        
        # Parse results
        results = []
        for item in response.data:
            results.append(
                VectorSearchResult(
                    content=item["content"],
                    metadata=item["metadata"],
                    similarity=item["similarity"]
                )
            )
            
        return results

async def _mock_embedding(text: str, dimension: int = 1536) -> List[float]:
    """
    Generate a deterministic mock embedding for testing purposes.
    This is used when OpenAI API key is not available.
    
    Args:
        text: The text to generate a mock embedding for
        dimension: The dimension of the embedding vector
        
    Returns:
        List[float]: A deterministic mock embedding vector
    """
    import hashlib
    import random
    
    # Create a deterministic seed from the text
    hash_obj = hashlib.md5(text.encode())
    seed = int(hash_obj.hexdigest(), 16) % (2**32)
    random.seed(seed)
    
    # Generate a deterministic embedding
    embedding = [random.uniform(-1, 1) for _ in range(dimension)]
    
    # Normalize the embedding
    norm = sum(x**2 for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def get_embedding(text: str, model: str = "models/embedding-001", attempt_count: int = 0) -> List[float]:
    """
    Generate an embedding for the given text using Google's embedding model.
    Implements sophisticated retry mechanism with exponential backoff for rate limits.
    Falls back to a mock embedding if Google API key is not available or after max retries.
    
    Args:
        text: The text to generate an embedding for
        model: The Google embedding model to use
        attempt_count: Current retry attempt (used internally for recursion)
        
    Returns:
        List[float]: The embedding vector
    """
    # Handle empty text
    if not text or not text.strip():
        return await _mock_embedding("", dimension=768)
        
    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.strip() == "":
        print("Warning: Google API key not found. Using mock embedding.")
        return await _mock_embedding(text, dimension=768)
    
    # Define max retries and retry conditions
    max_retries = 4  # Maximum number of retry attempts
    
    try:
        # Import the Google Generative AI library
        import google.generativeai as genai
        
        # Configure the client
        genai.configure(api_key=google_api_key)
        
        # Generate the embedding
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        
        return result["embedding"]
    except Exception as e:
        error_str = str(e).lower()
        
        # Check if this is a rate limit or quota error
        is_rate_limit_error = (
            "429" in error_str or 
            "quota" in error_str or 
            "rate limit" in error_str or
            "resourceexhausted" in error_str or
            "too many requests" in error_str
        )
        
        # Implement retry logic with exponential backoff for rate limit errors
        if is_rate_limit_error and attempt_count < max_retries:
            # Calculate retry delay with jitter to avoid thundering herd
            import random
            base_delay = min(30, (2 ** attempt_count) * 2)  # Cap at 30 seconds
            jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
            retry_delay = base_delay * jitter
            
            # Different messaging based on error type
            if "quota" in error_str:
                print(f"Google Embedding API quota exceeded, waiting {retry_delay:.1f} seconds before retry {attempt_count + 1}/{max_retries}")
            else:
                print(f"Google Embedding API rate limit hit, waiting {retry_delay:.1f} seconds before retry {attempt_count + 1}/{max_retries}")
            
            # Sleep and retry
            await asyncio.sleep(retry_delay)
            return await get_embedding(text, model, attempt_count + 1)
        
        # If we've exhausted retries or it's not a rate limit error
        print(f"Error getting embedding from Google after {attempt_count} retries: {e}")
        if ENABLE_EMBEDDING_FALLBACK:
            print("Falling back to mock embedding.")
            return await _mock_embedding(text, dimension=768)
        else:
            raise

async def batch_get_embeddings(
    texts: List[str], 
    model: str = "models/embedding-001"
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a batch.
    Falls back to mock embeddings if Google API key is not available.
    
    Args:
        texts: List of texts to generate embeddings for
        model: The Google embedding model to use
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    # Filter out empty texts
    valid_texts = [text for text in texts if text.strip()]
    
    if not valid_texts:
        return []
    
    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.strip() == "":
        print("Warning: Google API key not found. Using mock embeddings for batch.")
        return [await _mock_embedding(text, dimension=768) for text in valid_texts]
    
    try:
        # Import the Google Generative AI library
        import google.generativeai as genai
        
        # Configure the client
        genai.configure(api_key=google_api_key)
        
        # Generate embeddings one by one (Google doesn't support batch embedding in a single API call)
        embeddings = []
        for text in valid_texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        
        return embeddings
    except Exception as e:
        print(f"Error getting batch embeddings from Google: {e}")
        print("Falling back to mock embeddings for batch.")
        return [await _mock_embedding(text, dimension=768) for text in valid_texts]

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split a large text into smaller chunks for processing.
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # If text is smaller than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk_size,
        # save the current chunk and start a new one
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:  # Don't add empty chunks
                chunks.append(current_chunk)
            
            # If the paragraph itself is longer than chunk_size,
            # split it into smaller pieces
            if len(paragraph) > chunk_size:
                words = paragraph.split()
                current_chunk = ""
                
                for word in words:
                    if len(current_chunk) + len(word) + 1 > chunk_size:
                        chunks.append(current_chunk)
                        # Start new chunk with overlap
                        overlap_point = max(0, len(current_chunk) - overlap)
                        current_chunk = current_chunk[overlap_point:] + " " + word
                    else:
                        if current_chunk:
                            current_chunk += " "
                        current_chunk += word
            else:
                # Start new chunk with the current paragraph
                # Include overlap from previous chunk if possible
                if chunks and overlap > 0:
                    last_chunk = chunks[-1]
                    overlap_text = last_chunk[-overlap:] if len(last_chunk) > overlap else last_chunk
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


async def batch_get_embeddings(
    texts: List[str], 
    model: str = "models/embedding-001"
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a batch.
    Falls back to mock embeddings if Google API key is not available.
    
    Args:
        texts: List of texts to generate embeddings for
        model: The Google embedding model to use
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    # Filter out empty texts
    valid_texts = [text for text in texts if text.strip()]
    
    if not valid_texts:
        return []
    
    # Check if Google API key is available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key.strip() == "":
        print("Warning: Google API key not found. Using mock embeddings for batch.")
        return [await _mock_embedding(text, dimension=768) for text in valid_texts]
    
    try:
        # Import the Google Generative AI library
        import google.generativeai as genai
        
        # Configure the client
        genai.configure(api_key=google_api_key)
        
        # Generate embeddings one by one (Google doesn't support batch embedding in a single API call)
        embeddings = []
        for text in valid_texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        
        return embeddings
    except Exception as e:
        print(f"Error getting batch embeddings from Google: {e}")
        print("Falling back to mock embeddings for batch.")
        return [await _mock_embedding(text, dimension=768) for text in valid_texts]

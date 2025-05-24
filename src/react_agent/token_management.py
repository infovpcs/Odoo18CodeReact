"""Token management utilities for LLM API calls."""

import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Google Gemini models token limits (these may change, so keep updated)
MODEL_TOKEN_LIMITS = {
    "gemini-1.0-pro": {"input": 30720, "output": 8192, "total": 30720},
    "gemini-1.5-pro": {"input": 1000000, "output": 8192, "total": 1000000},
    "gemini-1.5-flash": {"input": 1000000, "output": 8192, "total": 1000000},
    "gemini-2.0-pro": {"input": 1000000, "output": 8192, "total": 1000000},
    "gemini-2.0-flash": {"input": 1000000, "output": 8192, "total": 1000000},
    # Add other models as needed with their limits
    "default": {"input": 30000, "output": 2048, "total": 30000}
}

def get_model_token_limits(model_name: str) -> Dict[str, int]:
    """Get token limits for a specific model.
    
    Args:
        model_name: The name of the model
        
    Returns:
        Dict with input, output and total token limits
    """
    # Clean up model name to match our dictionary keys
    cleaned_name = model_name.lower()
    if ":" in cleaned_name:
        # Extract just the model part, e.g., "google:gemini-1.0-pro" -> "gemini-1.0-pro"
        cleaned_name = cleaned_name.split(":", 1)[1]
    
    # Return the limits for the specific model, or default if not found
    for model_key in MODEL_TOKEN_LIMITS:
        if model_key in cleaned_name:
            return MODEL_TOKEN_LIMITS[model_key]
    
    # If we can't find a specific match, return default limits
    return MODEL_TOKEN_LIMITS["default"]

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.
    
    This is a simple approximation - for production, consider using the 
    tokenizer specific to your model.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Simple heuristic: about 4 characters per token for English text
    # This is a rough estimate and should be replaced with proper tokenization
    # for production use
    return len(text) // 4 + 1

def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate tokens in a BaseMessage object.
    
    Args:
        message: The message to estimate tokens for
        
    Returns:
        Estimated token count
    """
    content = message.content
    
    # Handle different content types
    if isinstance(content, str):
        return estimate_tokens(content)
    elif isinstance(content, dict):
        # For structured content like tool calls
        return estimate_tokens(str(content))
    elif isinstance(content, list):
        # For multi-modal content
        total = 0
        for item in content:
            if isinstance(item, str):
                total += estimate_tokens(item)
            elif isinstance(item, dict):
                total += estimate_tokens(str(item))
        return total
    
    # Fallback
    return estimate_tokens(str(content))

def estimate_messages_tokens(messages: List[BaseMessage]) -> int:
    """Estimate total tokens in a list of messages.
    
    Args:
        messages: List of messages to estimate tokens for
        
    Returns:
        Total estimated token count
    """
    total = 0
    for message in messages:
        total += estimate_message_tokens(message)
    
    # Add overhead for message formatting (roles, etc.)
    overhead = len(messages) * 4
    return total + overhead

def should_split_messages(messages: List[BaseMessage], model_name: str) -> bool:
    """Check if messages should be split based on token limits.
    
    Args:
        messages: List of messages to check
        model_name: Name of the model to use
        
    Returns:
        True if messages should be split, False otherwise
    """
    limits = get_model_token_limits(model_name)
    estimated_tokens = estimate_messages_tokens(messages)
    
    return estimated_tokens > limits["input"] * 0.9  # Use 90% as safety margin

def split_messages(messages: List[BaseMessage], model_name: str) -> List[List[BaseMessage]]:
    """Split a list of messages into chunks that fit within token limits.
    
    Args:
        messages: List of messages to split
        model_name: Name of the model to use
        
    Returns:
        List of message chunks that each fit within token limits
    """
    limits = get_model_token_limits(model_name)
    max_tokens = int(limits["input"] * 0.9)  # 90% safety margin
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    # Always include system message in each chunk if present
    system_message = None
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            system_message = message
            system_tokens = estimate_message_tokens(message)
            break
    
    for message in messages:
        # Skip system message as we handle it separately
        if isinstance(message, SystemMessage) and message == system_message:
            continue
            
        message_tokens = estimate_message_tokens(message)
        
        # If adding this message would exceed the limit, start a new chunk
        if current_tokens + message_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
            
            # Always include system message in each new chunk
            if system_message:
                current_chunk.append(system_message)
                current_tokens += system_tokens
        
        # Add message to current chunk
        current_chunk.append(message)
        current_tokens += message_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

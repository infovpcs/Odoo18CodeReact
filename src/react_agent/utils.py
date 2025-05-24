"""Utility & helper functions for Odoo React Agent."""

from typing import Dict, Optional, Any, Tuple
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message.
    
    Args:
        msg (BaseMessage): The message to extract text from.
        
    Returns:
        str: The extracted text content.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


# Global model status tracker for each agent component
MODEL_STATUS = {
    "vector_search": {"provider": "", "model": "", "status": "idle", "last_error": None},
    "call_model": {"provider": "", "model": "", "status": "idle", "last_error": None},
    "critic": {"provider": "", "model": "", "status": "idle", "last_error": None},
    "tools": {"provider": "", "model": "", "status": "idle", "last_error": None},
    "manager": {"provider": "", "model": "", "status": "idle", "last_error": None}
}


def update_model_status(agent_name: str, provider: str, model: str, status: str, error: Optional[Exception] = None) -> None:
    """Update the status of a model for a specific agent component.
    
    Args:
        agent_name (str): The name of the agent component.
        provider (str): The provider of the model.
        model (str): The name of the model.
        status (str): The current status of the model (idle, loading, running, error, complete).
        error (Optional[Exception]): Any error that occurred during model execution.
    """
    if agent_name in MODEL_STATUS:
        MODEL_STATUS[agent_name] = {
            "provider": provider,
            "model": model,
            "status": status,
            "last_error": str(error) if error else None,
            "timestamp": __import__('time').time()
        }


def get_model_status(agent_name: str = None) -> Dict[str, Any]:
    """Get the current status of models for all agent components or a specific one.
    
    Args:
        agent_name (str, optional): The name of a specific agent component.
        
    Returns:
        Dict: The model status information.
    """
    if agent_name and agent_name in MODEL_STATUS:
        return MODEL_STATUS[agent_name]
    return MODEL_STATUS


def bind_tools_safely(model: BaseChatModel, tools: list, model_type: str = "unknown") -> BaseChatModel:
    """Safely bind tools to a model with proper error handling for models that don't support tool binding.
    
    Args:
        model (BaseChatModel): The chat model to bind tools to.
        tools (list): List of tools to bind to the model.
        model_type (str): Type of model (e.g., "google", "ollama", etc.) for logging purposes.
        
    Returns:
        BaseChatModel: The model with tools bound if supported, or the original model if not.
    """
    # Skip binding for Ollama models which don't support direct tool binding
    if model_type.lower() == "ollama":
        print(f"Note: Ollama models don't support LangChain's direct tool binding interface. "  
              f"Using base model without tool binding capabilities.")
        return model
    
    # For all other models, attempt to bind tools with proper error handling
    try:
        # Attempt to bind tools to the model
        model_with_tools = model.bind_tools(tools)
        return model_with_tools
    except NotImplementedError as e:
        # Handle the case where the model doesn't support tool binding
        print(f"Warning: Model doesn't support LangChain's tool binding interface: {e}")
        print(f"Using base model without tool binding capabilities.")
        return model
    except Exception as e:
        # Log other errors but still return the base model
        print(f"Error binding tools to model: {e}")
        print(f"Using base model without tool binding capabilities.")
        return model


def load_chat_model(fully_specified_name: str, attempt_count: int = 0, agent_name: str = "call_model") -> BaseChatModel:
    """Load a chat model from a fully specified name with improved error handling and status tracking.

    Args:
        fully_specified_name (str): String in the format 'provider/model' or 'provider:model'.
        attempt_count (int): Number of attempts made to load the model (for tracking retries).
        agent_name (str): The name of the agent component using this model.
        
    Returns:
        BaseChatModel: The initialized chat model.
        
    Raises:
        ValueError: If the model name format is invalid or if all fallback attempts fail.
    """
    from react_agent.configuration import Configuration
    import time
    
    # Get configuration for fallback options
    config = Configuration.from_context()
    max_retries = 3
    
    # Parse provider and model name
    provider = ""
    model = ""
    try:
        if ":" in fully_specified_name:
            provider, model = fully_specified_name.split(":", maxsplit=1)
        elif "/" in fully_specified_name:
            provider, model = fully_specified_name.split("/", maxsplit=1)
        else:
            error_msg = f"Model name must be in 'provider/model' or 'provider:model' format. Received: {fully_specified_name}"
            update_model_status(agent_name, "", "", "error", ValueError(error_msg))
            raise ValueError(error_msg)
        
        # Check if this is an Ollama model (which doesn't support tool binding)
        is_ollama_model = provider.lower() == "ollama"
        
        # Update status to loading
        update_model_status(agent_name, provider, model, "loading")
        
        # Check if this is a Google model which might have rate limits
        is_google_model = provider.lower() in ["google", "google_genai", "googlegenai", "gemini"]
        
        # Initialize the model with special handling for Google models
        try:
            if is_google_model:
                # Use specialized initialization for Google Gemini models
                # with better tool binding support
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    # Set lower temperature for more deterministic tool calling
                    # This follows Google's recommendation for better tool binding
                    model_kwargs = {
                        "temperature": 0.0,  # Lower temperature for better tool calling reliability
                        "convert_system_message_to_human": True,  # Important for Gemini to interpret system messages
                        "top_p": 0.95,  # More focused token selection
                        "top_k": 64,     # More focused token selection
                    }
                    
                    # Create the model
                    chat_model = ChatGoogleGenerativeAI(model=model, **model_kwargs)
                    
                    # Explicit warning about Gemini 2.0-flash
                    if "gemini-2.0-flash" in model.lower():
                        print("Note: Using gemini-2.0-flash, which may have limitations with tool calling. "  
                              "Consider using gemini-2.5-flash instead if tool binding is unreliable.")
                except ImportError:
                    # Fall back to standard initialization if langchain_google_genai isn't available
                    print("Using standard LangChain initialization for Google model. For better tool binding, "
                          "install langchain-google-genai>=0.1.5")
                    chat_model = init_chat_model(model, model_provider=provider)
            else:
                # Standard initialization for non-Google models
                chat_model = init_chat_model(model, model_provider=provider)
                
            # Update status to ready
            update_model_status(agent_name, provider, model, "ready")
            return chat_model
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors specifically with Google models
            is_rate_limit_error = (
                "429" in error_str or 
                "quota" in error_str or 
                "rate limit" in error_str or
                "resourceexhausted" in error_str or
                "too many requests" in error_str
            )
            
            if is_google_model and is_rate_limit_error:
                # More sophisticated retry strategy for Google models
                max_google_retries = 5  # Increase max retries for Google models
                
                if attempt_count < max_google_retries:
                    # Calculate retry delay with jitter to avoid thundering herd
                    import random
                    base_delay = min(30, (2 ** attempt_count) * 2)  # Cap at 30 seconds
                    jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
                    retry_delay = base_delay * jitter
                    
                    # Update status to retrying
                    update_model_status(agent_name, provider, model, "retrying", e)
                    
                    # Different messaging based on error type
                    if "quota" in error_str:
                        print(f"Google API quota exceeded, waiting {retry_delay:.1f} seconds before retry {attempt_count + 1}/{max_google_retries}")
                    else:
                        print(f"Google API rate limit hit, waiting {retry_delay:.1f} seconds before retry {attempt_count + 1}/{max_google_retries}")
                    
                    # Sleep and retry
                    time.sleep(retry_delay)
                    return load_chat_model(fully_specified_name, attempt_count + 1, agent_name)
            
            # If we've exhausted retries or it's not a rate limit error, check if we should use fallback
            update_model_status(agent_name, provider, model, "error", e)
            
            if config.enable_fallback and hasattr(config, 'fallback_model'):
                print(f"Failed to load {fully_specified_name}, falling back to {config.fallback_model}")
                # Update status to fallback
                update_model_status(agent_name, provider, model, "fallback", e)
                # Recursively call with fallback model, but reset attempt count to avoid infinite recursion
                return load_chat_model(config.fallback_model, 0, agent_name)
            
            # If no fallback is configured or fallback is disabled, raise the original error
            raise
    except ValueError as format_error:
        # Only handle format errors here, other exceptions are handled above
        update_model_status(agent_name, provider, model, "error", format_error)
        raise format_error

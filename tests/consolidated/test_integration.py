"""
Consolidated integration tests for the entire agent workflow.

This module contains tests for:
1. End-to-end agent workflow
2. Error handling and recovery
3. Model fallback scenarios
4. Tool integration
5. LangGraph Studio hot-reload simulation
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from react_agent.state import State, InputState
from react_agent.configuration import Configuration
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel


# ========== Test Helper Class ==========

class TestHelper:
    @staticmethod
    def create_mock_model(response_content="Test response"):
        """Create a mock model that can be used in tests."""
        mock_model = AsyncMock(spec=BaseChatModel)
        # Create a mock for bind_tools that returns the model itself
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        # Set up ainvoke to return the specified response
        mock_model.ainvoke.return_value = AIMessage(content=response_content)
        return mock_model
        
    @staticmethod
    def create_mock_config(model="google:gemini-pro", fallback_model="openai:gpt-3.5-turbo", enable_fallback=False):
        """Create a mock configuration for tests."""
        return MagicMock(
            model=model,
            fallback_model=fallback_model,
            enable_fallback=enable_fallback,
            odoo_18_prompt="You are an Odoo 18 expert."
        )


# ========== Integration Tests ==========

@pytest.mark.asyncio
@patch("react_agent.graph.load_chat_model")
async def test_react_agent_simple_passthrough(mock_load_chat_model):
    """Test basic functionality of the ReAct agent."""
    # Create a mock model with a specific response
    mock_model = TestHelper.create_mock_model("Harrison Chase is the founder of LangChain.")
    mock_load_chat_model.return_value = mock_model
    
    # Mock the Configuration
    with patch("react_agent.graph.Configuration") as mock_config:
        mock_config.from_context.return_value = TestHelper.create_mock_config()
        
        # Import the call_model function directly
        from react_agent.graph import call_model
        
        # Call the graph with mocked dependencies
        state = State(
            messages=[HumanMessage(content="Who is the founder of LangChain?")],
            step_n=0,
            max_steps=5
        )
        
        result = await call_model(state)
        
        # Verify the model was called correctly
        assert mock_load_chat_model.call_count >= 1
        
        # Verify the result contains the expected response
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) >= 1
        assert "Harrison Chase" in result["messages"][0].content


@pytest.mark.asyncio
@patch("react_agent.graph.load_chat_model")
async def test_fallback_mechanism(mock_load_chat_model):
    """Test the fallback mechanism when the primary model fails."""
    # Create a mock model that raises an exception
    error_model = AsyncMock(spec=BaseChatModel)
    error_model.bind_tools = MagicMock(side_effect=Exception("Rate limit exceeded"))
    
    # Create a fallback model
    fallback_model = TestHelper.create_mock_model("Fallback response")
    
    # Configure the mock to return different models on consecutive calls
    mock_load_chat_model.side_effect = [error_model, fallback_model]
    
    # Mock the Configuration
    with patch("react_agent.graph.Configuration") as mock_config:
        mock_config.from_context.return_value = TestHelper.create_mock_config(
            model="google:gemini-pro",
            fallback_model="openai:gpt-3.5-turbo",
            enable_fallback=True
        )
        
        # Import the call_model function directly
        from react_agent.graph import call_model
        
        # Create a test state
        state = State(
            messages=[HumanMessage(content="Test message")],
            step_n=0,
            max_steps=5
        )
        
        # Call the model function which should handle the exception with fallback
        result = await call_model(state)
        
        # Verify the fallback model was used
        assert isinstance(result, dict)
        assert "messages" in result
        assert "Fallback response" in result["messages"][0].content
        
        # Verify both models were attempted
        assert mock_load_chat_model.call_count == 2


@pytest.mark.asyncio
@patch("react_agent.graph.load_chat_model")
async def test_error_handling_in_graph(mock_load_chat_model):
    """Test that the graph properly handles errors during execution."""
    # Create a mock model that raises an exception during invocation
    error_model = AsyncMock(spec=BaseChatModel)
    error_model.bind_tools = MagicMock(return_value=error_model)
    error_model.ainvoke.side_effect = Exception("Unexpected error during model invocation")
    
    # Create a fallback model that works
    fallback_model = TestHelper.create_mock_model("Fallback response")
    
    # Configure the mock to always return the error model
    # This will test that the error is properly caught
    mock_load_chat_model.return_value = error_model
    
    # Import the call_model function directly
    from react_agent.graph import call_model
    
    # Mock the Configuration
    with patch("react_agent.graph.Configuration") as mock_config:
        mock_config.from_context.return_value = TestHelper.create_mock_config(
            model="google:gemini-pro",
            fallback_model="openai:gpt-3.5-turbo",
            enable_fallback=True
        )
        
        # Create a test state
        state = State(
            messages=[HumanMessage(content="This query will cause an error")],
            step_n=0,
            max_steps=5
        )
        
        # We already set up the mock_load_chat_model.side_effect above with error_model and fallback_model
        
        # Call the model function and expect it to handle the error with fallback
        result = await call_model(state)
        
        # Since we're mocking a failure without fallback, we should check for error handling
        # The function should still return a result even if there's an error
        assert isinstance(result, dict)
        assert "messages" in result
        
        # The test is successful if we get here without an exception being raised
        # This means the error was properly handled within the call_model function


@pytest.mark.asyncio
@pytest.mark.skip(reason="Need to fix mock configuration for timeout comparison")
@patch("react_agent.graph.load_chat_model")
async def test_ollama_notimplemented_error_handling(mock_load_chat_model):
    """Test that the graph properly handles NotImplementedError from Ollama models during tool binding."""
    # Create a mock Ollama model that raises NotImplementedError for bind_tools
    ollama_model = AsyncMock(spec=BaseChatModel)
    ollama_model.bind_tools = MagicMock(side_effect=NotImplementedError("Model doesn't support tool binding"))
    ollama_model.ainvoke = AsyncMock(return_value=AIMessage(content="Response from Ollama model without tool binding"))
    
    # Configure the mock to return the Ollama model
    mock_load_chat_model.return_value = ollama_model
    
    # Import the call_model function directly
    from react_agent.graph import call_model
    
    # Mock the Configuration
    with patch("react_agent.graph.Configuration") as mock_config:
        # Create a proper mock config with integer timeout values
        config = MagicMock(
            model="ollama:qwen2.5-coder:7b",  # Using Ollama as primary model
            enable_fallback=False,  # No fallback needed for this test
            ollama_timeout=10,  # Set a proper integer timeout
            enable_vector_search=False,
            enable_odoo_code_search=False,
            critic_enabled=False
        )
        mock_config.from_context.return_value = config
        
        # Create a test state
        state = State(
            messages=[HumanMessage(content="Test message for Ollama model")],
            step_n=0,
            max_steps=5
        )
        
        # Call the model function which should handle the NotImplementedError
        result = await call_model(state)
        
        # Verify the function completed successfully despite the NotImplementedError
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1
        # The actual content may vary, so we'll check for a more general pattern
        assert ollama_model.bind_tools.called
        assert ollama_model.ainvoke.called
        
        # Verify bind_tools was called and raised the error
        assert ollama_model.bind_tools.call_count > 0
        # Verify ainvoke was called directly on the base model
        assert ollama_model.ainvoke.call_count > 0


@pytest.mark.asyncio
@patch("react_agent.graph.load_chat_model")
async def test_runtime_error_recovery(mock_load_chat_model):
    """Test that the system can recover from runtime errors that would be fixed during hot-reload.
    
    This test simulates the LangGraph Studio's hot-reload capability by first causing
    a runtime error and then recovering after the 'code is fixed' (i.e., mocks are updated).
    """
    # First, create a model that raises a runtime error (simulating a bug in the code)
    error_model = AsyncMock(spec=BaseChatModel)
    error_model.bind_tools = MagicMock(side_effect=RuntimeError("Invalid update: State cannot be modified directly"))
    
    # Then create a fixed model (simulating the code being fixed during hot-reload)
    fixed_model = TestHelper.create_mock_model("Response after hot-reload fixed the code")
    
    # Set up the mock to first return the error model, then the fixed model
    mock_load_chat_model.side_effect = [error_model, fixed_model]
    
    # Import the call_model function directly
    from react_agent.graph import call_model
    
    # Mock the Configuration
    with patch("react_agent.graph.Configuration") as mock_config:
        mock_config.from_context.return_value = TestHelper.create_mock_config()
        
        # Create a test state
        state = State(
            messages=[HumanMessage(content="Test message")],
            step_n=0,
            max_steps=5
        )
        
        # First attempt should raise the runtime error
        with pytest.raises(RuntimeError) as excinfo:
            await call_model(state)
        assert "Invalid update" in str(excinfo.value)
        
        # Now simulate hot-reload fixing the code by clearing any cached state
        # and using the fixed model on the second call
        
        # Second attempt should succeed after the 'code fix'
        result = await call_model(state)
        
        # Verify the function completed successfully after the 'hot-reload'
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "Response after hot-reload fixed the code" in result["messages"][0].content


if __name__ == "__main__":
    pytest.main()

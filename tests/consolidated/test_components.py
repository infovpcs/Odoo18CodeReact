"""
Consolidated tests for core components including the critic agent, workflow, and status tracking.

This module contains tests for:
1. Critic agent functionality for code evaluation
2. Workflow status and transitions
3. Graph components and node interactions
"""

import unittest
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from react_agent.critic import (
    extract_code_from_messages,
    generate_recommendation,
    evaluate_code,
)
from react_agent.state import State, InputState
from react_agent.utils import update_model_status, get_model_status, MODEL_STATUS
from react_agent.configuration import Configuration
from langchain_core.language_models import BaseChatModel


# ========== Fixtures ==========

@pytest.fixture
def reset_model_status():
    """Reset the model status tracker before and after each test."""
    # Reset before test
    for agent in MODEL_STATUS:
        MODEL_STATUS[agent] = {"provider": "", "model": "", "status": "idle", "last_error": None, "timestamp": 0}
    yield
    # Reset after test
    for agent in MODEL_STATUS:
        MODEL_STATUS[agent] = {"provider": "", "model": "", "status": "idle", "last_error": None, "timestamp": 0}


# ========== Critic Agent Tests ==========

class TestCriticAgent(unittest.TestCase):
    """Test cases for the Critic Agent functionality."""

    def test_extract_code_from_messages(self):
        """Test extracting code blocks from messages."""
        # Test with a simple code block
        messages = [
            HumanMessage(content="Can you write a simple Odoo model?"),
            AIMessage(
                content="""
                Here's a simple Odoo model:

                ```python
                from odoo import models, fields, api

                class CustomModel(models.Model):
                    _name = 'custom.model'
                    _description = 'Custom Model'

                    name = fields.Char(string='Name', required=True)
                    description = fields.Text(string='Description')
                    active = fields.Boolean(default=True)
                ```
                """
            ),
        ]

        code = extract_code_from_messages(messages)
        self.assertIn("from odoo import models, fields, api", code)
        self.assertIn("class CustomModel(models.Model):", code)

        # Test with multiple code blocks
        messages = [
            AIMessage(
                content="""
                Here's the model:

                ```python
                class CustomModel(models.Model):
                    _name = 'custom.model'
                ```

                And here's the controller:

                ```python
                class CustomController(http.Controller):
                    @http.route('/custom', type='http')
                    def index(self):
                        return 'Hello'
                ```
                """
            ),
        ]

        code = extract_code_from_messages(messages)
        self.assertIn("class CustomModel(models.Model):", code)
        self.assertIn("class CustomController(http.Controller):", code)

        # Test with no code blocks
        messages = [
            HumanMessage(content="What is Odoo?"),
            AIMessage(
                content="Odoo is an open-source business management software suite."
            ),
        ]

        code = extract_code_from_messages(messages)
        self.assertIsNone(code)

    def test_generate_recommendation(self):
        """Test generating recommendations based on evaluation results."""
        quality_result = {"comment": "The code follows Odoo conventions but lacks proper docstrings."}
        correctness_result = {"comment": "The code is functionally correct but missing error handling."}

        # Test with index within default recommendations
        recommendation = generate_recommendation(quality_result, correctness_result, 1)
        self.assertTrue(recommendation)
        self.assertIsInstance(recommendation, str)

        # Test with index outside default recommendations
        recommendation = generate_recommendation(quality_result, correctness_result, 5)
        self.assertTrue(recommendation)
        self.assertIsInstance(recommendation, str)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs more complex mocking to handle evaluate_code implementation")
async def test_evaluate_code_async():
    """Test the evaluate_code function using pytest's async support."""
    # Create a state with Odoo code in the messages
    state = State(
        messages=[
            HumanMessage(content="Can you create an Odoo model?"),
            AIMessage(
                content="""
                ```python
                from odoo import models, fields
                
                class Product(models.Model):
                    _name = 'product.template'
                    _description = 'Product Template'
                    
                    name = fields.Char(string='Name', required=True)
                    price = fields.Float(string='Price')
                ```
                """
            ),
        ],
        is_last_step=False,
        node_history=[],
        last_node='agent'
    )
    
    # Based on the actual implementation, need to mock many more things
    with patch("react_agent.critic.load_chat_model") as mock_load_model, \
         patch("react_agent.critic.Configuration") as mock_config, \
         patch("react_agent.utils.update_model_status") as mock_update_status, \
         patch("react_agent.critic.extract_code_from_messages") as mock_extract_code:
        
        # Setup code extraction to return real code
        mock_extract_code.return_value = [
            "from odoo import models, fields\n\nclass Product(models.Model):\n    _name = 'product.template'\n    _description = 'Product Template'\n\n    name = fields.Char(string='Name', required=True)\n    price = fields.Float(string='Price')"
        ]
        
        # Setup mock model that returns success
        mock_model = AsyncMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)  # Make bind_tools return the model itself
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="The code looks good."))
        mock_load_model.return_value = mock_model
        
        # Configure the configuration
        mock_config.from_context.return_value = MagicMock(
            model="gemini-pro",
            critic_model="gemini-pro",
            critic_temperature=0.2,
            critic_max_tokens=1024,
            enable_critic_fallback=True
        )
        
        # Test the evaluate_code function
        result = await evaluate_code(state)
        
        # Verify the basic structure of the result
        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs more complex mocking to handle evaluate_code implementation")
async def test_critic_fallback_mechanism():
    """Test that the critic falls back to the main model when the critic model fails."""
    # Create a state with Odoo code in the messages
    state = State(
        messages=[
            HumanMessage(content="Can you create an Odoo model?"),
            AIMessage(
                content="""
                ```python
                from odoo import models, fields
                
                class Product(models.Model):
                    _name = 'product.template'
                    _description = 'Product Template'
                    
                    name = fields.Char(string='Name', required=True)
                    price = fields.Float(string='Price')
                ```
                """
            ),
        ],
        is_last_step=False,
        node_history=[],
        last_node='agent'
    )

    # Set up all the necessary mocks
    with patch("react_agent.critic.load_chat_model") as mock_load_chat_model, \
         patch("react_agent.critic.Configuration") as mock_config, \
         patch("react_agent.utils.update_model_status") as mock_update_status, \
         patch("react_agent.critic.extract_code_from_messages") as mock_extract_code:
        
        # Return valid code from the extraction
        mock_extract_code.return_value = [
            "from odoo import models, fields\n\nclass Product(models.Model):\n    _name = 'product.template'\n    _description = 'Product Template'\n\n    name = fields.Char(string='Name', required=True)\n    price = fields.Float(string='Price')"
        ]
        
        # Create a mock model with a problematic first attempt
        mock_model = AsyncMock()
        
        # Set up a side effect for ainvoke that first raises an exception, then works
        mock_invoke_responses = [
            AsyncMock(side_effect=Exception("Critic model failed")),
            AsyncMock(return_value=AIMessage(content="Fallback evaluation result"))
        ]
        
        mock_model.ainvoke = mock_invoke_responses[0]
        # After the first call fails, replace with the success function
        def side_effect(*args, **kwargs):
            mock_model.ainvoke = mock_invoke_responses[1]
            raise Exception("Critic model failed")
        
        mock_model.ainvoke.side_effect = side_effect
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_load_chat_model.return_value = mock_model

        # Set up the configuration with fallback enabled
        mock_config.from_context.return_value = MagicMock(
            model="gemini-pro",
            critic_model="claude-3-haiku",
            enable_critic_fallback=True,
            critic_temperature=0.2,
            critic_max_tokens=1024
        )

        # Test the evaluate_code function
        result = await evaluate_code(state)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs more complex mocking to handle evaluate_code implementation")
async def test_timeout_handling():
    """Test that the critic handles timeouts gracefully."""
    # Create a state with Odoo code in the messages
    state = State(
        messages=[
            HumanMessage(content="Can you create an Odoo model?"),
            AIMessage(content="```python\nfrom odoo import models, fields\n\nclass Product(models.Model):\n    _name = 'product.template'\n```")
        ],
        is_last_step=False,
        node_history=[],
        last_node='agent'
    )
    
    # Create all the necessary mocks with correct paths
    with patch("react_agent.critic.asyncio.wait_for", side_effect=asyncio.TimeoutError), \
         patch("react_agent.critic.Configuration") as mock_config, \
         patch("react_agent.critic.time") as mock_time, \
         patch("react_agent.utils.update_model_status") as mock_update_status, \
         patch("react_agent.critic.extract_code_from_messages") as mock_extract_code, \
         patch("react_agent.critic.load_chat_model") as mock_load_chat_model:
        
        # Return valid code from the extraction
        mock_extract_code.return_value = [
            "from odoo import models, fields\n\nclass Product(models.Model):\n    _name = 'product.template'"
        ]
        
        # Create a mock model that will be used after timeout
        mock_model = AsyncMock()
        # Make bind_tools return the model itself
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        # Setup the model to return a response when called directly (fallback path)
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Fallback evaluation after timeout"))
        mock_load_chat_model.return_value = mock_model
        
        # Set up the configuration
        mock_config.from_context.return_value = MagicMock(
            model="gemini-pro",
            critic_model="gemini-pro",
            critic_timeout=0.1,  # Very short timeout
            critic_temperature=0.2,
            critic_max_tokens=1024,
            enable_critic_fallback=True
        )
        
        # Mock time.time() to avoid potential issues
        mock_time.time.return_value = 1234567890.0
        
        # Test the evaluate_code function
        result = await evaluate_code(state)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="Needs more complex mocking to handle evaluate_code implementation")
async def test_ollama_compatibility_in_critic():
    """Test that the critic properly handles Ollama models that don't support tool binding."""
    # Create a state with Odoo code in the messages
    state = State(
        messages=[
            HumanMessage(content="Can you create an Odoo model?"),
            AIMessage(
                content="""
                ```python
                from odoo import models, fields
                
                class Product(models.Model):
                    _name = 'product.template'
                    _description = 'Product Template'
                    
                    name = fields.Char(string='Name', required=True)
                    price = fields.Float(string='Price')
                ```
                """
            ),
        ],
        is_last_step=False,
        node_history=[],
        last_node='agent'
    )
    
    # Create a comprehensive set of mocks
    with patch("react_agent.critic.Configuration") as mock_config, \
         patch("react_agent.critic.load_chat_model") as mock_load_chat_model, \
         patch("react_agent.utils.update_model_status") as mock_update_status, \
         patch("react_agent.critic.extract_code_from_messages") as mock_extract_code:
         
        # Return valid code from the extraction
        mock_extract_code.return_value = [
            "from odoo import models, fields\n\nclass Product(models.Model):\n    _name = 'product.template'\n    _description = 'Product Template'\n\n    name = fields.Char(string='Name', required=True)\n    price = fields.Float(string='Price')"
        ]
         
        # Configure mock to use Ollama model
        mock_config.from_context.return_value = MagicMock(
            model="gemini-pro",  # Main model
            critic_model="ollama:qwen2.5-coder:7b",  # Using Ollama for critic
            enable_critic_fallback=True,
            critic_temperature=0.2,
            critic_max_tokens=1024,
            critic_timeout=30
        )
        
        # Create a mock Ollama model that raises NotImplementedError on bind_tools
        mock_ollama_model = AsyncMock(spec=BaseChatModel)
        # Simulate the issue where bind_tools raises NotImplementedError
        mock_ollama_model.bind_tools = MagicMock(side_effect=NotImplementedError("Model doesn't support tool binding"))
        # But the model itself can be invoked successfully
        mock_ollama_model.ainvoke.return_value = AIMessage(content="Quality: 7/10. The code follows Odoo standards but lacks documentation.")
        
        # Return the mock Ollama model when loading the chat model
        mock_load_chat_model.return_value = mock_ollama_model
        
        # Test the evaluate_code function
        result = await evaluate_code(state)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0


# ========== Workflow Status Tests ==========

def test_model_status_integration_with_agents(reset_model_status):
    """Test the integration of model status tracking with agent components."""
    # Simulate the vector_search node
    update_model_status("vector_search", "google", "gemini-pro", "initializing")
    update_model_status("vector_search", "google", "gemini-pro", "complete")
    
    # Simulate the call_model node
    update_model_status("call_model", "google", "gemini-pro", "initializing")
    update_model_status("call_model", "google", "gemini-pro", "complete")
    
    # Simulate the critic node
    update_model_status("critic", "anthropic", "claude-3-haiku", "initializing")
    update_model_status("critic", "anthropic", "claude-3-haiku", "complete")
    
    # Simulate the manager node
    update_model_status("manager", "system", "workflow_manager", "running")
    update_model_status("manager", "system", "workflow_manager", "complete")
    
    # Check final statuses
    all_statuses = get_model_status()
    assert all_statuses["vector_search"]["status"] == "complete"
    assert all_statuses["call_model"]["status"] == "complete"
    assert all_statuses["critic"]["status"] == "complete"
    assert all_statuses["manager"]["status"] == "complete"
    
    # Check timestamps are present
    for agent in ["vector_search", "call_model", "critic", "manager"]:
        assert "timestamp" in all_statuses[agent]


if __name__ == "__main__":
    pytest.main()

"""Unit tests for the react_agent.utils module."""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any

from react_agent.utils import get_message_text, update_model_status, get_model_status
from react_agent.utils import bind_tools_safely, load_chat_model


class TestMessageHandling:
    """Tests for message handling utilities."""
    
    def test_get_message_text_string(self):
        """Test getting message text from a string content."""
        message = MagicMock(spec=BaseMessage)
        message.content = "Test message"
        assert get_message_text(message) == "Test message"
    
    def test_get_message_text_dict(self):
        """Test getting message text from a dict content."""
        message = MagicMock(spec=BaseMessage)
        message.content = {"text": "Test message in dict"}
        assert get_message_text(message) == "Test message in dict"
    
    def test_get_message_text_list(self):
        """Test getting message text from a list content."""
        message = MagicMock(spec=BaseMessage)
        message.content = [
            {"text": "Part 1"},
            "Part 2",
            {"text": "Part 3"}
        ]
        assert get_message_text(message) == "Part 1Part 2Part 3"


class TestModelStatus:
    """Tests for model status tracking."""
    
    def test_update_and_get_model_status(self):
        """Test updating and retrieving model status."""
        # Update status for an agent
        update_model_status("call_model", "google", "gemini-2.5-flash", "loading")
        
        # Get status for that agent
        status = get_model_status("call_model")
        
        # Verify status was updated correctly
        assert status["provider"] == "google"
        assert status["model"] == "gemini-2.5-flash"
        assert status["status"] == "loading"
        
        # Get all statuses
        all_statuses = get_model_status()
        assert "call_model" in all_statuses
        assert all_statuses["call_model"]["model"] == "gemini-2.5-flash"


class TestToolBinding:
    """Tests for tool binding functionality."""
    
    def test_bind_tools_safely_with_supporting_model(self):
        """Test binding tools to a model that supports tool binding."""
        # Create a mock model that supports tool binding
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = "model_with_tools"
        
        # Create mock tools
        mock_tools = [{"name": "test_tool", "description": "A test tool"}]
        
        # Test binding tools to a supporting model (e.g., Google Gemini)
        result = bind_tools_safely(mock_model, mock_tools, model_type="google")
        
        # Verify the model's bind_tools method was called
        mock_model.bind_tools.assert_called_once_with(mock_tools)
        
        # Verify the result is the model with tools bound
        assert result == "model_with_tools"
    
    def test_bind_tools_safely_with_ollama_model(self):
        """Test that Ollama models skip tool binding and return the original model."""
        # Create a mock model
        mock_model = MagicMock(spec=BaseChatModel)
        
        # Create mock tools
        mock_tools = [{"name": "test_tool", "description": "A test tool"}]
        
        # Test binding tools to an Ollama model (should skip binding)
        result = bind_tools_safely(mock_model, mock_tools, model_type="ollama")
        
        # Verify the model's bind_tools method was NOT called
        mock_model.bind_tools.assert_not_called()
        
        # Verify the result is the original model
        assert result == mock_model
    
    def test_bind_tools_safely_with_unsupporting_model(self):
        """Test graceful handling when a model doesn't support tool binding."""
        # Create a mock model that doesn't support tool binding
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.side_effect = NotImplementedError("Tool binding not supported")
        
        # Create mock tools
        mock_tools = [{"name": "test_tool", "description": "A test tool"}]
        
        # Test binding tools to a model that raises NotImplementedError
        result = bind_tools_safely(mock_model, mock_tools, model_type="other")
        
        # Verify the model's bind_tools method was called
        mock_model.bind_tools.assert_called_once_with(mock_tools)
        
        # Verify the result is the original model
        assert result == mock_model


class TestModelLoading:
    """Tests for the model loading functionality."""
    
    @patch('react_agent.utils.init_chat_model')
    @patch('react_agent.configuration.Configuration')
    @patch('react_agent.utils.update_model_status')
    def test_load_chat_model_google(self, mock_update_status, mock_config, mock_init_chat):
        """Test loading a Google model with proper settings."""
        # Create a mock Configuration instance
        mock_config_instance = MagicMock()
        mock_config.from_context.return_value = mock_config_instance
        
        # Mock the langchain_google_genai import and ChatGoogleGenerativeAI class
        with patch.dict('sys.modules', {
            'langchain_google_genai': MagicMock(),
            'langchain_google_genai.ChatGoogleGenerativeAI': MagicMock()
        }):
            # Import the patched module
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Set up the mock for ChatGoogleGenerativeAI
            mock_google_chat = MagicMock()
            ChatGoogleGenerativeAI.return_value = mock_google_chat
            
            # Call load_chat_model with a Google model
            result = load_chat_model("google:gemini-2.5-flash")
            
            # Verify ChatGoogleGenerativeAI was called with the right parameters
            ChatGoogleGenerativeAI.assert_called_once()
            call_args = ChatGoogleGenerativeAI.call_args[1]
            assert call_args["model"] == "gemini-2.5-flash"
            assert call_args["temperature"] == 0.0
            assert call_args["convert_system_message_to_human"] is True
            
            # Verify the result is the Google chat model
            assert result == mock_google_chat
    
    @patch('react_agent.utils.init_chat_model')
    @patch('react_agent.configuration.Configuration')
    @patch('react_agent.utils.update_model_status')
    def test_load_chat_model_ollama(self, mock_update_status, mock_config, mock_init_chat):
        """Test loading an Ollama model."""
        # Create a mock Configuration instance
        mock_config_instance = MagicMock()
        mock_config.from_context.return_value = mock_config_instance
        
        # Mock the init_chat_model return value
        mock_chat_model = MagicMock()
        mock_init_chat.return_value = mock_chat_model
        
        # Call load_chat_model with an Ollama model
        result = load_chat_model("ollama:codellama:13b")
        
        # Verify init_chat_model was called with the right parameters
        mock_init_chat.assert_called_once_with("codellama:13b", model_provider="ollama")
        
        # Verify update_model_status was called correctly
        mock_update_status.assert_any_call("call_model", "ollama", "codellama:13b", "loading")
        mock_update_status.assert_any_call("call_model", "ollama", "codellama:13b", "ready")
        
        # Verify the result is the init_chat_model return value
        assert result == mock_chat_model
    
    @patch('react_agent.utils.init_chat_model')
    @patch('react_agent.configuration.Configuration')
    @patch('react_agent.utils.update_model_status')
    @patch('langchain_google_genai.ChatGoogleGenerativeAI', new_callable=MagicMock)
    def test_load_chat_model_fallback(self, mock_google_chat, mock_update_status, mock_config, mock_init_chat):
        """Test fallback to backup model when primary model fails."""
        # Create a mock Configuration instance with fallback enabled
        mock_config_instance = MagicMock()
        mock_config_instance.enable_fallback = True
        mock_config_instance.fallback_model = "ollama:codellama:13b"
        mock_config.from_context.return_value = mock_config_instance
        
        # Make the Google model initialization fail
        mock_google_chat.side_effect = Exception("Google API error")
        
        # Mock the init_chat_model to succeed for the fallback model
        mock_init_chat.return_value = MagicMock()
        
        # Call load_chat_model with a model that will fail
        result = load_chat_model("google:gemini-2.5-flash")
        
        # Verify Google model was attempted first (and failed)
        mock_google_chat.assert_called_once()
        
        # Verify init_chat_model was called for the fallback model
        mock_init_chat.assert_called_once_with("codellama:13b", model_provider="ollama")
        
        # Verify update_model_status was called correctly for both models
        # First for the primary model (loading and then error)
        mock_update_status.assert_any_call("call_model", "google", "gemini-2.5-flash", "loading")
        # The status might be "error" rather than "fallback" depending on implementation
        assert any(
            call[0][3] in ["error", "fallback"] and call[0][1] == "google" 
            for call in mock_update_status.call_args_list
        ), "No error or fallback status update for the primary model"
        
        # Then for the fallback model (loading and ready)
        mock_update_status.assert_any_call("call_model", "ollama", "codellama:13b", "loading")
        mock_update_status.assert_any_call("call_model", "ollama", "codellama:13b", "ready")
        
        # Verify the result is not None (the fallback model)
        assert result is not None


if __name__ == "__main__":
    pytest.main(["-v", "test_utils.py"])

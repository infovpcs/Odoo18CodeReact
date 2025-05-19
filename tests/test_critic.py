"""Tests for the Critic Agent functionality.

This module contains tests for the code evaluation capabilities of the Critic Agent.
"""

import unittest
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from react_agent.critic import (
    extract_code_from_messages,
    generate_recommendation,
    evaluate_code,
)
from react_agent.state import State


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

    def test_evaluate_code_sync(self):
        """Synchronous wrapper for the async test_evaluate_code function."""
        # This is a placeholder that will be skipped
        # The actual test is implemented as a standalone pytest function below
        pass


@pytest.mark.asyncio
@patch("react_agent.critic.create_llm_as_judge")
async def test_evaluate_code_async(mock_create_llm_as_judge):
    """Test the evaluate_code function using pytest's async support."""
    # Create mock evaluators
    mock_quality_evaluator = AsyncMock()
    mock_quality_evaluator.ainvoke = AsyncMock(return_value={
        "comment": "Quality: 9/10. Excellent structure with proper field definitions, methods, and tracking."
    })

    mock_correctness_evaluator = AsyncMock()
    mock_correctness_evaluator.ainvoke = AsyncMock(return_value={
        "comment": "Correctness: 9/10. The model is functionally correct with proper inheritance and methods."
    })

    mock_create_llm_as_judge.side_effect = [
        mock_quality_evaluator,
        mock_correctness_evaluator,
    ]

    # Create a state with Odoo code in the messages
    state = State(
        messages=[
            HumanMessage(content="Can you create an Odoo model?"),
            AIMessage(
                content="""
                Here's an Odoo model:

                ```python
                from odoo import models, fields, api
                
                class CustomerFeedback(models.Model):
                    _name = 'customer.feedback'
                    _description = 'Customer Feedback'
                    _inherit = ['mail.thread', 'mail.activity.mixin']
                    
                    name = fields.Char(string='Reference', required=True)
                    partner_id = fields.Many2one('res.partner', string='Customer')
                    feedback = fields.Text(string='Feedback')
                    rating = fields.Selection([
                        ('1', 'Poor'),
                        ('2', 'Average'),
                        ('3', 'Good'),
                        ('4', 'Excellent')
                    ], string='Rating')
                ```
                """
            ),
        ],
        is_last_step=False,
    )

    # Mock the Configuration
    with patch("react_agent.critic.Configuration") as mock_config:
        mock_config.from_context.return_value = MagicMock(model="gemini-pro")

        # Test the evaluate_code function
        result = await evaluate_code(state)

        # Verify the result
        assert "messages" in result
        assert len(result["messages"]) == 1
        feedback = result["messages"][0].content
        assert "Code Evaluation Feedback" in feedback
        assert "Quality Assessment" in feedback
        assert "Correctness Assessment" in feedback


if __name__ == "__main__":
    unittest.main()
"""Real-time test case for the complete OdooReactAgent workflow.

This module contains a comprehensive test that simulates the entire workflow
of the OdooReactAgent, including code generation, evaluation by the Critic Agent,
and the complete interaction flow.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from react_agent.critic import evaluate_code
from react_agent.graph import call_model, should_evaluate_code
from react_agent.state import State, InputState
from react_agent.configuration import Configuration


@pytest.mark.asyncio
async def test_real_time_complete_workflow():
    """Test the complete real-time workflow of the OdooReactAgent.
    
    This test simulates a real user interaction with the agent, including:
    1. Initial request for Odoo code
    2. Code generation by the model
    3. Code evaluation by the Critic Agent
    4. Response to the evaluation
    
    The test verifies that all components work together properly and that
    the async handling in the Critic Agent functions correctly.
    """
    # Step 1: Create initial state with a user request
    initial_state = State(
        messages=[
            HumanMessage(content="Create an Odoo 18 model for tracking project tasks with fields for name, description, deadline, assigned user, and status.")
        ],
        is_last_step=False
    )
    
    # Step 2: Mock the call_model function to simulate code generation
    with patch("react_agent.graph.load_chat_model") as mock_load_chat_model:
        # Create a mock model that returns a predefined response
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(
            content="Here's an Odoo 18 model for tracking project tasks:\n\n```python\nfrom odoo import models, fields, api\nfrom datetime import timedelta\n\nclass ProjectTask(models.Model):\n    _name = 'project.task.custom'\n    _description = 'Custom Project Task'\n    _inherit = ['mail.thread', 'mail.activity.mixin']\n    _order = 'deadline, priority desc, name'\n    \n    name = fields.Char(string='Task Name', required=True, tracking=True)\n    description = fields.Html(string='Description')\n    deadline = fields.Date(string='Deadline', tracking=True)\n    user_id = fields.Many2one('res.users', string='Assigned To', tracking=True)\n    status = fields.Selection([\n        ('draft', 'Draft'),\n        ('in_progress', 'In Progress'),\n        ('done', 'Done'),\n        ('cancelled', 'Cancelled')\n    ], string='Status', default='draft', tracking=True)\n    priority = fields.Selection([\n        ('0', 'Low'),\n        ('1', 'Medium'),\n        ('2', 'High'),\n        ('3', 'Urgent')\n    ], string='Priority', default='1', tracking=True)\n    active = fields.Boolean(default=True)\n    \n    @api.model\n    def create(self, vals):\n        # Add any custom logic for task creation\n        return super(ProjectTask, self).create(vals)\n    \n    def write(self, vals):\n        # Add any custom logic for task updates\n        return super(ProjectTask, self).write(vals)\n    \n    def action_mark_done(self):\n        \"\"\"Mark the task as done.\"\"\"\n        self.write({'status': 'done'})\n    \n    def action_mark_in_progress(self):\n        \"\"\"Mark the task as in progress.\"\"\"\n        self.write({'status': 'in_progress'})\n    \n    def action_cancel(self):\n        \"\"\"Cancel the task.\"\"\"\n        self.write({'status': 'cancelled'})\n    \n    @api.onchange('deadline')\n    def _onchange_deadline(self):\n        \"\"\"Show warning if deadline is approaching.\"\"\"\n        if self.deadline:\n            today = fields.Date.today()\n            warning_days = 3\n            if self.deadline and (self.deadline - today) <= timedelta(days=warning_days):\n                return {\n                    'warning': {\n                        'title': 'Deadline Approaching',\n                        'message': f'The deadline is within {warning_days} days!'\n                    }\n                }\n```\n\nThis model includes all the requested fields and adds some additional useful features:\n\n1. It inherits from mail.thread and mail.activity.mixin for chatter and activity support\n2. Includes tracking on important fields\n3. Adds a priority field for task prioritization\n4. Includes an active field for archiving\n5. Provides action methods for changing status\n6. Adds an onchange method to warn about approaching deadlines\n7. Uses proper ordering by deadline, priority, and name\n\nYou can use this model in your Odoo 18 application for tracking project tasks."
        ))
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_load_chat_model.return_value = mock_model
        
        # Call the model to generate code
        model_result = await call_model(initial_state)
        
        # Verify the model response
        assert "messages" in model_result
        assert len(model_result["messages"]) == 1
        assert "```python" in model_result["messages"][0].content
        assert "class ProjectTask(models.Model):" in model_result["messages"][0].content
        
        # Step 3: Update state with the model's response
        updated_state = State(
            messages=initial_state.messages + model_result["messages"],
            is_last_step=False
        )
        
        # Verify that code evaluation should be triggered
        assert should_evaluate_code(updated_state) is True
        
        # Step 4: Mock the LLM-as-judge evaluators for the Critic Agent
        with patch("react_agent.critic.create_llm_as_judge") as mock_create_llm_as_judge:
            # Set up mock evaluators
            mock_quality_evaluator = AsyncMock()
            mock_quality_evaluator.ainvoke = AsyncMock(return_value={
                "comment": "Quality: 9/10. Excellent structure with proper field definitions, methods, and tracking. The code follows Odoo conventions and includes helpful features like tracking, activity support, and status actions."
            })

            mock_correctness_evaluator = AsyncMock()
            mock_correctness_evaluator.ainvoke = AsyncMock(return_value={
                "comment": "Correctness: 8/10. The model is functionally correct with proper inheritance and methods. The onchange method correctly warns about approaching deadlines. One minor issue is that the create and write methods don't add any actual custom logic."
            })

            mock_create_llm_as_judge.side_effect = [
                mock_quality_evaluator,
                mock_correctness_evaluator,
            ]
            
            # Mock the Configuration
            with patch("react_agent.critic.Configuration") as mock_config:
                mock_config.from_context.return_value = MagicMock(model="gemini-pro")
                
                # Step 5: Run the code evaluation
                critic_result = await evaluate_code(updated_state)
                
                # Verify the critic result
                assert "messages" in critic_result
                assert len(critic_result["messages"]) == 1
                feedback = critic_result["messages"][0].content
                assert "Code Evaluation Feedback" in feedback
                assert "Quality Assessment" in feedback
                assert "Correctness Assessment" in feedback
                assert "9/10" in feedback  # Quality score
                assert "8/10" in feedback  # Correctness score
                
                # Step 6: Update state with the critic's feedback
                final_state = State(
                    messages=updated_state.messages + critic_result["messages"],
                    is_last_step=False
                )
                
                # Step 7: Mock the model's response to the critic's feedback
                mock_model.ainvoke = AsyncMock(return_value=AIMessage(
                    content="Thank you for the feedback! I'll improve the create and write methods to add actual custom logic:\n\n```python\n@api.model\ndef create(self, vals):\n    # Add custom logic: set a default deadline if not provided\n    if not vals.get('deadline'):\n        vals['deadline'] = fields.Date.today() + timedelta(days=7)\n    return super(ProjectTask, self).create(vals)\n\ndef write(self, vals):\n    # Add custom logic: log status changes\n    if 'status' in vals and self.status != vals['status']:\n        self.message_post(\n            body=f\"Task status changed from {self.status} to {vals['status']}\",\n            subtype_xmlid=\"mail.mt_note\"\n        )\n    return super(ProjectTask, self).write(vals)\n```\n\nThese improvements add meaningful custom logic to the methods:\n1. The create method now sets a default deadline of 7 days from today if none is provided\n2. The write method logs status changes to the chatter\n\nThese changes address the minor issue mentioned in the correctness assessment."
                ))
                
                # Call the model again with the updated state including feedback
                final_result = await call_model(final_state)
                
                # Verify the final response
                assert "messages" in final_result
                assert len(final_result["messages"]) == 1
                assert "Thank you for the feedback" in final_result["messages"][0].content
                assert "@api.model" in final_result["messages"][0].content
                assert "def create(self, vals):" in final_result["messages"][0].content
                assert "def write(self, vals):" in final_result["messages"][0].content
                assert "message_post" in final_result["messages"][0].content


if __name__ == "__main__":
    asyncio.run(test_real_time_complete_workflow())
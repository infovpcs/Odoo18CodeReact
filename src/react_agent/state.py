"""Define the state structures for the agent."""

from __future__ import annotations

import time
import operator
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Annotated
from langchain_core.messages import BaseMessage


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[List[BaseMessage], operator.add] = field(default_factory=list)
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.
    """
    
    def __post_init__(self):
        """Ensure that messages is never empty and properly formatted to prevent errors."""
        from langchain_core.messages import HumanMessage
        
        # CRITICAL: Always provide a default message to prevent EmptyInputError
        # This is essential for the __start__ node
        if not self.messages:
            # Create a non-empty placeholder message
            self.messages = [HumanMessage(content="Please help me with Odoo 18 development")]
            print("InputState: Created default message to prevent EmptyInputError")
        
        # Handle YAML input - convert dict or plain data to proper HumanMessage
        processed_messages = []
        for msg in self.messages:
            if isinstance(msg, dict):
                # If we got a dict message (common from YAML input), convert to HumanMessage
                content = msg.get('content', str(msg))
                processed_messages.append(HumanMessage(content=content if content else "Default input"))
            elif not hasattr(msg, 'type'):
                # If it's not a proper message object with a type attribute
                content = str(msg)
                processed_messages.append(HumanMessage(content=content if content else "Default input"))
            else:
                # Keep properly formatted messages as is
                # But ensure content is not empty
                if hasattr(msg, 'content') and not msg.content:
                    msg.content = "Default input"
                processed_messages.append(msg)
        
        # Replace with properly formatted messages
        self.messages = processed_messages
        print(f"InputState initialized with messages: {self.messages}")


class State(BaseModel):
    """The state of the agent."""
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    is_last_step: bool = False
    code_attempts: int = 0
    quality_score: float = 0.0
    correctness_score: float = 0.0
    max_attempts: int = 6  # Default max attempts, can be configured
    max_steps: int = 10  # Default maximum number of steps before ending conversation
    start_time: float = Field(default_factory=time.time)  # Track when the conversation started
    tool_call_count: int = 0  # Track consecutive tool calls
    eval_count: int = 0  # Track consecutive code evaluations
    step_n: int = 0  # Current step number
    vector_search_completed: bool = False  # Flag to track if vector search was already completed
    
    # Vector search and code examples
    code_examples: List[dict] = Field(default_factory=list)  # Code examples from vector search
    vector_search_status: str = ""  # Status of vector search operation
    
    # LLM-generated planning and analysis
    generated_planning: str = ""  # LLM-generated planning document
    generated_tasks: str = ""  # LLM-generated task list
    technical_analysis: str = ""  # LLM-generated technical analysis
    initial_analysis: dict = Field(default_factory=dict)  # Initial analysis of the query
    
    # Workflow tracking
    last_node: str = ""  # Track the last node visited to prevent loops
    node_history: List[str] = Field(default_factory=list)  # History of nodes visited to detect patterns
    agent_status: dict = Field(default_factory=dict)  # Track status of each agent component
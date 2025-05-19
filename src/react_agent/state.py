"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: List[BaseMessage] = field(default_factory=list)
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


class State(BaseModel):
    """The state of the agent."""
    messages: List[BaseMessage] = Field(default_factory=list)
    is_last_step: bool = False
    code_attempts: int = 0
    quality_score: float = 0.0
    correctness_score: float = 0.0
    max_attempts: int = 3  # Default max attempts, can be configured
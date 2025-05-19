"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.critic import evaluate_code
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


# Define the function that calls the model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.odoo_18_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the nodes in our graph
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("critic", evaluate_code)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def should_evaluate_code(state: State) -> bool:
    """Determine if code evaluation is needed.

    This function checks if the last AI message contains code that should be evaluated.

    Args:
        state: The current state of the conversation.

    Returns:
        True if code evaluation is needed, False otherwise.
    """
    messages = state.messages
    
    # Only evaluate after AI messages
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    
    # Don't evaluate if the message has tool calls
    if messages[-1].tool_calls:
        return False
    
    # Check if the message contains code blocks (simple heuristic)
    if isinstance(messages[-1].content, str) and "```" in messages[-1].content:
        # Only evaluate every third code generation to avoid excessive evaluations
        code_messages_count = sum(
            1 for msg in messages 
            if isinstance(msg, AIMessage) and 
            isinstance(msg.content, str) and 
            "```" in msg.content and 
            not msg.tool_calls
        )
        
        # Evaluate if this is the first code message or every third one
        return code_messages_count == 1 or code_messages_count % 3 == 0
    
    return False


def route_model_output(state: State) -> Literal["__end__", "tools", "critic"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls
    or if code evaluation is needed.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__", "tools", or "critic").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there are tool calls, execute the requested actions
    if last_message.tool_calls:
        return "tools"
    # If code evaluation is needed, route to critic
    if should_evaluate_code(state):
        return "critic"
    # Otherwise we finish
    return "__end__"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add normal edges to create cycles
# After using tools or critic, we always return to the model
builder.add_edge("tools", "call_model")
builder.add_edge("critic", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")

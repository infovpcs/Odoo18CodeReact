"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Dict, List, Literal, Any, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
from react_agent.configuration import Configuration
from react_agent.critic import evaluate_code
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model, update_model_status
from react_agent.vector_store import get_embedding, SupabaseVectorStore

# Define vector search node
async def vector_search(state: State) -> State:
    """
    Enhance the state with vector search results and LLM-generated planning and task information.
    
    This node uses an LLM to analyze the user's query, generate detailed planning and task markdown,
    and performs semantic search in the vector database to find relevant Odoo code examples.
    
    Args:
        state: The current state of the conversation
        
    Returns:
        Updated state with vector search results and LLM-generated planning and task information
    """
    # IMPORTANT: Check if we've already processed this message to prevent multiple processing
    # This is crucial to prevent the vector_search node from being called multiple times
    if hasattr(state, 'vector_search_completed') and state.vector_search_completed:
        print("Vector search: Already processed this message, skipping duplicate processing")
        return state
    
    # Load configuration
    configuration = Configuration.from_context()
    
    # For debug purposes
    print(f"Vector search: Processing state: {state}")
    print(f"Vector search: Processing input messages: {state.messages if hasattr(state, 'messages') else 'No messages'}")
    
    # Add timeout mechanism to prevent getting stuck
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Create default message if needed
    if not hasattr(state, 'messages') or not state.messages:
        state.messages = [HumanMessage(content="Please help me with Odoo 18 development")]
        print("Vector search: Created default message for empty state")
    
    # Convert any non-message objects to HumanMessages
    processed_messages = []
    for msg in state.messages:
        if isinstance(msg, dict):
            # Handle dict messages (common from YAML input)
            content = msg.get('content', str(msg))
            processed_messages.append(HumanMessage(content=content if content else "Default input"))
        elif not hasattr(msg, 'type'):
            # Handle other non-message objects
            content = str(msg)
            processed_messages.append(HumanMessage(content=content if content else "Default input"))
        else:
            # Ensure valid content in existing messages
            if hasattr(msg, 'content') and not msg.content:
                msg.content = "Default input"
            processed_messages.append(msg)
    
    # Replace with properly formatted messages
    state.messages = processed_messages
    print(f"Vector search: Processed messages: {state.messages}")
    
    # Set a flag to indicate we've processed this message
    # This prevents the vector_search node from being called multiple times on the same message
    state.vector_search_completed = True
    # Track that we're in the vector_search node
    state.last_node = 'vector_search'
    if hasattr(state, 'node_history'):
        state.node_history.append('vector_search')
    else:
        state.node_history = ['vector_search']
        
    # Update model status to show we're starting vector search
    update_model_status("vector_search", "unknown", "vector_search", "initializing")
    
    # Extract query from input - handle different message formats
    # First try to get messages with type 'human'
    user_messages = [msg for msg in state.messages if hasattr(msg, 'type') and msg.type == 'human']
    
    # If no typed messages found, try to get content from any message
    if not user_messages and state.messages:
        # For YAML input or other formats, try to extract content from any message
        query = state.messages[-1].content if hasattr(state.messages[-1], 'content') else str(state.messages[-1])
        print(f"Vector search: Using content from untyped message: {query}")
    else:
        # Use the last human message if available
        last_user_message = user_messages[-1] if user_messages else None
        query = last_user_message.content if last_user_message else ""
        print(f"Vector search: Using content from human message: {query}")
    
    # If we couldn't extract a query, return state unchanged
    if not query.strip():
        print("Vector search: No query found in messages, returning state unchanged")
        state.vector_search_completed = True
        return state
    
    # Initialize update dictionary for state
    state_updates = {}
    
    # Simpler approach to avoid nested async calls
    # Load model with proper error handling
    try:
        # Initialize the model with status tracking
        update_model_status("vector_search", "unknown", configuration.model, "loading")
        planning_model = load_chat_model(configuration.model, agent_name="vector_search")
        
        # Check if we're using an Ollama model that might have issues
        is_ollama_model = "ollama:" in configuration.model.lower()
        
        # Use different approaches based on model type
        if is_ollama_model:
            print(f"Vector search: Using Ollama model ({configuration.model}), using simplified approach")
            # For Ollama models, use a basic template to avoid tool binding issues
            planning = f"Planning for: {query}\n\nWill implement a custom Odoo 18 module for the required functionality."
            tasks = f"Tasks for implementing: {query}\n\n1. Create module structure\n2. Implement models\n3. Create views\n4. Add controllers\n5. Write JavaScript components\n6. Test functionality"
            analysis = f"Technical analysis for: {query}\n\nWill require extending Odoo models and creating custom web components."
        else:
            # For other models, try using the model directly
            try:
                # Create planning prompt
                planning_prompt = [
                    SystemMessage(content="""You are an expert Odoo 18 developer and project planner. 
                    Your task is to analyze a user's request for an Odoo 18 feature and create:
                    1. A detailed project planning document in markdown format
                    2. A structured task list in markdown format
                    3. A technical analysis with code structure recommendations"""),
                    HumanMessage(content=query)
                ]
                
                # Direct model call - simpler approach with fewer nested asyncs
                planning_response = await planning_model.ainvoke(planning_prompt)
                planning_content = planning_response.content
                
                # Basic parsing - we'll use the full content as planning
                planning = planning_content
                tasks = "Task list will be developed during implementation."
                analysis = "Technical analysis will be provided during implementation."
                
            except Exception as e:
                print(f"Vector search: Error with model call: {str(e)}")
                # Fallback to basic template
                planning = f"Planning for: {query}\n\nWill implement a custom Odoo 18 module for the required functionality."
                tasks = f"Tasks for implementing: {query}\n\n1. Create module structure\n2. Implement models\n3. Create views\n4. Add controllers\n5. Write JavaScript components\n6. Test functionality"
                analysis = f"Technical analysis for: {query}\n\nWill require extending Odoo models and creating custom web components."
        
        # Update state with planning information
        state.generated_planning = planning
        state.generated_tasks = tasks
        state.technical_analysis = analysis
        
        print(f"Vector search: Generated planning content successfully")
        print(f"Planning: {len(state.generated_planning)} chars, Tasks: {len(state.generated_tasks)} chars")
        
        # Mark vector search as complete
        update_model_status("vector_search", "unknown", configuration.model, "completed")
        
        # Now let's try to get vector search results if configured
        # We'll do this in a separate try block to ensure planning content is saved even if vector search fails
        vector_search_attempted = False
        
    except Exception as e:
        # Catch any error in the planning process
        print(f"Vector search: Error during planning generation: {str(e)}")
        # Set basic fallback content
        state.generated_planning = f"Planning for: {query}\n\nWill implement a custom Odoo 18 module for the required functionality."
        state.generated_tasks = f"Tasks for implementing: {query}\n\n1. Create module structure\n2. Implement models\n3. Create views\n4. Add controllers\n5. Write JavaScript components\n6. Test functionality"
        state.technical_analysis = f"Technical analysis for: {query}\n\nWill require extending Odoo models and creating custom web components."
        update_model_status("vector_search", "unknown", configuration.model, "error")
        
        # Add planning data to state updates
        state_updates["generated_planning"] = planning_data.get("planning_markdown", "")
        state_updates["generated_tasks"] = planning_data.get("task_markdown", "")
        state_updates["technical_analysis"] = planning_data.get("technical_analysis", "")
        
    except Exception as e:
        print(f"Failed to generate planning and task information: {str(e)}")
        # Update status to error
        update_model_status("vector_search", "unknown", configuration.model, "error", e)
        # Create basic analysis as fallback
        state_updates["generated_planning"] = f"# Project Planning for {query}\n\nFailed to generate detailed planning."
        state_updates["generated_tasks"] = f"# Task List for {query}\n\nFailed to generate detailed tasks."
        state_updates["technical_analysis"] = f"Technical analysis unavailable: {str(e)}"
    
    # Perform basic analysis of the query as fallback
    requirements = []
    if "module" in query.lower():
        requirements.append("Custom module development")
    if "odoo 18" in query.lower() or "odoo18" in query.lower():
        requirements.append("Odoo 18 compatibility")
    if "point of sale" in query.lower() or "pos" in query.lower():
        requirements.append("Point of Sale functionality")
    if "multi-currency" in query.lower() or "currency" in query.lower():
        requirements.append("Multi-currency support")
    if "interface" in query.lower() or "ui" in query.lower():
        requirements.append("User interface customization")
    if "accounting" in query.lower():
        requirements.append("Accounting integration")
    
    # Add initial analysis to state updates
    state_updates["initial_analysis"] = {
        "query": query,
        "identified_requirements": requirements,
        "framework": "Odoo 18",
        "complexity_estimate": "medium" if len(requirements) > 2 else "low"
    }
    
    # Attempt to perform vector search if configured
    try:
        vector_store_url = os.environ.get("SUPABASE_URL")
        vector_store_key = os.environ.get("SUPABASE_KEY")
        
        if vector_store_url and vector_store_key:
            try:
                # Set a timeout for vector search operations
                async def perform_vector_search_with_timeout():
                    # Get embedding
                    embedding = await get_embedding(query)
                    
                    # Initialize vector store
                    # The SupabaseVectorStore constructor doesn't take parameters - it uses environment variables
                    # Make sure SUPABASE_URL and SUPABASE_KEY are set in the environment
                    if vector_store_url:
                        os.environ["SUPABASE_URL"] = vector_store_url
                    if vector_store_key:
                        os.environ["SUPABASE_KEY"] = vector_store_key
                        
                    vector_store = SupabaseVectorStore()
                    
                    # Search for relevant code examples
                    search_results = await vector_store.search_similar(
                        query_embedding=embedding,
                        limit=configuration.max_search_results,
                        similarity_threshold=0.7
                    )
                    
                    # Return search results
                    return search_results
                
                # Use timeout to prevent getting stuck
                search_results = await asyncio.wait_for(
                    perform_vector_search_with_timeout(),
                    timeout=20  # 20 second timeout for vector search
                )
                
                # Process search results
                code_examples = []
                for result in search_results:
                    # VectorSearchResult has content, metadata, and similarity fields
                    example = {
                        "content": result.content,
                        "relevance": result.similarity,
                        "metadata": result.metadata
                    }
                    code_examples.append(example)
                
                state.code_examples = code_examples
                state.vector_search_status = "success"
                print(f"Vector search: Found {len(code_examples)} relevant code examples")
                
            except asyncio.TimeoutError:
                print("Vector search: Vector search operation timed out")
                state.vector_search_status = "timeout"
            except Exception as e:
                print(f"Vector search: Error during vector search: {str(e)}")
                state.vector_search_status = f"error: {str(e)}"
        else:
            print("Vector search: Vector search disabled (no Supabase configuration)")
            state.vector_search_status = "disabled"
    except Exception as e:
        print(f"Vector search: Unexpected error in vector search setup: {str(e)}")
        state.vector_search_status = f"setup error: {str(e)}"
        update_model_status("vector_search", "unknown", "embedding", "error")
    
    # Mark the vector search as completed to prevent multiple processing
    state.vector_search_completed = True
    print("Vector search: Processing completed successfully")
    
    # Mark the vector search node as complete for routing
    state.last_node = 'vector_search'
    if hasattr(state, 'node_history'):
        state.node_history.append('vector_search')
    else:
        state.node_history = ['vector_search']
    
    # Update the initial analysis with the dictionary we created earlier
    if not hasattr(state, 'initial_analysis') or not state.initial_analysis:
        # Use the analysis dictionary we created at line 217
        state.initial_analysis = {
            "query": query,
            "identified_requirements": requirements,
            "framework": "Odoo 18",
            "complexity_estimate": "medium" if len(requirements) > 2 else "low"
        }
    
    return state

# Define the function that calls the model
async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent" and handle tool calls directly.

    This function prepares the prompt, initializes the model with tool binding,
    processes the response, and handles tool calls directly without routing through
    the manager node. This creates a more efficient workflow where tools are used
    to directly enhance the model's responses.
    
    This function also handles token limit management for API calls, splitting large
    inputs into smaller chunks when necessary to avoid exceeding token limits.

    Args:
        state (State): The current state of the conversation.

    Returns:
        dict: A dictionary containing the model's response message and status information.
    """
    configuration = Configuration.from_context()

    # Import token management utilities
    from react_agent.token_management import (
        should_split_messages, 
        split_messages, 
        get_model_token_limits
    )
    
    # Initialize the model with tool binding and fallback support
    try:
        # Track that we're in the call_model node
        state.last_node = 'call_model'
        if hasattr(state, 'node_history'):
            state.node_history.append('call_model')
            
        # Update model status to show we're loading the model
        from react_agent.utils import update_model_status
        update_model_status("call_model", "unknown", configuration.model, "initializing")
        
        # Load the model with the node name for status tracking
        base_model = load_chat_model(configuration.model, agent_name="call_model")
        
        # Handle models that don't support direct tool binding
        try:
            model = base_model.bind_tools(TOOLS)
        except NotImplementedError:
            # For models that don't support bind_tools, use them without tool binding
            print(f"Model {configuration.model} doesn't support direct tool binding. Using model without tool binding.")
            model = base_model
            
    except Exception as e:
        if not configuration.enable_fallback:
            raise
            
        print(f"Main model failed, falling back to: {configuration.fallback_model}")
        # Update status to show fallback
        from react_agent.utils import update_model_status
        update_model_status("call_model", "unknown", configuration.fallback_model, "fallback", e)
        
        # Fall back to the configured fallback model
        base_fallback_model = load_chat_model(configuration.fallback_model, agent_name="call_model")
        
        # Handle models that don't support direct tool binding
        try:
            model = base_fallback_model.bind_tools(TOOLS)
        except NotImplementedError:
            # For models that don't support bind_tools, use them without tool binding
            print(f"Fallback model {configuration.fallback_model} doesn't support direct tool binding. Using model without tool binding.")
            model = base_fallback_model

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage as LangChainAIMessage
    
    # Prepare the messages for the model
    messages = []
    
    # Add system prompt if not already in messages
    system_message_content = configuration.odoo_18_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Ensure system message has content
    if not system_message_content.strip():
        system_message_content = "You are an expert Odoo developer. Please help with Odoo development tasks."
    
    # Check if there's already a system message in the state
    has_system_message = False
    for msg in state.messages:
        if (isinstance(msg, dict) and msg.get('role') == 'system' and msg.get('content', '').strip()) or \
           (hasattr(msg, 'type') and msg.type == 'system' and getattr(msg, 'content', '').strip()):
            has_system_message = True
            break
    
    # Add system message if not already present
    if not has_system_message:
        # Check if we have LLM-generated planning and analysis
        if hasattr(state, 'generated_planning') and state.generated_planning and \
           hasattr(state, 'generated_tasks') and state.generated_tasks and \
           hasattr(state, 'technical_analysis') and state.technical_analysis:
            
            # Enhance system message with the generated planning and analysis
            enhanced_system_message = f"{system_message_content}\n\n"
            enhanced_system_message += """I'm providing you with a detailed analysis of the request to help you generate better code:

## Project Planning
{planning}

## Task List
{tasks}

## Technical Analysis
{analysis}

Please use this information to guide your response and ensure your code follows the architecture and requirements outlined above.""".format(
                planning=state.generated_planning,
                tasks=state.generated_tasks,
                analysis=state.technical_analysis
            )
            
            messages.append(SystemMessage(content=enhanced_system_message))
        else:
            messages.append(SystemMessage(content=system_message_content))
    
    # Add existing messages, converting to appropriate message types
    for msg in state.messages:
        try:
            if isinstance(msg, dict):
                # Convert dict to appropriate message type
                role = msg.get('role', '')
                content = str(msg.get('content', '')).strip()
                if not content:  # Skip empty messages
                    continue
                    
                if role == 'system':
                    messages.append(SystemMessage(content=content))
                elif role == 'user':
                    messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    messages.append(LangChainAIMessage(content=content))
                else:
                    # Default to human message for unknown types
                    messages.append(HumanMessage(content=content))
            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                content = str(msg.content).strip()
                if not content:  # Skip empty messages
                    continue
                    
                if msg.type == 'system':
                    if not has_system_message:  # Only add if we haven't added a system message yet
                        messages.append(SystemMessage(content=content))
                elif msg.type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg.type == 'ai':
                    messages.append(LangChainAIMessage(content=content))
            elif hasattr(msg, 'content'):
                # Handle other message types with content
                content = str(msg.content).strip()
                if content:  # Only add if content is not empty
                    messages.append(HumanMessage(content=content))
        except Exception as e:
            # Skip any messages that cause errors during processing
            print(f"Warning: Error processing message: {e}")
            continue
    
    # Ensure we have at least one non-system message
    non_system_messages = [
        msg for msg in messages 
        if not (hasattr(msg, 'type') and msg.type == 'system')
    ]
    
    # Check different workflow paths to customize the prompt appropriately
    coming_from_vector_search = hasattr(state, 'node_history') and len(state.node_history) >= 2 and state.node_history[-2] == 'vector_search'
    coming_from_critic = hasattr(state, 'node_history') and len(state.node_history) >= 2 and state.node_history[-2] == 'critic'
    has_planning_data = hasattr(state, 'generated_planning') and hasattr(state, 'generated_tasks') and hasattr(state, 'technical_analysis')
    
    # Check if we have critic feedback that needs to be addressed in code regeneration
    has_critic_feedback = hasattr(state, 'critic_feedback') and state.critic_feedback
    has_critic_score = hasattr(state, 'critic_score')
    
    # If we're coming from the critic node with feedback, add a code regeneration prompt
    if coming_from_critic and has_critic_feedback:
        # Create a targeted code regeneration prompt based on critic feedback
        score_text = f"Score: {state.critic_score:.1f}/10" if has_critic_score else ""
        code_fix_request = f"""The code reviewer provided the following feedback about the previous code implementation. Please address these issues and regenerate the code with the suggested fixes:

{score_text}
{state.critic_feedback}

Please regenerate the complete Odoo 18 module code with these improvements. Keep all the good parts while fixing the identified issues. Include all necessary files with proper Odoo 18 structure."""
        
        # Add this as a human message to ensure code regeneration with fixes
        messages.append(HumanMessage(content=code_fix_request))
        print("Added code regeneration prompt based on critic feedback")
    
    # If we have planning data from vector_search, add an explicit code generation prompt
    elif coming_from_vector_search and has_planning_data and len(non_system_messages) <= 1:
        # Create a comprehensive code generation prompt based on the planning data
        code_request = f"""Based on the planning information and technical analysis provided, please generate the Odoo 18 module code.

Project Planning:
{state.generated_planning[:500]}...

Task List:
{state.generated_tasks[:500]}...

Technical Analysis:
{state.technical_analysis[:500]}...

Please implement this Odoo 18 module with complete code structure following Odoo best practices. Include all necessary files."""
        
        # Add this as a human message to ensure code generation
        messages.append(HumanMessage(content=code_request))
        print("Added explicit code generation prompt based on planning data")
    
    # If no non-system messages, add a default user message
    elif not non_system_messages:
        messages.append(HumanMessage(content="Please generate the complete Odoo 18 module code following Odoo 18 best practices."))
    
    # Log the messages being sent to the model for debugging
    print("Sending messages to model:", [{"type": type(m).__name__, "content": getattr(m, 'content', str(m))} for m in messages])
    
    # Get the model's response with fallback support
    try:
        # Update model status to show we're running the model
        from react_agent.utils import update_model_status
        update_model_status("call_model", "unknown", getattr(model, "_llm_type", configuration.model), "running")
        
        # Track tool call count since tools bypass the manager now
        # If we came from tools node, increment the tool call count
        if hasattr(state, 'node_history') and len(state.node_history) >= 2 and state.node_history[-2] == 'tools':
            state.tool_call_count = getattr(state, 'tool_call_count', 0) + 1
            print(f"Call_model: Detected return from tools, tool_call_count = {state.tool_call_count}")
            
            # Check for too many consecutive tool calls
            if getattr(state, 'tool_call_count', 0) > 8:
                print(f"Call_model: Too many consecutive tool calls ({state.tool_call_count}), ending conversation")
                state.is_last_step = True
                return {"messages": [LangChainAIMessage(content="I've made too many consecutive tool calls. Let me summarize what I've found so far.")]}
        else:
            # Reset tool call count when not coming from tools
            state.tool_call_count = 0
        
        # Determine timeout based on model type - longer for Ollama models on CPU
        is_ollama_model = 'ollama' in configuration.model.lower() or 'ollama' in getattr(model, '_llm_type', '').lower()
        model_timeout = configuration.ollama_timeout if is_ollama_model else 120  # Use configured timeout for Ollama, 2 minutes for others
        
        # For Ollama models, we'll implement a retry mechanism with backoff
        max_retries = 5 if is_ollama_model else 1  # Increase max retries for Ollama models
        retry_count = 0
        retry_delay = 10  # Start with 10 seconds delay and increase
        response = None
        
        # Implement retry loop for model invocation
        while retry_count < max_retries:
            try:
                print(f"Invoking model (attempt {retry_count+1}/{max_retries}) with timeout of {model_timeout} seconds")
                
                # Check if this is a Google/Gemini model that might need token splitting
                is_google_model = 'google' in configuration.model.lower() or 'gemini' in configuration.model.lower()
                
                # Import token management utilities if needed
                if is_google_model:
                    from react_agent.token_management import (
                        should_split_messages, 
                        split_messages, 
                        get_model_token_limits
                    )
                    
                    # Check if we need to split messages based on token count
                    if should_split_messages(messages, configuration.model):
                        print(f"Token limit would be exceeded. Splitting into smaller chunks for {configuration.model}")
                        chunked_messages = split_messages(messages, configuration.model)
                        print(f"Split into {len(chunked_messages)} chunks based on token limits")
                        
                        # Process each chunk and compile results
                        chunked_responses = []
                        for i, chunk in enumerate(chunked_messages):
                            print(f"Processing chunk {i+1}/{len(chunked_messages)}")
                            chunk_response = await asyncio.wait_for(
                                model.ainvoke(chunk),
                                timeout=model_timeout
                            )
                            if hasattr(chunk_response, 'content') and chunk_response.content:
                                chunked_responses.append(chunk_response.content)
                        
                        # Combine the chunked responses
                        if chunked_responses:
                            combined_content = "\n\n".join(chunked_responses)
                            from langchain_core.messages import AIMessage as LangChainAIMessage
                            response = LangChainAIMessage(content=combined_content)
                        else:
                            # If no chunks were successful, raise an error to trigger fallback
                            raise ValueError("Failed to process any chunks successfully")
                    else:
                        # No splitting needed, invoke normally
                        response = await asyncio.wait_for(
                            model.ainvoke(messages),
                            timeout=model_timeout
                        )
                else:
                    # Not a Google model, invoke normally
                    response = await asyncio.wait_for(
                        model.ainvoke(messages),
                        timeout=model_timeout
                    )
                
                # Check if response contains actual content (some models return empty responses)
                # Use less strict validation for Ollama models
                valid_response = False
                
                if hasattr(response, 'content') and response.content:
                    content_length = len(response.content.strip())
                    # For Ollama models, accept any non-empty response
                    # For other models, require at least 50 characters
                    if is_ollama_model and content_length > 0:
                        valid_response = True
                        print(f"Received valid Ollama response with {content_length} characters")
                    elif not is_ollama_model and content_length > 50:
                        valid_response = True
                    elif content_length > 0:
                        print(f"Response length ({content_length} chars) is below threshold but not empty")
                        valid_response = True  # Accept any non-empty response as valid
                    
                if valid_response:
                    # Update status to complete on successful invocation
                    from react_agent.utils import update_model_status
                    update_model_status("call_model", "unknown", getattr(model, "_llm_type", configuration.model), "complete")
                    # Break out of retry loop on success
                    break
                else:
                    # Received empty response, treat as a failure and retry
                    print(f"Model returned empty response, will retry ({retry_count+1}/{max_retries})")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Waiting {retry_delay} seconds before retrying...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    response = None  # Clear the empty response
            except asyncio.TimeoutError:
                print(f"Model invocation timed out after {model_timeout} seconds (attempt {retry_count+1}/{max_retries})")
                from react_agent.utils import update_model_status
                update_model_status("call_model", "unknown", getattr(model, "_llm_type", configuration.model), "timeout")
                
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            except Exception as e:
                print(f"Error during model invocation: {str(e)}")
                from react_agent.utils import update_model_status
                update_model_status("call_model", "unknown", getattr(model, "_llm_type", configuration.model), "error", e)
                
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        # Check if we have a valid response after all retry attempts
        if response is None or not hasattr(response, 'content') or not response.content:
            # We've exhausted all retries and still don't have a valid response
            if is_ollama_model:
                error_message = LangChainAIMessage(
                    content="I apologize for the delay. The Ollama model running on your local machine needs more time to process this request. "
                           "This is common when running large language models on CPU. "
                           "You can try:"
                           "\n1. Simplifying your request"
                           "\n2. Letting the current request continue processing (it may still be working in the background)"
                           "\n3. Increasing the 'ollama_timeout' in your configuration"
                           "\n4. Using a smaller model like 'llama2:7b' or 'mistral:7b'"
                )
            else:
                error_message = LangChainAIMessage(
                    content="I apologize, but I couldn't generate a proper response after multiple attempts. "
                           "Please try again or consider using a different model."
                )
            return {"messages": [error_message]}
        
        # Cast response to the expected type
        response = cast(LangChainAIMessage, response)
        
        # Ensure the response has content
        if not getattr(response, 'content', '').strip():
            response.content = "I apologize, but I couldn't generate a response. Please try again."
        
        # Handle tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return handle_tool_calls(state, response)
            
        return {"messages": [response]}
        
    except asyncio.TimeoutError:
        print("Model call timed out after 60 seconds, trying fallback")
        if configuration.enable_fallback and hasattr(configuration, 'fallback_model'):
            try:
                # Fall back to the configured fallback model with timeout
                base_fallback_model = load_chat_model(configuration.fallback_model)
                
                # Handle models that don't support direct tool binding
                try:
                    fallback_model = base_fallback_model.bind_tools(TOOLS)
                except NotImplementedError:
                    # For models that don't support bind_tools, use them without tool binding
                    print(f"Fallback model {configuration.fallback_model} doesn't support direct tool binding. Using model without tool binding.")
                    fallback_model = base_fallback_model
                    
                response = await asyncio.wait_for(
                    fallback_model.ainvoke(messages),
                    timeout=60  # 60 second timeout for fallback
                )
                
                if hasattr(response, 'content') and response.content:
                    response_message = response.content
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_message = response.message.content
                else:
                    response_message = str(response)
                
                print("Successfully used fallback model after timeout")
                return {"messages": [AIMessage(content=response_message)]}
            except Exception as fallback_error:
                print(f"Fallback model also failed after timeout: {str(fallback_error)}")
                return {"messages": [AIMessage(content="I apologize, but both the primary and fallback models timed out. Please try again later.")]}
        else:
            return {"messages": [AIMessage(content="I apologize, but the model response timed out. Please try again later.")]}
                
    except Exception as e:
        # Check if this is a rate limit error
        is_rate_limit = (
            "429" in str(e) or 
            "quota" in str(e).lower() or 
            "rate limit" in str(e).lower() or
            "ResourceExhausted" in str(e)
        )
        
        if is_rate_limit and configuration.enable_fallback and hasattr(configuration, 'fallback_model'):
            print(f"Rate limit hit, falling back to: {configuration.fallback_model}")
            try:
                # Fall back to the configured fallback model
                base_fallback_model = load_chat_model(configuration.fallback_model)
                
                # Handle models that don't support direct tool binding
                try:
                    fallback_model = base_fallback_model.bind_tools(TOOLS)
                except NotImplementedError:
                    # For models that don't support bind_tools, use them without tool binding
                    print(f"Fallback model {configuration.fallback_model} doesn't support direct tool binding. Using model without tool binding.")
                    fallback_model = base_fallback_model
                    
                response = await asyncio.wait_for(
                    fallback_model.ainvoke(messages),
                    timeout=60  # 60 second timeout
                )
                
                if hasattr(response, 'content') and response.content:
                    response_message = response.content
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_message = response.message.content
                else:
                    response_message = str(response)
                
                print("Successfully used fallback model")
                return {"messages": [AIMessage(content=response_message)]}
            except Exception as fallback_error:
                print(f"Fallback model also failed: {str(fallback_error)}")
                # Continue to the next fallback
        
        # Fallback to simpler message format if the above fails
        try:
            fallback_messages = [SystemMessage(content=system_message_content)]
            
            # Add only the last few messages to avoid context window issues
            for msg in messages[-5:]:  # Limit to last 5 messages
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        fallback_messages.append(HumanMessage(content=msg.content))
                    elif msg.type == 'ai':
                        fallback_messages.append(AIMessage(content=msg.content))
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    model.ainvoke(fallback_messages),
                    timeout=60  # 60 second timeout
                )
                response = cast(LangChainAIMessage, response)
                
                if not getattr(response, 'content', '').strip():
                    response.content = "I apologize, but I encountered an error generating a response."
                    
                return {"messages": [response]}
            except asyncio.TimeoutError:
                # If this also times out, try the fallback model one last time
                if configuration.enable_fallback and hasattr(configuration, 'fallback_model'):
                    try:
                        fallback_model = load_chat_model(configuration.fallback_model)
                        response = await asyncio.wait_for(
                            fallback_model.ainvoke(fallback_messages),
                            timeout=60  # 60 second timeout
                        )
                        return {"messages": [AIMessage(content=getattr(response, 'content', "I apologize for the technical difficulties."))]}
                    except Exception:
                        pass
                        
                return {"messages": [AIMessage(content="I apologize, but I'm having technical difficulties. Let me try a simpler approach without using tools.")]}
            
        except Exception as inner_e:
            print(f"Fallback message handling also failed: {str(inner_e)}")
            error_msg = "I apologize, but I'm having trouble generating a response. Please try again later."
            return {"messages": [AIMessage(content=error_msg)]}
            
def handle_tool_calls(state: State, response: AIMessage) -> Dict[str, Any]:
    """Handle tool calls in the model's response."""
    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and hasattr(response, 'tool_calls') and response.tool_calls:
        # Return a single message with the error content
        return {
            "messages": [
                AIMessage(
                    id=getattr(response, 'id', None),
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    # Return the single response message
    return {"messages": [response]}

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

def route_model_output(state: State) -> Literal["__end__", "critic", "call_model"]:
    """Determine the next node based on the model's output.

    This function is used as a conditional edge router in the graph.
    It explicitly decides whether to end the conversation, evaluate code,
    or continue with the model based on clear conditions.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__", "critic", or "call_model").
    """
    # Print current routing state for debugging
    print(f"Router: Current node history: {state.node_history[-5:] if hasattr(state, 'node_history') and len(state.node_history) >= 5 else getattr(state, 'node_history', [])}")
    print(f"Router: Last node: {getattr(state, 'last_node', 'None')}")
    
    # Check if this is the last step or if manager has decided to end
    if state.is_last_step:
        print("Router: is_last_step is True, ending conversation")
        return "__end__"
    
    # Check if we've reached the maximum number of steps
    if state.step_n >= state.max_steps - 1:
        print("Router: Reached maximum steps, ending conversation")
        state.is_last_step = True
        return "__end__"
    
    # Enhanced loop detection to prevent recursion errors
    if hasattr(state, 'node_history') and len(state.node_history) >= 5:
        # Look for repeating patterns
        last_nodes = state.node_history[-10:] if len(state.node_history) >= 10 else state.node_history
        
        # Check for any node appearing too frequently in recent history
        node_counts = {node: last_nodes.count(node) for node in set(last_nodes)}
        if any(count >= 4 for node, count in node_counts.items()):
            print(f"Router: Node {max(node_counts, key=node_counts.get)} appears too frequently, ending to prevent infinite loop")
            state.is_last_step = True
            return "__end__"
            
        # Check for alternating pattern between any nodes
        if len(last_nodes) >= 6:
            pattern_detected = False
            # Check for repeating pattern of length 2
            if last_nodes[-6:] == last_nodes[-6:-4] * 3:
                pattern_detected = True
            # Check for repeating pattern of length 3
            elif len(last_nodes) >= 9 and last_nodes[-9:] == last_nodes[-9:-6] * 3:
                pattern_detected = True
                
            if pattern_detected:
                print("Router: Detected repeating pattern in node history, ending to prevent infinite loop")
                state.is_last_step = True
                return "__end__"
    
    # Check if the user's last message indicates they want to end the conversation
    if hasattr(state, 'messages') and state.messages:
        # Find the most recent human message
        for msg in reversed(state.messages):
            if hasattr(msg, 'type') and msg.type == 'human' and hasattr(msg, 'content'):
                # Check if this human message contains any exit phrases
                human_content = msg.content.lower()
                exit_phrases = ["goodbye", "bye", "exit", "quit", "end"]
                
                # Extract only whole words for matching to avoid false positives
                import re
                words = re.findall(r'\b\w+\b', human_content)
                
                if any(phrase in exit_phrases for phrase in words):
                    print(f"Router: User requested to end the conversation with phrase: {[p for p in words if p in exit_phrases]}")
                    state.is_last_step = True
                    return "__end__"
                break  # Only check the most recent human message
    
    # Note: Tool calls are now handled directly by the call_model function
    # and not routed through the manager node

    # Check if code evaluation is needed (but skip if we just came from critic to avoid loops)
    critic_transition = getattr(state, 'last_node', '') == 'critic'
    
    if should_evaluate_code(state) and not critic_transition:
        # Track that we're going to the critic node
        state.last_node = 'critic'
        # Add to node history
        if hasattr(state, 'node_history'):
            state.node_history.append('critic')
        print(f"Router: Routing to critic node, history: {state.node_history[-5:] if hasattr(state, 'node_history') and len(state.node_history) >= 5 else state.node_history}")
        return "critic"

    # Track that we're going to the call_model node
    state.last_node = 'call_model'
    # Add to node history
    if hasattr(state, 'node_history'):
        state.node_history.append('call_model')
    print(f"Router: Routing to call_model node, history: {state.node_history[-5:] if hasattr(state, 'node_history') and len(state.node_history) >= 5 else state.node_history}")
    # Default to calling the model if we're not ending
    return "call_model"

# Define a manager node to control workflow and prevent loops
async def workflow_manager(state: State) -> State:
    """
    Manager node to control the workflow and prevent loops.
    
    This function analyzes the current state and makes decisions about whether
    to continue or exit the workflow. It also tracks metrics about the conversation
    and ensures the workflow doesn't get stuck in loops.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        Dict[str, Any]: The updated state or empty dict to avoid updating messages.
    """
    # Check if we're coming from the critic node - in this case, we need to avoid updating the messages key
    coming_from_critic = getattr(state, 'last_node', '') == 'critic'
    
    # Track that we're in the manager node
    state.last_node = 'manager'
    
    # Update status to show we're in the manager node
    from react_agent.utils import update_model_status
    update_model_status("manager", "system", "workflow_manager", "running")
    
    # Increment step counter
    state.step_n += 1
    
    # Add to node history if not already there
    if hasattr(state, 'node_history'):
        state.node_history.append('manager')
    else:
        state.node_history = ['manager']
        
    # Collect status info from all agents
    from react_agent.utils import get_model_status
    state.agent_status = get_model_status()
    
    # Print debug information
    print(f"Manager: Step {state.step_n}, Node history: {state.node_history[-5:] if len(state.node_history) >= 5 else state.node_history}")
    
    # Check if we should end the conversation
    if state.step_n >= state.max_steps - 1:
        state.is_last_step = True
        print(f"Manager: Reached maximum steps ({state.max_steps}), ending conversation")
    
    # Enhanced loop detection to prevent recursion errors
    if len(state.node_history) >= 8:
        # Count occurrences of each node in the last steps
        last_nodes = state.node_history[-8:]
        node_counts = {node: last_nodes.count(node) for node in set(last_nodes)}
        
        # If any node appears too frequently, end the conversation (lowered threshold)
        if any(count >= 3 for node, count in node_counts.items()):
            state.is_last_step = True
            print(f"Manager: Detected potential loop, node counts: {node_counts}, ending conversation")
            
        # Look for repeating sequences that might indicate a loop
        if len(last_nodes) >= 6:
            # Check for repeating pattern of length 2
            if last_nodes[-6:] == last_nodes[-6:-4] * 3:
                print("Manager: Detected repeating pattern of length 2, ending conversation")
                state.is_last_step = True
                
        # Check if we have critic evaluation results
        has_critic_scores = hasattr(state, 'overall_score')
        has_critic_feedback = hasattr(state, 'critic_feedback') and state.critic_feedback
        critic_run_count = getattr(state, 'critic_run_count', 0)
        
        # If we have a good score from the critic (>= 7.0), we can end the loop
        if has_critic_scores and getattr(state, 'overall_score', 0) >= 7.0:
            print(f"Manager: Code received good score ({state.overall_score:.1f}/10), no further fixes needed")
            # Don't end conversation but skip further critic evaluations
            state.skip_critic = True
        
        # Check if we've had too many critic evaluation attempts (limit to 3)
        if critic_run_count >= 3:
            print(f"Manager: Reached maximum critic evaluation attempts ({critic_run_count}), finishing the loop")
            # Don't send to critic again, but don't end conversation
            state.skip_critic = True
        
        # Check total number of state transitions - if too high, likely in a loop
    # We've increased the recursion_limit to 50, but still want to catch loops before that
    if len(state.node_history) >= 40:  # Increased to match our new recursion_limit of 50
        print(f"Manager: Workflow has made {len(state.node_history)} transitions, likely in a loop. Ending conversation.")
        state.is_last_step = True

        # Check if we've been in critic too many times in a row
        if len(state.node_history) >= 5 and state.node_history[-5:].count('critic') >= 3:
            print("Manager: Too many critic evaluations in a row, forcing model response")
            # Skip critic and go directly to call_model
            state.skip_critic = True

        # Check if the last few nodes show a pattern of getting stuck
        if len(state.node_history) >= 8:
            # Look for repeating patterns like manager->critic->call_model->manager->critic->call_model
            pattern = state.node_history[-8:]
            if pattern.count('manager') >= 3 and pattern.count('critic') >= 2:
                # If we detect a repeating pattern, force skipping critic to break the loop
                print("Manager: Detected repeating pattern in workflow, skipping further critic evaluations")
                state.skip_critic = True
    
    # Always return the full state - with conditional edges we don't need to worry about InvalidUpdateError
    # as each edge transition is properly handled by the graph
    return state

# Define a new graph with proper conditional edges
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Register all nodes
builder.add_node("vector_search", vector_search)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("critic", evaluate_code)
builder.add_node("manager", workflow_manager)

# Set the entrypoint to vector_search
builder.add_edge("__start__", "vector_search")

# Connect vector_search to call_model (first step after input)
builder.add_edge("vector_search", "call_model")

# Define a function to handle conditional routing from call_model node
def route_call_model_output(state: State) -> Literal["tools", "manager"]:
    """Determine whether to route to tools or manager based on model output.
    
    Args:
        state (State): The current state of the conversation.
        
    Returns:
        str: The name of the next node ("tools" or "manager").
    """
    # Check if we have messages and the last message is an AI message with tool calls
    has_messages = hasattr(state, 'messages') and state.messages
    if has_messages:
        last_message = state.messages[-1]
        has_tool_calls = (isinstance(last_message, AIMessage) and 
                         hasattr(last_message, 'tool_calls') and 
                         last_message.tool_calls)
        if has_tool_calls:
            print(f"Router: Detected {len(last_message.tool_calls)} tool calls, routing to tools node")
            return "tools"
    
    # Default to manager if no tool calls are present
    print("Router: No tool calls detected, routing to manager node")
    return "manager"

# Use conditional edges for model output routing with the dedicated function
builder.add_conditional_edges("call_model", route_call_model_output)

# Tools always go back to call_model to process tool results
builder.add_edge("tools", "call_model")

# Manager uses route_model_output to determine the next node
builder.add_conditional_edges(
    "manager",
    route_model_output  # Use the existing function that returns "__end__", "critic", or "call_model"
)

# Critic always goes back to manager for evaluation
builder.add_edge("critic", "manager")

# Get configuration for graph compilation
config = Configuration.from_context()

# Compile the builder into an executable graph with configurable recursion limit
graph = builder.compile(
    name="ReAct Agent"
)
"""Define a Critic Agent for evaluating Odoo 18 generated code.

This module implements a code evaluation agent using OpenEvals to assess
code quality, correctness, and adherence to Odoo standards.
"""

from typing import Dict, List, Optional, Any

from langchain_core.messages import AIMessage, HumanMessage
from openevals.llm import create_llm_as_judge

from react_agent.configuration import Configuration
from react_agent.state import State
import asyncio

# Define Odoo-specific evaluation prompts
ODOO_CODE_QUALITY_PROMPT = """
You are an expert Odoo developer tasked with evaluating the quality of Odoo 18 code.
You will be given a piece of code and need to evaluate it based on the following criteria:

1. Adherence to Odoo coding standards and conventions
2. Proper use of Odoo ORM methods and APIs
3. Code organization and structure
4. Readability and maintainability
5. Performance considerations
6. Security best practices

Code to evaluate:
{outputs}

Please provide a detailed assessment of the code quality, highlighting both strengths and areas for improvement.
Your evaluation should be specific to Odoo 18 development practices.

Score the code on a scale of 1-10, where 1 is poor quality and 10 is excellent quality.
Provide your score as a number followed by a detailed explanation.
"""

ODOO_CORRECTNESS_PROMPT = """
You are an expert Odoo developer tasked with evaluating the correctness of Odoo 18 code.
You will be given a piece of code and need to evaluate it based on the following criteria:

1. Functional correctness (does the code do what it's supposed to do?)
2. Proper use of Odoo APIs and methods
3. Handling of edge cases and errors
4. Compatibility with Odoo 18 specifically

**Additional Considerations:**
- Implement `mail.thread` integration for chatter functionality.
- Follow Odoo 18 view guidelines (use `list` instead of `tree`).
- Replace deprecated `attrs` with Odoo 18-compatible options.
- Ensure security, error handling, and thorough testing.

Code to evaluate:
{outputs}

Please provide a detailed assessment of the code correctness, highlighting any issues or bugs.
Your evaluation should be specific to Odoo 18 development practices.

Score the code on a scale of 1-10, where 1 is incorrect and 10 is perfectly correct.
Provide your score as a number followed by a detailed explanation.
"""

async def evaluate_code(state: State) -> Dict[str, List[AIMessage]]:
    # Extract code to evaluate from the last message
    messages = state.messages
    last_message = messages[-1]
    code_to_evaluate = last_message.content
    
    # Initialize evaluators with Google model
    quality_evaluator = create_llm_as_judge(prompt=ODOO_CODE_QUALITY_PROMPT, model="google_genai:gemini-2.0-flash")
    correctness_evaluator = create_llm_as_judge(prompt=ODOO_CORRECTNESS_PROMPT, model="google_genai:gemini-2.0-flash")
    
    # Run evaluations
    quality_result = await asyncio.to_thread(lambda: quality_evaluator(outputs=code_to_evaluate))
    correctness_result = await asyncio.to_thread(lambda: correctness_evaluator(outputs=code_to_evaluate))
    
    # Extract numerical scores using regex
    import re
    quality_score_match = re.search(r'Score: (\d+(\.\d+)?)', quality_result.get('comment', ''))
    correctness_score_match = re.search(r'Score: (\d+(\.\d+)?)', correctness_result.get('comment', ''))
    
    # Update state with scores
    if quality_score_match:
        state.quality_score = float(quality_score_match.group(1))
    if correctness_score_match:
        state.correctness_score = float(correctness_score_match.group(1))
    
    # Format feedback with dynamic prompt for model
    feedback = f"""## Code Evaluation Feedback
    
    ### Quality Assessment (Score: {state.quality_score}/10)
    {quality_result.get('comment', 'No quality feedback available.')}
    
    ### Correctness Assessment (Score: {state.correctness_score}/10)
    {correctness_result.get('comment', 'No correctness feedback available.')}
    
    ### Recommendations
    Based on the above assessments, consider making the following improvements to your code:
    
    1. {generate_recommendation(quality_result, correctness_result, 1)}
    2. {generate_recommendation(quality_result, correctness_result, 2)}
    3. {generate_recommendation(quality_result, correctness_result, 3)}
    
    Please revise your code based on this feedback. This is attempt {state.code_attempts} of {state.max_attempts}.
    """
    
    return {
        "messages": [
            AIMessage(
                content=feedback
            )
        ]
    }


def generate_recommendation(quality_result: Dict, correctness_result: Dict, index: int) -> str:
    """Generate a recommendation based on quality and correctness evaluations.
    
    Args:
        quality_result: The quality evaluation result
        correctness_result: The correctness evaluation result
        index: The recommendation index (1-3)
        
    Returns:
        A recommendation string
    """
    # Extract comments from evaluation results
    quality_comment = quality_result.get('comment', '')
    correctness_comment = correctness_result.get('comment', '')
    
    # Look for improvement suggestions in the comments
    import re
    suggestions = []
    
    # Extract points from quality comment
    quality_points = re.findall(r'\d+\. ([^\n]+)', quality_comment)
    suggestions.extend(quality_points)
    
    # Extract points from correctness comment
    correctness_points = re.findall(r'\d+\. ([^\n]+)', correctness_comment)
    suggestions.extend(correctness_points)
    
    # If no structured points found, look for sentences with improvement keywords
    if not suggestions:
        improvement_keywords = ['improve', 'should', 'could', 'better', 'consider', 'recommend']
        for keyword in improvement_keywords:
            pattern = f'[^.!?]*{keyword}[^.!?]*[.!?]'
            suggestions.extend(re.findall(pattern, quality_comment, re.IGNORECASE))
            suggestions.extend(re.findall(pattern, correctness_comment, re.IGNORECASE))
    
    # If still no suggestions, provide generic recommendations
    generic_recommendations = [
        "Ensure proper documentation for all methods and classes",
        "Follow Odoo naming conventions for models, fields, and methods",
        "Add appropriate security checks and access controls",
        "Optimize database queries to improve performance",
        "Implement proper error handling and validation"
    ]
    
    # Return a suggestion based on the index, or a generic one if not enough suggestions
    if index <= len(suggestions):
        return suggestions[index - 1].strip()
    elif index <= len(generic_recommendations):
        return generic_recommendations[index - 1]
    else:
        return "Review your code for any additional improvements"


def extract_code_from_messages(messages: List[Any]) -> Optional[str]:
    """Extract code blocks from the conversation messages.
    
    Args:
        messages: List of conversation messages.
        
    Returns:
        str or None: Extracted code or None if no code is found.
    """
    code_blocks = []
    
    for message in messages:
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            # Simple extraction of code blocks between triple backticks
            if isinstance(content, str):
                code_start = content.find("```")
                while code_start != -1:
                    code_start += 3
                    # Skip language identifier if present
                    if content[code_start:].find("\n") != -1:
                        code_start = content.find("\n", code_start) + 1
                    
                    code_end = content.find("```", code_start)
                    if code_end != -1:
                        code_blocks.append(content[code_start:code_end].strip())
                        content = content[code_end + 3:]
                        code_start = content.find("```")
                    else:
                        break
    
    return "\n\n".join(code_blocks) if code_blocks else None


def generate_recommendation(quality_result: Dict[str, Any], correctness_result: Dict[str, Any], index: int) -> str:
    """Generate a specific recommendation based on evaluation results.
    
    Args:
        quality_result: Results from the quality evaluation.
        correctness_result: Results from the correctness evaluation.
        index: Index of the recommendation to generate.
        
    Returns:
        str: A recommendation for improving the code.
    """
    # Extract comments from results
    quality_comment = quality_result.get('comment', '')
    correctness_comment = correctness_result.get('comment', '')
    
    # Default recommendations if specific ones can't be extracted
    default_recommendations = [
        "Ensure your code follows Odoo's naming conventions and structure.",
        "Use Odoo ORM methods instead of direct SQL queries when possible.",
        "Add proper docstrings and comments to improve code maintainability."
    ]
    
    # Try to extract specific recommendations from the comments
    # This is a simple implementation and could be improved with more sophisticated NLP
    if index <= len(default_recommendations):
        return default_recommendations[index - 1]
    else:
        return "Review the feedback above and make appropriate improvements."
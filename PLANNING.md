# Project Planning

## Agent Graph

The project follows an enhanced agent graph structure:

- **Start**: The entry point of the workflow.
- **Agent**: The core processing unit that interacts with tools.
- **Critic Agent**: A specialized evaluation agent that reviews code generated for Odoo 18.
- **Tools**: External resources or services the agent can utilize.
- **End**: The termination point of the workflow.

This structure allows for modular and scalable integration with various tools, including online search capabilities, custom MCP servers, and code evaluation frameworks.

## Architecture
- Utilize LangGraph for agent-based workflows.
- Integrate the LangGraph template for React Agent Python to enhance project structure and functionality.
- Integrate Gemini model for enhanced AI capabilities.
- Follow Odoo module structure with standard directories.
- Implement custom chatbot interface in `app.py` for Odoo 18 React Agent interaction.
- Implement a Critic Agent node in the LangGraph workflow using OpenEvals:
  - Use LLM-as-judge evaluators from OpenEvals to assess code quality, correctness, and adherence to Odoo standards
  - Create a feedback loop where evaluation results are passed back to the main agent
  - Implement code improvement suggestions based on evaluation feedback

## Goals
- Implement a simple workflow using Gemini and LangGraph.
- Ensure compatibility with Odoo React Agent.
- Create an effective code evaluation and improvement system using OpenEvals.
- Improve code quality and adherence to Odoo 18 standards through automated feedback.

## Style
- Use Python as the primary language.
- Follow PEP8 guidelines.

## Constraints
- Maintain cross-version compatibility with Odoo.
- Ensure modular code structure.

## Recent Updates
- Fixed a prompt formatting bug in the agent graph by removing an unused `name` argument, ensuring compatibility with the defined prompt template and preventing runtime errors.
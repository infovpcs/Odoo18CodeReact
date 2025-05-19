# Task List

## Current Tasks

- **Integrate Agent Graph**: Implement the agent graph structure as outlined in the PLANNING.md.
- **Tool Integration**: Connect the agent with online search tools and custom MCP servers.
- **LangGraph Template Setup**: Configure and adapt the LangGraph template for React Agent Python.
- **Implement Critic Agent**: Add a new node to the LangGraph workflow that evaluates Odoo 18 generated code using OpenEvals framework.
  - Integrate with `https://github.com/langchain-ai/openevals` for code evaluation capabilities
  - Use the implementation pattern from `https://github.com/catherine-langchain/agentevals/blob/main/react-agent-eval.ipynb` as reference
  - Create feedback loop where evaluation results are passed back to the main agent for code improvement

## Discovered During Work

- **Add Unit Tests**: Ensure the agent graph and tool integrations are thoroughly tested.
- **Document API Endpoints**: Provide detailed documentation for any new API endpoints created during integration.
- **Create Odoo-Specific Evaluation Criteria**: Develop custom evaluation prompts for OpenEvals that specifically target Odoo 18 code quality standards.
- **Add Dependencies**: Update project dependencies to include OpenEvals and related packages.
- **Implement Evaluation Metrics**: Define clear metrics for code quality assessment and track improvements over time.

## Completed Tasks

- **Prompt Formatting Fix**: Removed an unused `name` argument from the Odoo 18 system prompt formatting in the agent graph to resolve a KeyError and align with the prompt template.
- **Custom Chatbot Implementation**: Added a custom chatbot in `app.py` for Odoo 18 React Agent interaction.
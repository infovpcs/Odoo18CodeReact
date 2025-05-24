# Project Planning

## Agent Graph

The project follows an enhanced agent graph structure:

- **Start**: The entry point of the workflow.
- **Agent**: The core processing unit that interacts with tools.
- **Critic Agent**: A specialized evaluation agent that reviews code generated for Odoo 18.
- **Tools**: External resources or services the agent can utilize, including the Odoo 18 Code Search/Load Utility.
- **End**: The termination point of the workflow.

This structure allows for modular and scalable integration with various tools, including online search capabilities, custom MCP servers, Odoo 18 code search/validation, and code evaluation frameworks.

## Architecture
- Utilize LangGraph for agent-based workflows.
- Integrate the LangGraph template for React Agent Python to enhance project structure and functionality.
- Integrate Gemini model for enhanced AI capabilities with Ollama fallback.
- Follow Odoo module structure with standard directories.
- Implement custom chatbot interface in `app.py` for Odoo 18 React Agent interaction.
- Implement an enhanced Critic Agent node in the LangGraph workflow using OpenEvals:
  - Use LLM-as-judge evaluators from OpenEvals to assess code quality, correctness, and adherence to Odoo standards
  - Create a feedback loop where evaluation results are passed back to the main agent
  - Implement code improvement suggestions based on evaluation feedback
  - Provide direct access to Odoo codebase search and web tools for more informed evaluation
  - Support models that don't implement tool binding (like Ollama) via custom handler
  - Organize prompt templates in a dedicated prompts.py file for better maintainability
  - Implement retry mechanisms and better error handling for critic evaluations
- Implement Odoo 18 Code Retrieval System with tiered approach:
  1. **Local Repository (Primary)**: 
     - Direct file access for specific file requests
     - Fast grep-like search for exact pattern matching
     - No network latency, no rate limits
  2. **Supabase Vector Database RAG (Enhanced)**: 
     - Semantic search using embeddings of Odoo 18 codebase stored in Supabase
     - PostgreSQL pgvector extension for efficient similarity search
     - Metadata filtering by module, file type, and Odoo version
     - Structured indexing of models, controllers, views, and other Odoo components
     - Better understanding of code semantics and relationships
     - Optimized for conceptual queries and finding relevant code examples
  3. **External APIs (Fallback)**:
     - GitHub API when local resources aren't available
     - Hugging Face dataset as final fallback
  - Validate generated code against Odoo 18 best practices and deprecations
  - Support both Main and Critic agents for code generation and validation
  - Integrate validation results into Critic feedback and scoring

## Goals
- Implement a robust workflow using Gemini and LangGraph with Ollama fallback.
- Ensure compatibility with Odoo React Agent.
- Create an effective code evaluation and improvement system using OpenEvals.
- Improve code quality and adherence to Odoo 18 standards through automated feedback.
- Enhance agent capabilities with high-performance Odoo 18 code retrieval system.
- Optimize response time and relevance through local repository and RAG approaches.
- Ensure resilience through multi-tiered fallback mechanisms.

## Style
- Use Python as the primary language.
- Follow PEP8 guidelines.

## Constraints
- Maintain cross-version compatibility with Odoo.
- Ensure modular code structure.

## Recent Updates
- Redesigned the Odoo 18 Code Retrieval System to use a tiered approach with local repository as primary source, vector database RAG for semantic search, and external APIs as fallbacks.
- Implemented Ollama model fallback mechanism for when the primary Gemini model encounters rate limits or failures.
- Fixed NotImplementedError when calling bind_tools() on Ollama models (qwen2.5-coder:7b and llama3:8b) by adding try/except blocks and gracefully falling back to using the base model without tool binding.
- Enhanced error handling and timeout management in the agent graph to prevent getting stuck in loops.
- Fixed a prompt formatting bug in the agent graph by removing an unused `name` argument, ensuring compatibility with the defined prompt template and preventing runtime errors.
- Successfully integrated Supabase vector database with pgvector extension for semantic search of Odoo code.
- Implemented Google embedding API integration for generating embeddings, replacing OpenAI embeddings.
- Enhanced the vector database population script to handle multiple file types from the Odoo codebase (Python, XML, JavaScript, CSS, etc.).
- Added chunking functionality to handle large files and avoid API payload size limits.
- Updated documentation with comprehensive instructions for setting up and using the Supabase vector database.
- Leveraged LangGraph Studio's hot-reload capability for faster development workflow, which automatically applies code changes without needing to manually restart the server.
"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from os import name
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent.prompts import SYSTEM_PROMPT, ODOO_18_SYSTEM_PROMPT

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    odoo_18_prompt: str = field(
        default=ODOO_18_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for Odoo 18 React Agent specific interactions. "
            "This prompt sets the context and behavior for Odoo 18 React development."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google:gemini-1.5-flash",  # Using gemini-1.5-flash which has a free quota available
        metadata={
            "description": "The language model to use for the agent's main interactions. "
            "Should be in the form: provider:model-name. "
            "For reliable tool binding, use models like OpenAI, Anthropic, or Google Gemini. "
            "Options include: google:gemini-pro, openai:gpt-4-turbo, anthropic:claude-3-haiku"
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    
    fallback_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="ollama:llama3.2:latest",
        metadata={
            "description": "The fallback language model to use when the primary model fails or reaches rate limits. "
                        "Should be in the form: provider:model-name."
        },
    )
    
    enable_fallback: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable fallback to the fallback model when the primary model fails or reaches rate limits."
        },
    )
    
    ollama_base_url: str = field(
        default="http://localhost:11434",
        metadata={
            "description": "The base URL for the Ollama API. Defaults to 'http://localhost:11434'."
        },
    )
    
    ollama_timeout: int = field(
        default=300,
        metadata={
            "description": "The timeout in seconds for Ollama API requests. Defaults to 300 seconds (5 minutes) to accommodate CPU-based Ollama models."
        },
    )
    
    critic_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google:gemma-3-27b-it",  # Using gemini-1.5-flash which has a free quota available
        metadata={
            "description": "The language model to use for the critic/validator. "
                        "This model evaluates code quality and correctness. "
                        "Should be in the form: provider:model-name. "
                        "Options include: google:gemini-pro, openai:gpt-4-turbo, anthropic:claude-3-haiku"
        },
    )
    
    enable_critic_fallback: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable fallback for the critic model when it fails or reaches rate limits."
        },
    )
    
    critic_temperature: float = field(
        default=0.2,
        metadata={
            "description": "The temperature to use for the critic model. Lower values make the output more deterministic."
        },
    )
    
    critic_max_tokens: int = field(
        default=2000,
        metadata={
            "description": "The maximum number of tokens to generate for critic evaluations."
        },
    )
    
    recursion_limit: int = field(
        default=50,
        metadata={
            "description": "The maximum number of graph recursions allowed before throwing an error. "
            "Increase this value if your workflow requires more steps, but be careful to avoid infinite loops."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

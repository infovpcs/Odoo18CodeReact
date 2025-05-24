"""Odoo 18 Code Retrieval System for Main and Critic agents.

This module provides a tiered approach for retrieving Odoo 18 code:
1. Local Repository (Primary): Fast direct file access and grep-like search
2. Vector Database RAG (Enhanced): Semantic search using Supabase vector database
3. External APIs (Fallback): GitHub API and Hugging Face dataset

It also includes validation against Odoo 18 best practices and deprecations.
"""

from typing import Any, Dict, List, Optional, Union, cast
import os
import re
import json
import aiohttp
import asyncio
import subprocess
import glob
import time
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool, StructuredTool
from react_agent.configuration import Configuration

# Get configuration
config = Configuration.from_context()

# Local repository constants
ODOO_LOCAL_REPO = os.getenv("ODOO_LOCAL_REPO_PATH", "/Users/vinusoft85/workspace/odoo18_repo")
ODOO_REPO_NAME = "odoo/odoo"
ODOO_BRANCH = "18.0"

# GitHub API constants (fallback)
GITHUB_API_URL = "https://api.github.com"
ODOO_REPO = "odoo/odoo"

# Hugging Face dataset fallback constants
HF_DATASET_REPO = "odoo/odoo-18-code-samples"
HF_API_URL = "https://huggingface.co/api"

# API tokens
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Check if local repository exists
LOCAL_REPO_EXISTS = os.path.isdir(ODOO_LOCAL_REPO)

# Deprecation patterns and best practices
ODOO_DEPRECATIONS = [
    {"pattern": r"from\s+openerp", "message": "'openerp' import is deprecated, use 'from odoo' instead"},
    {"pattern": r"import\s+openerp", "message": "'openerp' import is deprecated, use 'import odoo' instead"},
    {"pattern": r"\bopenerp\b", "message": "'openerp' keyword is deprecated, use 'odoo' instead"},
    {"pattern": r"\bcr\s*,\s*uid\b", "message": "Old API style (cr, uid) is deprecated, use self.env"},
    {"pattern": r"osv\.osv", "message": "osv.osv is deprecated, use models.Model instead"},
    {"pattern": r"osv\.memory", "message": "osv.memory is deprecated, use models.TransientModel instead"},
    {"pattern": r"report_sxw", "message": "report_sxw is deprecated, use QWeb reports instead"},
    {"pattern": r"_columns\s*=", "message": "_columns dict is deprecated, use field definitions directly in the class"},
    {"pattern": r"fields\.function", "message": "fields.function is deprecated, use computed fields instead"},
]

ODOO_BEST_PRACTICES = [
    {"pattern": r"sudo\(\)\.(create|write|unlink|browse)", "message": "Avoid using sudo() directly with create/write/unlink operations"},
    {"pattern": r"env\s*\[[\"']\w+[\"']\]", "message": "Prefer self.env[model] over string access for better static analysis"},
    {"pattern": r"search\([^)]+\)\.(create|write|unlink|browse)", "message": "Consider using recordsets directly instead of search().action"},
    {"pattern": r"@api\.model\s+def\s+create\s*\(", "message": "For create methods, @api.model_create_multi is preferred over @api.model"},
    {"pattern": r"for\s+\w+\s+in\s+self\b.*?:\s*\w+\.write\s*\(", "message": "Consider using self.write() instead of looping through records"},
    {"pattern": r"_auto\s*=\s*False", "message": "Avoid using _auto = False unless absolutely necessary, as it disables automatic table creation"},
]


async def _github_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Make a request to the GitHub API.
    
    Args:
        endpoint: The API endpoint to request
        params: Optional query parameters
        
    Returns:
        The JSON response from the API
    """
    configuration = Configuration.from_context()
    github_token = GITHUB_TOKEN
    
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{GITHUB_API_URL}/{endpoint}", 
                params=params,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"GitHub API error: {response.status}", "message": await response.text()}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}


async def _huggingface_fallback(query: str) -> Dict[str, Any]:
    """
    Fallback to Hugging Face dataset when GitHub is unavailable.
    
    Args:
        query: The search query for Odoo code
        
    Returns:
        Dictionary with search results
    """
    # This would be implemented to search a pre-loaded Hugging Face dataset
    # containing Odoo 18 code samples
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{HF_API_URL}/datasets/{HF_DATASET_REPO}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                params={"search": query}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": "Hugging Face API error", "results": []}
    except Exception as e:
        return {"error": f"Hugging Face fallback failed: {str(e)}", "results": []}


async def _search_odoo_code(query: str, module: Optional[str] = None, file_type: str = "py") -> Dict[str, Any]:
    """
    This tool searches for Odoo 18 code matching the query using multiple sources:
    1. Local repository (if available)
    2. Vector database (if configured)
    3. GitHub API (fallback)
    4. Hugging Face dataset (final fallback)
    
    Args:
        query: The search query (e.g., 'class Partner', 'def _compute_amount')
        module: Optional specific Odoo module to search within (e.g., 'sale', 'account')
        file_type: File extension to filter by (default: 'py')
        
    Returns:
        Dictionary with search results including file paths and repository information
    """
    config = Configuration.from_context()
    
    # Try vector search if Supabase is configured
    try:
        import os
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
            vector_results = await _search_vector_db(query, module, file_type)
            if vector_results and not vector_results.get("error"):
                return vector_results
    except Exception as e:
        # Log the error but continue to fallback
        print(f"Vector search error: {str(e)}")
    
    # Prepare the search query for GitHub
    search_query = query
    if module:
        search_query = f"{query} path:addons/{module}"
    else:
        search_query = f"{query} path:addons"
    
    # Add repo and branch constraints
    search_query = f"{search_query} repo:{ODOO_REPO} branch:{ODOO_BRANCH}"
    
    try:
        # Try GitHub API
        github_result = await _github_request(
            f"/search/code",
            {"q": search_query, "per_page": 10}
        )
        
        if "error" not in github_result:
            return {
                "source": "github",
                "query": query,
                "module": module,
                "total_count": github_result.get("total_count", 0),
                "results": [
                    {"path": item.get("path", ""), "url": item.get("html_url", "")} 
                    for item in github_result.get("items", [])
                ]
            }
    except Exception as e:
        # Log the error but continue to fallback
        print(f"GitHub API error: {str(e)}")
    
    # Fallback to Hugging Face dataset
    try:
        hf_result = await _huggingface_fallback(query)
        return {
            "source": "huggingface_fallback",
            "query": query,
            "module": module,
            "total_count": hf_result.get("total_count", 0),
            "results": [
                {"path": item.get("path", ""), "url": item.get("html_url", "")} 
                for item in hf_result.get("results", [])
            ]
        }
    except Exception as e:
        # Log the error but continue to fallback
        print(f"Hugging Face fallback error: {str(e)}")
        return {
            "error": f"Hugging Face fallback error: {str(e)}",
            "query": query,
            "module": module,
            "results": []
        }


async def _load_odoo_code(file_path: str) -> Dict[str, Any]:
    """
    Load Odoo 18 code file from GitHub repository.
    
    This tool loads the content of a specific file from the Odoo 18 GitHub repository.
    
    Args:
        file_path: The path to the file in the repository (e.g., 'addons/sale/models/sale_order.py')
        
    Returns:
        Dictionary with the file content and path information
    """
    config = Configuration.from_context()
    
    # Construct the raw content URL
    raw_url = f"https://raw.githubusercontent.com/{ODOO_REPO}/{ODOO_BRANCH}/{file_path}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(raw_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return {
                        "file_path": file_path,
                        "content": content,
                        "url": raw_url
                    }
                else:
                    return {
                        "error": f"Failed to load file (HTTP {response.status})",
                        "file_path": file_path
                    }
    except Exception as e:
        return {
            "error": f"Failed to load file: {str(e)}",
            "file_path": file_path
        }


def _validate_odoo_code(code: str) -> Dict[str, Any]:
    """
    Validate Odoo code against best practices and deprecation warnings.
    
    This tool checks Odoo code for deprecated patterns and best practice violations.
    
    Args:
        code: The Odoo code to validate
        
    Returns:
        Dictionary with validation results including deprecation warnings and best practice suggestions
    """
    
    results = {
        "valid": True,
        "deprecation_warnings": [],
        "best_practice_suggestions": []
    }
    
    # Check for deprecation warnings
    for item in ODOO_DEPRECATIONS:
        matches = re.finditer(item["pattern"], code)
        found_matches = False
        for match in matches:
            found_matches = True
            results["deprecation_warnings"].append({
                "line": code.count('\n', 0, match.start()) + 1,
                "pattern": match.group(0),
                "message": item["message"]
            })
        
        # If we found any matches for this deprecation pattern, mark as invalid
        if found_matches:
            results["valid"] = False
    
    # Check for best practice violations
    for item in ODOO_BEST_PRACTICES:
        matches = re.finditer(item["pattern"], code)
        for match in matches:
            results["best_practice_suggestions"].append({
                "line": code.count('\n', 0, match.start()) + 1,
                "pattern": match.group(0),
                "message": item["message"]
            })
        if results["deprecation_warnings"] or results["best_practice_suggestions"]:
            results["valid"] = False
    
    return results


async def _generate_odoo_snippet(feature: str, module: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a minimal working example snippet from Odoo 18 core code.
    
    This tool searches for relevant code in the Odoo 18 repository and generates
    a minimal working example that demonstrates the requested feature.
    
    Args:
        feature: The Odoo feature to generate a snippet for (e.g., 'create invoice', 'define model')
        module: Optional specific Odoo module to search within (e.g., 'sale', 'account')
        
    Returns:
        Dictionary with the generated snippet and reference information
    """
    # Handle both direct calls and ainvoke calls
    if isinstance(feature, dict):
        params = feature
        feature = params.get("feature")
        module = params.get("module")
    
    # First search for relevant code
    search_results = await _search_odoo_code(feature, module)
    
    if "error" in search_results or len(search_results.get("results", [])) == 0:
        return {
            "error": "Could not find relevant code examples",
            "feature": feature,
            "module": module
        }
    
    # Get the most relevant result
    top_result = search_results["results"][0]
    file_path = top_result.get("path", "")
    
    # Load the file content
    file_content = await _load_odoo_code(file_path)
    
    if "error" in file_content:
        return {
            "error": "Could not load file content",
            "feature": feature,
            "file_path": file_path
        }
    
    # Generate a minimal working example
    snippet = {
        "code": "# Minimal working example based on Odoo 18 code\n",
        "imports": [],
        "models": [],
        "methods": [],
        "reference": {
            "file_path": file_path,
            "url": f"https://github.com/{ODOO_REPO}/blob/{ODOO_BRANCH}/{file_path}"
        }
    }
    
    # Extract imports, models, and methods from the file content
    if "content" in file_content:
        code = file_content["content"]
        
        # Extract imports
        import_pattern = r'^\s*(?:from|import)\s+.*$'
        imports = re.findall(import_pattern, code, re.MULTILINE)
        if imports:
            snippet["imports"] = imports
            snippet["code"] += "\n# Imports\n" + "\n".join(imports) + "\n"
        
        # Extract model classes
        model_pattern = r'^\s*class\s+(\w+)\s*\(\s*(?:models\.\w+|osv\.\w+)\s*\)\s*:.*?(?=^\s*class|\Z)'  
        models = re.findall(model_pattern, code, re.MULTILINE | re.DOTALL)
        
        if models:
            snippet["models"] = models
            
            # For each model, extract its definition
            for model in models:
                model_def_pattern = fr'^\s*class\s+{model}\s*\(.*?\)\s*:.*?(?=^\s*class|\Z)'
                model_def = re.search(model_def_pattern, code, re.MULTILINE | re.DOTALL)
                if model_def:
                    snippet["code"] += f"\n# Model: {model}\n{model_def.group(0)}\n"
        
        # Extract methods related to the feature
        method_pattern = r'^\s*def\s+(\w+)\s*\(.*?\)\s*:.*?(?=^\s*def|^\s*class|\Z)'
        methods = re.findall(method_pattern, code, re.MULTILINE | re.DOTALL)
        
        if methods:
            snippet["methods"] = methods
            
            # For each method, extract its definition if it seems relevant to the feature
            relevant_methods = [m for m in methods if feature.lower() in m.lower()]
            for method in relevant_methods[:2]:  # Limit to 2 most relevant methods
                method_def_pattern = fr'^\s*def\s+{method}\s*\(.*?\)\s*:.*?(?=^\s*def|^\s*class|\Z)'
                method_def = re.search(method_def_pattern, code, re.MULTILINE | re.DOTALL)
                if method_def:
                    snippet["code"] += f"\n# Method: {method}\n{method_def.group(0)}\n"
    
    return {
        "feature": feature,
        "module": module,
        "snippet": snippet["code"],
        "reference": {
            "file_path": file_path,
            "repo": ODOO_REPO,
            "branch": ODOO_BRANCH
        }
    }


# Function to find all Odoo modules in local repository
def _get_local_odoo_modules() -> List[str]:
    """Get a list of all Odoo modules in the local repository.
    
    Returns:
        List of module names
    """
    if not LOCAL_REPO_EXISTS:
        return []
    
    try:
        modules_path = os.path.join(ODOO_LOCAL_REPO, "addons")
        if not os.path.isdir(modules_path):
            return []
        
        # Get all directories in the addons folder
        modules = [d for d in os.listdir(modules_path) 
                  if os.path.isdir(os.path.join(modules_path, d)) and 
                  not d.startswith(".")]  # Exclude hidden directories
        
        return modules
    except Exception as e:
        print(f"Error getting local Odoo modules: {str(e)}")
        return []

# Create the StructuredTool instances at module level
# For async tools, we need to use the coroutine parameter to support ainvoke
search_odoo_code = StructuredTool.from_function(
    coroutine=_search_odoo_code,  # Use combined search with vector DB and fallbacks
    name="search_odoo_code",
    description="Search for Odoo 18 code using vector DB, local repository, or GitHub"
)

load_odoo_code = StructuredTool.from_function(
    coroutine=_load_odoo_code,  # Use standard load function
    name="load_odoo_code",
    description="Load Odoo 18 code file from local repository or GitHub"
)

validate_odoo_code = StructuredTool.from_function(
    func=_validate_odoo_code,
    name="validate_odoo_code",
    description="Validate Odoo code against best practices and deprecation warnings"
)

generate_odoo_snippet = StructuredTool.from_function(
    coroutine=_generate_odoo_snippet,
    name="generate_odoo_snippet",
    description="Generate a minimal working example snippet from Odoo 18 core code"
)

# Add a tool to list available Odoo modules
list_odoo_modules = StructuredTool.from_function(
    func=_get_local_odoo_modules,
    name="list_odoo_modules",
    description="List all available Odoo 18 modules in the local repository"
)


# Vector search functions
async def _search_vector_db(query: str, module: Optional[str] = None, file_type: str = "py") -> Dict[str, Any]:
    """
    Search for Odoo code using Supabase vector database.
    
    Args:
        query: The search query
        module: Optional specific Odoo module to search within
        file_type: File extension to filter by
        
    Returns:
        Dictionary with search results
    """
    from .vector_store import get_embedding, SupabaseVectorStore
    
    try:
        # Get embedding for the query
        query_embedding = await get_embedding(query)
        
        # Initialize Supabase client
        vector_store = SupabaseVectorStore()
        
        # Prepare metadata filters
        filter_metadata = {}
        if module:
            filter_metadata["module"] = module
        if file_type:
            filter_metadata["file_type"] = file_type
        
        # Search for similar code
        results = await vector_store.search_similar(
            query_embedding=query_embedding,
            limit=10,
            similarity_threshold=0.6,
            filter_metadata=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "path": result.metadata.get("file_path", ""),
                "url": f"https://github.com/{ODOO_REPO}/blob/{ODOO_BRANCH}/{result.metadata.get('file_path', '')}",
                "content": result.content[:500] + ("..." if len(result.content) > 500 else ""),
                "similarity": result.similarity
            })
        
        return {
            "source": "vector_db",
            "query": query,
            "module": module,
            "total_count": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        return {
            "error": f"Vector search failed: {str(e)}",
            "query": query,
            "module": module,
            "results": []
        }

# Export the tools
ODOO_CODE_TOOLS = [
    search_odoo_code,
    load_odoo_code,
    validate_odoo_code,
    generate_odoo_snippet,
    list_odoo_modules
]
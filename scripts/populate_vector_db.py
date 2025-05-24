#!/usr/bin/env python
"""
Script to populate the Supabase vector database with Odoo code samples.

This script:
1. Scans an Odoo repository for Python files
2. Generates embeddings for each file
3. Stores the embeddings in Supabase vector database
"""

import os
import asyncio
import argparse
from typing import List, Dict, Any
import glob
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from react_agent.vector_store import SupabaseVectorStore, get_embedding, batch_get_embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def chunk_text(text: str, max_chunk_size: int = 8000) -> List[str]:
    """
    Split a large text into smaller chunks.
    
    Args:
        text: The text to split into chunks
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List[str]: List of text chunks
    """
    # If the text is small enough, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    # Try to split at logical boundaries (newlines)
    lines = text.split('\n')
    current_chunk = ""
    
    for line in lines:
        # If adding this line would exceed the max chunk size, start a new chunk
        if len(current_chunk) + len(line) + 1 > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

async def process_odoo_file(file_path: str, odoo_version: str) -> List[Dict[str, Any]]:
    """
    Process an Odoo file and extract content with metadata.
    Handles large files by splitting them into chunks.
    Supports multiple file types including Python, XML, JavaScript, and CSS.
    
    Args:
        file_path: Path to the Odoo file
        odoo_version: Odoo version of the file
        
    Returns:
        List[Dict[str, Any]]: List of file chunks with metadata
    """
    # Skip binary files and certain file types
    skip_extensions = [
        '.pyc', '.pyo', '.so', '.egg', '.git', '.svn', '.hg',
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.pdf', '.zip',
        '.tar', '.gz', '.rar', '.mo', '.o', '.a', '.dll', '.exe',
        '.odp', '.ods', '.odt', '.xls', '.xlsx', '.doc', '.docx', '.ppt', '.pptx'
    ]
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext in skip_extensions:
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError, IsADirectoryError):
        # Skip files that can't be decoded as UTF-8 or have other issues
        print(f"Skipping file with issues: {file_path}")
        return []
    
    # Extract module name from path
    path_parts = Path(file_path).parts
    module_index = next((i for i, part in enumerate(path_parts) if part in ['addons', 'odoo', 'openerp']), None)
    module = path_parts[module_index + 1] if module_index is not None else "unknown"
    
    # Determine file type based on extension and path
    file_type = "unknown"
    
    # Python files
    if file_ext == '.py':
        if "/models/" in file_path:
            file_type = "model"
        elif "/controllers/" in file_path:
            file_type = "controller"
        elif "/wizards/" in file_path:
            file_type = "wizard"
        elif "/tests/" in file_path:
            file_type = "test"
        elif "/__manifest__.py" in file_path or "/__openerp__.py" in file_path:
            file_type = "manifest"
        else:
            file_type = "python"
    
    # XML files
    elif file_ext == '.xml':
        if "/views/" in file_path:
            file_type = "view"
        elif "/data/" in file_path:
            file_type = "data"
        elif "/security/" in file_path:
            file_type = "security"
        elif "/report/" in file_path:
            file_type = "report"
        else:
            file_type = "xml"
    
    # JavaScript files
    elif file_ext in ['.js', '.mjs']:
        if "/static/src/js/" in file_path:
            file_type = "js"
        elif "/static/src/components/" in file_path:
            file_type = "component"
        elif "/static/tests/" in file_path:
            file_type = "js_test"
        else:
            file_type = "javascript"
    
    # CSS/SCSS files
    elif file_ext in ['.css', '.scss', '.sass', '.less']:
        file_type = "style"
    
    # QWeb templates
    elif file_ext == '.qweb':
        file_type = "qweb"
    
    # Other file types
    elif file_ext == '.csv':
        file_type = "csv"
    elif file_ext == '.sql':
        file_type = "sql"
    elif file_ext in ['.md', '.rst', '.txt']:
        file_type = "documentation"
    
    # Split content into chunks if it's too large
    chunks = chunk_text(content)
    
    # Create a result for each chunk
    results = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = {
            "file_path": file_path,
            "module": module,
            "odoo_version": odoo_version,
            "file_type": file_type,
            "file_extension": file_ext.lstrip('.'),
            "chunk": i + 1,
            "total_chunks": len(chunks)
        }
        
        results.append({
            "content": chunk,
            "metadata": chunk_metadata
        })
    
    return results


async def populate_db(odoo_path: str, odoo_version: str, limit: int = None, file_types: List[str] = None):
    """
    Populate the vector database with Odoo code samples, focusing on the most relevant code
    for the Odoo 18 Code Agent while skipping unnecessary parts.
    
    This function selectively processes Odoo code files based on their relevance to an
    Odoo development assistant. It prioritizes:
    
    1. Core Odoo server-side framework code (essential for understanding Odoo architecture)
    2. Web framework components (OWL, controllers, views, etc.)
    3. Common business modules (account, sale, purchase, etc.)
    4. Reusable tools and utilities
    
    While excluding:
    - Test files (except those demonstrating important patterns)
    - Localization modules (l10n_*) as they're too specific
    - Third-party integrations with limited use
    - Duplicate or generated code
    - Binary and non-code files
    
    Args:
        odoo_path: Path to the Odoo codebase
        odoo_version: Odoo version
        limit: Maximum number of files to process (for testing)
        file_types: List of file extensions to process (default: all relevant types)
    """
    vector_store = SupabaseVectorStore()
    
    # Define file types to process
    if not file_types:
        file_types = [
            "py", "xml", "js", "css", "scss", "qweb", "csv", "sql", 
            "md", "rst", "txt", "mjs", "sass", "less"
        ]
    
    # Find all relevant files
    all_files = []
    for file_type in file_types:
        files = glob.glob(f"{odoo_path}/**/*.{file_type}", recursive=True)
        all_files.extend(files)
    
    # Define patterns for files/directories to include or exclude
    include_patterns = [
        # Core Odoo framework - server side
        "/odoo/", "/openerp/", "/odoo/addons/base/", "/odoo/modules/", "/odoo/models/", "/odoo/fields.py",
        "/odoo/api.py", "/odoo/tools/", "/odoo/http.py", "/odoo/sql_db.py", "/odoo/exceptions.py",
        
        # Web framework (OWL, controllers, views)
        "/addons/web/", "/addons/web_editor/", "/addons/website/", "/addons/web/static/src/",
        "/addons/point_of_sale/static/src/", "/addons/web/static/lib/owl/",
        
        # Common business modules (high value for developers)
        "/addons/account/", "/addons/sale/", "/addons/purchase/", "/addons/stock/",
        "/addons/crm/", "/addons/hr/", "/addons/product/", "/addons/project/",
        
        # Important utilities and patterns
        "/addons/mail/", "/addons/base_automation/", "/addons/web/static/src/core/",
        "/addons/web/static/src/views/", "/addons/web/static/src/search/",
        
        # Documentation and examples
        "/__manifest__.py", "/models/", "/controllers/", "/wizards/", "/views/",
        "/static/description/", "/doc/", "/README"
    ]
    
    exclude_patterns = [
        # Tests (except important ones)
        "/tests/", "_test.py", "test_", "/test/",
        
        # Localization modules
        "/l10n_", "/account_tax_python/",
        
        # Third-party integrations with limited use
        "/payment_", "/delivery_", "/pos_", "/hw_", "/google_", "/microsoft_",
        
        # Generated or duplicate code
        "/i18n/", "/static/lib/", "/node_modules/", "/.git/", "/__pycache__/",
        "/venv/", "/.env/", "/.vscode/", "/dist/", "/build/",
        
        # Specific large modules with limited general value
        "/addons/website_sale/static/", "/addons/point_of_sale/static/lib/",
        "/addons/web/static/lib/", "/addons/web/static/tests/"
    ]
    
    # Apply include/exclude filters
    filtered_files = []
    for file_path in all_files:
        # Skip if file matches any exclude pattern
        if any(pattern in file_path for pattern in exclude_patterns):
            continue
        
        # Include if file matches any include pattern or is in a core module
        if any(pattern in file_path for pattern in include_patterns):
            filtered_files.append(file_path)
    
    # Apply limit if specified
    if limit:
        filtered_files = filtered_files[:limit]
    
    print(f"Found {len(all_files)} total files")
    print(f"Selected {len(filtered_files)} relevant files for processing")
    
    # Process files in batches
    batch_size = 10
    for i in range(0, len(filtered_files), batch_size):
        batch = filtered_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(filtered_files)-1)//batch_size + 1}")
        
        # Process files - each file may return multiple chunks
        file_chunks_list = await asyncio.gather(*[
            process_odoo_file(file_path, odoo_version) 
            for file_path in batch
        ])
        
        # Flatten the list of chunks
        all_chunks = [chunk for chunks in file_chunks_list for chunk in chunks]
        
        if not all_chunks:
            print("No valid chunks to process in this batch")
            continue
        
        # Process chunks in smaller batches to avoid payload size limits
        chunk_batch_size = 5
        total_stored = 0
        
        for j in range(0, len(all_chunks), chunk_batch_size):
            chunk_batch = all_chunks[j:j+chunk_batch_size]
            
            # Generate embeddings
            contents = [chunk["content"] for chunk in chunk_batch]
            embeddings = await batch_get_embeddings(contents)
            
            # Store in database
            for chunk_data, embedding in zip(chunk_batch, embeddings):
                await vector_store.store_embedding(
                    content=chunk_data["content"],
                    embedding=embedding,
                    metadata=chunk_data["metadata"]
                )
                total_stored += 1
        
        print(f"Stored {total_stored} embeddings in the database")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate vector database with Odoo code samples")
    parser.add_argument("--odoo-path", type=str, required=True, help="Path to Odoo codebase")
    parser.add_argument("--odoo-version", type=str, required=True, help="Odoo version")
    parser.add_argument("--limit", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument(
        "--file-types", 
        type=str, 
        nargs="+", 
        help="File types to process (e.g., py xml js css). Default is all relevant types."
    )
    args = parser.parse_args()
    
    asyncio.run(populate_db(
        odoo_path=args.odoo_path, 
        odoo_version=args.odoo_version, 
        limit=args.limit,
        file_types=args.file_types
    ))

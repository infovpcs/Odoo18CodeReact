import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from react_agent.odoo_code_utils import search_odoo_code, load_odoo_code, validate_odoo_code, generate_odoo_snippet

@pytest.mark.asyncio
async def test_search_odoo_code_success():
    with patch("react_agent.odoo_code_utils._github_request", new=AsyncMock(return_value={"total_count": 1, "items": [{"path": "addons/base/models/res_users.py"}]})):
        from react_agent.odoo_code_utils import _search_odoo_code
        result = await _search_odoo_code("class User", "base")
        assert result["source"] == "github"
        assert result["total_count"] == 1
        assert any("res_users.py" in item["path"] for item in result["results"])

@pytest.mark.asyncio
async def test_search_odoo_code_fallback():
    with patch("react_agent.odoo_code_utils._github_request", new=AsyncMock(return_value={"error": "GitHub API error"})), \
         patch("react_agent.odoo_code_utils._huggingface_fallback", new=AsyncMock(return_value={"results": [{"path": "addons/base/models/res_partner.py"}]})):
        from react_agent.odoo_code_utils import _search_odoo_code
        result = await _search_odoo_code("class Partner", "base")
        assert result["source"] == "huggingface_fallback"
        assert any("res_partner.py" in item["path"] for item in result["results"])

@pytest.mark.asyncio
async def test_load_odoo_code_success():
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="class User(models.Model):\n    _name = 'res.users'")
        mock_get.return_value.__aenter__.return_value = mock_response
        from react_agent.odoo_code_utils import _load_odoo_code
        result = await _load_odoo_code("addons/base/models/res_users.py")
        assert result["file_path"] == "addons/base/models/res_users.py"
        assert "class User" in result["content"]

@pytest.mark.asyncio
async def test_load_odoo_code_failure():
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_get.return_value.__aenter__.return_value = mock_response
        from react_agent.odoo_code_utils import _load_odoo_code
        result = await _load_odoo_code("addons/base/models/does_not_exist.py")
        assert result["error"].startswith("Failed to load file")
        assert result["file_path"] == "addons/base/models/does_not_exist.py"

@pytest.mark.asyncio
async def test_generate_odoo_snippet_success():
    with patch("react_agent.odoo_code_utils._search_odoo_code", new=AsyncMock(return_value={"results": [{"path": "addons/base/models/res_users.py"}]})), \
         patch("react_agent.odoo_code_utils._load_odoo_code", new=AsyncMock(return_value={"content": "from odoo import models, fields\n\nclass User(models.Model):\n    _name = 'res.users'\n    name = fields.Char()"})):
        from react_agent.odoo_code_utils import _generate_odoo_snippet
        result = await _generate_odoo_snippet("user model")
        assert "from odoo import models, fields" in result["snippet"]
        assert "class User(models.Model):" in result["snippet"]
        assert "_name = 'res.users'" in result["snippet"]

@pytest.mark.asyncio
async def test_generate_odoo_snippet_failure():
    with patch("react_agent.odoo_code_utils._search_odoo_code", new=AsyncMock(return_value={"error": "Search failed"})):
        from react_agent.odoo_code_utils import _generate_odoo_snippet
        result = await _generate_odoo_snippet("nonexistent feature", "nonexistent")
        assert result["error"] == "Could not find relevant code examples"
        assert result["feature"] == "nonexistent feature"
        assert result["module"] == "nonexistent"

@pytest.mark.parametrize("code,expected_valid,deprecation_count,best_practice_count", [
    ("from odoo import models\nclass Partner(models.Model):\n    _name = 'res.partner'", True, 0, 0),
    ("from openerp import models\nclass Partner(models.Model):\n    _name = 'res.partner'", False, 2, 0),  # Expecting 2 deprecation warnings: 'from openerp' and 'openerp' keyword
    ("from odoo import models\nclass Partner(models.Model):\n    _name = 'res.partner'\n    _auto = False", False, 0, 1),
])
def test_validate_odoo_code(code, expected_valid, deprecation_count, best_practice_count):
    from react_agent.odoo_code_utils import _validate_odoo_code
    result = _validate_odoo_code(code)
    assert result["valid"] == expected_valid
    assert len(result["deprecation_warnings"]) == deprecation_count, \
        f"Expected {deprecation_count} deprecation warnings, got {len(result['deprecation_warnings'])}: {result['deprecation_warnings']}"
    assert len(result["best_practice_suggestions"]) == best_practice_count, \
        f"Expected {best_practice_count} best practice suggestions, got {len(result['best_practice_suggestions'])}: {result['best_practice_suggestions']}"
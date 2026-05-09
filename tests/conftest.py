"""Shared pytest fixtures for Agent Saan tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from agent_saan.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP test client wired to the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

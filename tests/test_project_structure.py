"""Tests verifying the project structure and tooling are correctly set up (Task 1)."""

import importlib
from pathlib import Path

import pytest


# ── Package layout ────────────────────────────────────────────────────────────

EXPECTED_PACKAGES = [
    "agent_saan",
    "agent_saan.orchestrator",
    "agent_saan.nlu",
    "agent_saan.memory",
    "agent_saan.tasks",
    "agent_saan.suggestions",
    "agent_saan.plugins",
    "agent_saan.safety",
    "agent_saan.personality",
    "agent_saan.api",
    "agent_saan.events",
    "agent_saan.models",
]


@pytest.mark.parametrize("package", EXPECTED_PACKAGES)
def test_package_importable(package: str) -> None:
    """Every sub-package must be importable without errors."""
    mod = importlib.import_module(package)
    assert mod is not None


def test_src_layout_exists() -> None:
    """The src/agent_saan directory must exist."""
    src = Path(__file__).parent.parent / "src" / "agent_saan"
    assert src.is_dir(), f"Expected src/agent_saan directory at {src}"


@pytest.mark.parametrize(
    "subdir",
    [
        "orchestrator",
        "nlu",
        "memory",
        "tasks",
        "suggestions",
        "plugins",
        "safety",
        "personality",
        "api",
        "events",
        "models",
    ],
)
def test_subpackage_init_exists(subdir: str) -> None:
    """Each sub-package directory must contain an __init__.py."""
    init_file = Path(__file__).parent.parent / "src" / "agent_saan" / subdir / "__init__.py"
    assert init_file.is_file(), f"Missing __init__.py in src/agent_saan/{subdir}/"


# ── Configuration ─────────────────────────────────────────────────────────────

def test_env_example_exists() -> None:
    """.env.example must exist at the project root."""
    env_example = Path(__file__).parent.parent / ".env.example"
    assert env_example.is_file(), ".env.example not found at project root"


@pytest.mark.parametrize(
    "var",
    [
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "JWT_SECRET_KEY",
        "ACTION_RATE_LIMIT_DEFAULT",
        "ACTION_RATE_LIMIT_MAX",
        "STM_MAX_TURNS",
        "STM_TTL_MINUTES",
        "PLUGIN_DEFAULT_TIMEOUT_SECONDS",
        "LOG_LEVEL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ],
)
def test_env_example_contains_variable(var: str) -> None:
    """Every required environment variable must appear in .env.example."""
    env_example = Path(__file__).parent.parent / ".env.example"
    content = env_example.read_text()
    assert var in content, f"{var} not found in .env.example"


# ── FastAPI application ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client) -> None:
    """GET /health must return 200 with status=ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_openapi_spec_served(client) -> None:
    """OpenAPI spec must be served at /api/v1/openapi.json."""
    response = await client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert "openapi" in spec
    assert spec["info"]["title"] == "Agent Saan"


# ── pyproject.toml ────────────────────────────────────────────────────────────

def test_pyproject_toml_exists() -> None:
    """pyproject.toml must exist at the project root."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.is_file()


def test_pyproject_requires_python_312() -> None:
    """pyproject.toml must require Python >=3.12."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert "3.12" in content

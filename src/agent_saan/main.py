"""FastAPI application factory for Agent Saan."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_saan import __version__
from agent_saan.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = FastAPI(
        title="Agent Saan",
        description="Personal AI assistant inspired by Iron Man's Friday.",
        version=__version__,
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        openapi_url="/api/v1/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    logger.info("Agent Saan v%s application created (env=%s)", __version__, settings.app_env)
    return app


app = create_app()

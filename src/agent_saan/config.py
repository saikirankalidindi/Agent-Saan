"""Application configuration loaded from environment variables via Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for Agent Saan.

    Values are read from environment variables (case-insensitive) or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM / AI API Keys ────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    elevenlabs_api_key: str = Field("", description="ElevenLabs TTS API key")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        "postgresql+asyncpg://agent_saan:changeme@localhost:5432/agent_saan",
        description="Async PostgreSQL connection URL",
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")

    # ── Authentication ────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(..., description="Secret key for signing JWT tokens")
    jwt_algorithm: str = Field("HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(60, description="JWT token TTL in minutes")

    # ── Safety / Rate Limiting ────────────────────────────────────────────────
    action_rate_limit_default: int = Field(
        100, ge=1, description="Default autonomous actions per hour"
    )
    action_rate_limit_max: int = Field(
        1000, ge=1, description="Maximum user-configurable actions per hour"
    )

    # ── Short-Term Memory ─────────────────────────────────────────────────────
    stm_max_turns: int = Field(50, ge=1, description="Maximum conversation turns in STM")
    stm_ttl_minutes: int = Field(30, ge=1, description="STM inactivity TTL in minutes")

    # ── Plugin System ─────────────────────────────────────────────────────────
    plugin_default_timeout_seconds: int = Field(
        10, ge=1, le=60, description="Default plugin action timeout in seconds"
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)")

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = Field(
        "http://localhost:4317", description="OTLP gRPC endpoint for traces"
    )
    otel_service_name: str = Field("agent-saan", description="Service name for OTel traces")

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field("development", description="Deployment environment")
    app_host: str = Field("0.0.0.0", description="Uvicorn bind host")
    app_port: int = Field(8000, ge=1, le=65535, description="Uvicorn bind port")
    app_reload: bool = Field(False, description="Enable uvicorn hot-reload (dev only)")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton)."""
    return Settings()  # type: ignore[call-arg]

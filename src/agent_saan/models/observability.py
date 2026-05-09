"""Observability and telemetry models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ObservabilityEvent(BaseModel):
    """A structured telemetry event emitted for every action dispatched."""

    action_id: str
    subsystem: str
    dispatch_timestamp: datetime
    completion_timestamp: datetime
    latency_ms: int = Field(ge=0)
    outcome: Literal["success", "error"]
    error_code: str | None = None

"""Event bus models for inter-subsystem communication."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BusEvent(BaseModel):
    """A structured event published to the Redis Pub/Sub event bus."""

    event_id: str  # UUID
    event_type: str  # e.g. "nlu.result", "safety.block", "task.created"
    source: str  # subsystem name
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime

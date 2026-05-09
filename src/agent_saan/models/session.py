"""Session and message models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Session(BaseModel):
    """Represents an active interaction session between the user and Agent Saan."""

    session_id: str
    user_id: str
    state: Literal["idle", "listening", "processing", "acting", "responding"]
    created_at: datetime
    last_active: datetime
    # Using Any to avoid circular import; callers should use UserInput from nlu module
    input_queue: list[Any] = Field(default_factory=list)


class Message(BaseModel):
    """A single message exchanged in a session."""

    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime

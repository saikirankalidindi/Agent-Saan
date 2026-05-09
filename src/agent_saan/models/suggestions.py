"""Suggestion engine models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Suggestion(BaseModel):
    """A proactive suggestion surfaced by the Suggestion Engine."""

    suggestion_id: str
    session_id: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=100)
    category: str
    topic: str
    created_at: datetime

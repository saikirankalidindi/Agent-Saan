"""NLU (Natural Language Understanding) models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Raw input submitted by the user in any modality."""

    input_id: str
    session_id: str
    modality: Literal["text", "audio", "image"]
    content: str | bytes
    timestamp: datetime


class Intent(BaseModel):
    """A parsed intent extracted from user input."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A named entity extracted from user input."""

    type: str
    value: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class NLUResult(BaseModel):
    """Structured output from the NLU Engine after parsing user input."""

    input_id: str
    intents: list[Intent]  # ranked by confidence descending
    entities: list[Entity] = Field(default_factory=list)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    language: str  # ISO 639-1 code
    is_ambiguous: bool

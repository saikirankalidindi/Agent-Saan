"""Memory models for short-term and long-term memory."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent_saan.models.nlu import Entity


class MemoryEntry(BaseModel):
    """A single entry in the long-term memory store."""

    entry_id: str
    user_id: str
    namespace: Literal["domain_facts", "user_preferences", "learned_patterns"]
    content: str
    source_type: Literal["imported", "user_stated", "feedback_derived"]
    embedding: list[float] = Field(default_factory=list)  # pgvector representation
    created_at: datetime
    tags: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    """A single turn in a conversation, stored in short-term memory."""

    turn_index: int = Field(ge=0)
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    entities: list[Entity] = Field(default_factory=list)

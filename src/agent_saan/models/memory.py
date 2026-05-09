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


class ConflictResult(BaseModel):
    """Result of a conflict detection check against long-term memory.

    A conflict is raised when a new statement is semantically similar
    (cosine similarity > 0.85) to an existing LTM entry AND the content
    of the two statements contradicts each other.
    """

    has_conflict: bool
    """True when a conflicting entry was found."""

    existing_entry: MemoryEntry | None = None
    """The existing LTM entry that conflicts with the new statement."""

    new_statement: str = ""
    """The new statement that triggered the conflict check."""

    similarity_score: float = 0.0
    """Cosine similarity between the new statement and the existing entry (0–1)."""


class ConversationTurn(BaseModel):
    """A single turn in a conversation, stored in short-term memory."""

    turn_index: int = Field(ge=0)
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    entities: list[Entity] = Field(default_factory=list)


class CorefResult(BaseModel):
    """Result of a co-reference resolution attempt.

    When the Memory_Store resolves pronouns and entity references against the
    Short_Term_Memory conversation history, it returns a ``CorefResult``.

    If the resolution confidence is below 0.7, ``resolved`` is ``False`` and
    ``clarification_needed`` is ``True`` — the Orchestrator must ask the user
    to clarify the reference before proceeding (Requirement 2.5).
    """

    resolved: bool
    """True when the reference was resolved with confidence >= 0.7."""

    resolved_text: str = ""
    """The input text with pronouns/references replaced by their referents.

    Only meaningful when ``resolved`` is ``True``.
    """

    confidence: float = 0.0
    """Confidence score of the resolution (0.0–1.0)."""

    clarification_needed: bool = False
    """True when the reference could not be resolved and the user must clarify."""

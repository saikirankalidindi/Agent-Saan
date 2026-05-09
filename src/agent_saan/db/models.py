"""SQLAlchemy ORM models for Agent Saan.

Each class maps to a PostgreSQL table defined in the Alembic migration.
All primary keys are UUIDs stored as TEXT for portability.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agent_saan.db.base import Base


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
class SessionORM(Base):
    """Active and historical user sessions."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    state: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="idle",
        comment="One of: idle, listening, processing, acting, responding",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    # Serialised list[UserInput] — stored as JSONB for flexibility
    input_queue: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True, default=list)

    # Relationships
    messages: Mapped[list[MessageORM]] = relationship(
        "MessageORM", back_populates="session", cascade="all, delete-orphan"
    )
    suggestions: Mapped[list[SuggestionORM]] = relationship(
        "SuggestionORM", back_populates="session", cascade="all, delete-orphan"
    )
    audit_entries: Mapped[list[AuditLogORM]] = relationship(
        "AuditLogORM", back_populates="session"
    )
    feedback_entries: Mapped[list[FeedbackORM]] = relationship(
        "FeedbackORM", back_populates="session"
    )


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------
class MessageORM(Base):
    """Individual messages within a session (user and assistant turns)."""

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="One of: user, assistant"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationship
    session: Mapped[SessionORM] = relationship("SessionORM", back_populates="messages")


# ---------------------------------------------------------------------------
# memory_entries
# ---------------------------------------------------------------------------
class MemoryEntryORM(Base):
    """Long-term memory entries with pgvector embeddings for semantic search."""

    __tablename__ = "memory_entries"

    entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    namespace: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        comment="One of: domain_facts, user_preferences, learned_patterns",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        comment="One of: imported, user_stated, feedback_derived",
    )
    # 1536-dimensional embedding from text-embedding-3-small
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Serialised list[str] tags
    tags: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True, default=list)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        comment="One of: active, conflict, deleted",
    )


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
class TaskORM(Base):
    """Task records managed by the Task Manager."""

    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="medium",
        comment="One of: high, medium, low",
    )
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        comment="One of: pending, in_progress, completed, pending_authorization, dismissed",
    )
    completion_source: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="One of: user, agent"
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    recurrence: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Cron expression or daily/weekly/monthly"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# ---------------------------------------------------------------------------
# suggestions
# ---------------------------------------------------------------------------
class SuggestionORM(Base):
    """Proactive suggestions generated by the Suggestion Engine."""

    __tablename__ = "suggestions"

    suggestion_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    rationale: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    topic: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationship
    session: Mapped[SessionORM] = relationship("SessionORM", back_populates="suggestions")


# ---------------------------------------------------------------------------
# audit_log
# ---------------------------------------------------------------------------
class AuditLogORM(Base):
    """Immutable audit log — application role has INSERT privilege only."""

    __tablename__ = "audit_log"

    log_id: Mapped[str] = mapped_column(String, primary_key=True)
    action_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("sessions.session_id", ondelete="SET NULL"), nullable=True, index=True
    )
    outcome: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="One of: allowed, blocked, cancelled"
    )
    guardrail_rule: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationship
    session: Mapped[SessionORM | None] = relationship(
        "SessionORM", back_populates="audit_entries"
    )


# ---------------------------------------------------------------------------
# user_preferences
# ---------------------------------------------------------------------------
class UserPreferencesORM(Base):
    """Per-user configuration and personalisation settings."""

    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    communication_style: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="casual",
        comment="One of: formal, casual, technical",
    )
    verbosity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="standard",
        comment="One of: concise, standard, detailed",
    )
    voice_output_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    tts_voice_gender: Mapped[str] = mapped_column(
        String(10), nullable=False, default="female", comment="One of: male, female"
    )
    tts_speech_rate_wpm: Mapped[int] = mapped_column(
        Integer, nullable=False, default=150, comment="80–200 words per minute"
    )
    tts_pitch: Mapped[str] = mapped_column(
        String(10), nullable=False, default="medium", comment="One of: low, medium, high"
    )
    action_rate_limit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=100, comment="100–1000 actions per hour"
    )
    safe_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # dict[str, float] — intent category → confidence weight multiplier
    category_confidence_weights: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, default=dict
    )


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------
class FeedbackORM(Base):
    """User feedback on agent responses (thumbs-up/down/correction)."""

    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("sessions.session_id", ondelete="SET NULL"), nullable=True, index=True
    )
    response_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    feedback_type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="One of: thumbs_up, thumbs_down, correction"
    )
    correction_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationship
    session: Mapped[SessionORM | None] = relationship(
        "SessionORM", back_populates="feedback_entries"
    )


# ---------------------------------------------------------------------------
# knowledge_base_imports
# ---------------------------------------------------------------------------
class KnowledgeBaseImportORM(Base):
    """Tracks document import jobs for the knowledge base."""

    __tablename__ = "knowledge_base_imports"

    import_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    format: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="One of: pdf, markdown, txt"
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="One of: pending, processing, completed, failed",
    )
    chunks_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

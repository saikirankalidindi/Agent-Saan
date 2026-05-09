"""Initial schema — all Agent Saan tables.

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

Creates:
  - pgvector extension
  - sessions
  - messages
  - memory_entries  (with vector(1536) embedding column + HNSW index)
  - tasks
  - suggestions
  - audit_log
  - user_preferences
  - feedback
  - knowledge_base_imports
  - audit_log_writer role (INSERT-only on audit_log)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Enable pgvector extension
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # 2. sessions
    # ------------------------------------------------------------------
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "state",
            sa.String(length=20),
            nullable=False,
            server_default="idle",
            comment="One of: idle, listening, processing, acting, responding",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_active",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("input_queue", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    # ------------------------------------------------------------------
    # 3. messages
    # ------------------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
            comment="One of: user, assistant",
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    # ------------------------------------------------------------------
    # 4. memory_entries  (pgvector column)
    # ------------------------------------------------------------------
    op.create_table(
        "memory_entries",
        sa.Column("entry_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "namespace",
            sa.String(length=30),
            nullable=False,
            comment="One of: domain_facts, user_preferences, learned_patterns",
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "source_type",
            sa.String(length=30),
            nullable=False,
            comment="One of: imported, user_stated, feedback_derived",
        ),
        # 1536-dimensional vector for text-embedding-3-small
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("tags", JSONB, nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="active",
            comment="One of: active, conflict, deleted",
        ),
        sa.PrimaryKeyConstraint("entry_id"),
    )
    op.create_index("ix_memory_entries_user_id", "memory_entries", ["user_id"])

    # HNSW index for cosine similarity search (pgvector)
    op.execute(
        "CREATE INDEX ix_memory_entries_embedding_hnsw "
        "ON memory_entries USING hnsw (embedding vector_cosine_ops)"
    )

    # ------------------------------------------------------------------
    # 5. tasks
    # ------------------------------------------------------------------
    op.create_table(
        "tasks",
        sa.Column("task_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "priority",
            sa.String(length=10),
            nullable=False,
            server_default="medium",
            comment="One of: high, medium, low",
        ),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
            server_default="pending",
            comment="One of: pending, in_progress, completed, pending_authorization, dismissed",
        ),
        sa.Column(
            "completion_source",
            sa.String(length=10),
            nullable=True,
            comment="One of: user, agent",
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "recurrence",
            sa.String(length=100),
            nullable=True,
            comment="Cron expression or daily/weekly/monthly",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index("ix_tasks_user_id", "tasks", ["user_id"])

    # ------------------------------------------------------------------
    # 6. suggestions
    # ------------------------------------------------------------------
    op.create_table(
        "suggestions",
        sa.Column("suggestion_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("topic", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("suggestion_id"),
    )
    op.create_index("ix_suggestions_session_id", "suggestions", ["session_id"])

    # ------------------------------------------------------------------
    # 7. audit_log  (append-only — no UPDATE/DELETE ever issued)
    # ------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("log_id", sa.String(), nullable=False),
        sa.Column("action_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column(
            "outcome",
            sa.String(length=20),
            nullable=False,
            comment="One of: allowed, blocked, cancelled",
        ),
        sa.Column("guardrail_rule", sa.String(length=255), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("log_id"),
    )
    op.create_index("ix_audit_log_action_id", "audit_log", ["action_id"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_session_id", "audit_log", ["session_id"])

    # ------------------------------------------------------------------
    # 8. user_preferences
    # ------------------------------------------------------------------
    op.create_table(
        "user_preferences",
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "communication_style",
            sa.String(length=20),
            nullable=False,
            server_default="casual",
            comment="One of: formal, casual, technical",
        ),
        sa.Column(
            "verbosity",
            sa.String(length=20),
            nullable=False,
            server_default="standard",
            comment="One of: concise, standard, detailed",
        ),
        sa.Column(
            "voice_output_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "tts_voice_gender",
            sa.String(length=10),
            nullable=False,
            server_default="female",
            comment="One of: male, female",
        ),
        sa.Column(
            "tts_speech_rate_wpm",
            sa.Integer(),
            nullable=False,
            server_default="150",
            comment="80–200 words per minute",
        ),
        sa.Column(
            "tts_pitch",
            sa.String(length=10),
            nullable=False,
            server_default="medium",
            comment="One of: low, medium, high",
        ),
        sa.Column(
            "action_rate_limit",
            sa.Integer(),
            nullable=False,
            server_default="100",
            comment="100–1000 actions per hour",
        ),
        sa.Column(
            "safe_mode",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("category_confidence_weights", JSONB, nullable=True),
        sa.PrimaryKeyConstraint("user_id"),
    )

    # ------------------------------------------------------------------
    # 9. feedback
    # ------------------------------------------------------------------
    op.create_table(
        "feedback",
        sa.Column("feedback_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("response_id", sa.String(), nullable=False),
        sa.Column(
            "feedback_type",
            sa.String(length=20),
            nullable=False,
            comment="One of: thumbs_up, thumbs_down, correction",
        ),
        sa.Column("correction_text", sa.Text(), nullable=True),
        sa.Column("intent_category", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"])
    op.create_index("ix_feedback_response_id", "feedback", ["response_id"])

    # ------------------------------------------------------------------
    # 10. knowledge_base_imports
    # ------------------------------------------------------------------
    op.create_table(
        "knowledge_base_imports",
        sa.Column("import_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(length=500), nullable=False),
        sa.Column(
            "format",
            sa.String(length=10),
            nullable=False,
            comment="One of: pdf, markdown, txt",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
            comment="One of: pending, processing, completed, failed",
        ),
        sa.Column(
            "chunks_indexed",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "failed_pages",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("import_id"),
    )
    op.create_index("ix_knowledge_base_imports_user_id", "knowledge_base_imports", ["user_id"])

    # ------------------------------------------------------------------
    # 11. INSERT-only role for audit_log
    #     (idempotent — skips creation if the role already exists)
    # ------------------------------------------------------------------
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'audit_log_writer') THEN
                CREATE ROLE audit_log_writer;
            END IF;
        END
        $$;
        """
    )
    op.execute("GRANT INSERT ON audit_log TO audit_log_writer")


def downgrade() -> None:
    # Revoke privileges and drop role
    op.execute("REVOKE INSERT ON audit_log FROM audit_log_writer")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'audit_log_writer') THEN
                DROP ROLE audit_log_writer;
            END IF;
        END
        $$;
        """
    )

    # Drop tables in reverse dependency order
    op.drop_table("knowledge_base_imports")
    op.drop_table("feedback")
    op.drop_table("user_preferences")
    op.drop_table("audit_log")
    op.drop_table("suggestions")
    op.drop_table("tasks")
    op.drop_table("memory_entries")
    op.drop_table("messages")
    op.drop_table("sessions")

    # Drop pgvector extension last
    op.execute("DROP EXTENSION IF EXISTS vector")

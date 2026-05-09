"""Integration tests for LongTermMemory (Requirements 2.2, 2.3, 2.4, 2.6).

All tests mock the asyncpg pool and the OpenAI embedding client so that no
live database or API calls are required.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_saan.memory.long_term import LongTermMemory, _CONFLICT_SIMILARITY_THRESHOLD
from agent_saan.models.memory import ConflictResult, MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1] * 1536  # 1536-dim vector of 0.1


def _make_entry(
    entry_id: str = "entry-1",
    user_id: str = "user-1",
    content: str = "I prefer dark mode",
    namespace: str = "user_preferences",
    source_type: str = "user_stated",
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Create a minimal MemoryEntry for testing."""
    return MemoryEntry(
        entry_id=entry_id,
        user_id=user_id,
        namespace=namespace,  # type: ignore[arg-type]
        content=content,
        source_type=source_type,  # type: ignore[arg-type]
        embedding=[],
        created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        tags=tags or [],
    )


def _make_db_row(
    entry_id: str = "entry-1",
    user_id: str = "user-1",
    content: str = "I prefer dark mode",
    namespace: str = "user_preferences",
    source_type: str = "user_stated",
    embedding: list[float] | None = None,
    tags: list[str] | None = None,
    similarity: float | None = None,
) -> dict[str, Any]:
    """Build a dict that mimics an asyncpg Record for memory_entries."""
    row: dict[str, Any] = {
        "entry_id": entry_id,
        "user_id": user_id,
        "namespace": namespace,
        "content": content,
        "source_type": source_type,
        "embedding": json.dumps(embedding or _FAKE_EMBEDDING),
        "created_at": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        "tags": json.dumps(tags or []),
    }
    if similarity is not None:
        row["similarity"] = similarity
    return row


def _make_ltm(pool: Any, openai_client: Any) -> LongTermMemory:
    """Construct a LongTermMemory with the given mocks."""
    return LongTermMemory(pool=pool, openai_client=openai_client)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai() -> AsyncMock:
    """Return a mock AsyncOpenAI client that always returns _FAKE_EMBEDDING."""
    client = AsyncMock()
    embedding_data = MagicMock()
    embedding_data.embedding = _FAKE_EMBEDDING
    client.embeddings.create.return_value = MagicMock(data=[embedding_data])
    return client


@pytest.fixture
def mock_conn() -> AsyncMock:
    """Return a mock asyncpg connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="DELETE 0")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    return conn


@pytest.fixture
def mock_pool(mock_conn: AsyncMock) -> MagicMock:
    """Return a mock asyncpg pool whose acquire() yields mock_conn."""
    pool = MagicMock()
    # acquire() is used as an async context manager
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


@pytest.fixture
def ltm(mock_pool: MagicMock, mock_openai: AsyncMock) -> LongTermMemory:
    """Return a LongTermMemory instance backed by mocks."""
    return _make_ltm(mock_pool, mock_openai)


# ---------------------------------------------------------------------------
# Tests: upsert  (Requirement 2.2)
# ---------------------------------------------------------------------------


class TestUpsert:
    """Tests for LongTermMemory.upsert — Requirement 2.2."""

    async def test_upsert_calls_embed_with_content(
        self, ltm: LongTermMemory, mock_openai: AsyncMock
    ) -> None:
        """upsert must embed the entry's content string."""
        entry = _make_entry(content="I prefer dark mode")
        await ltm.upsert(entry)

        mock_openai.embeddings.create.assert_awaited_once()
        call_kwargs = mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "I prefer dark mode"
        assert call_kwargs.kwargs["model"] == "text-embedding-3-small"

    async def test_upsert_executes_insert_on_conflict(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """upsert must issue an INSERT … ON CONFLICT DO UPDATE statement."""
        entry = _make_entry()
        await ltm.upsert(entry)

        mock_conn.execute.assert_awaited_once()
        sql: str = mock_conn.execute.call_args.args[0]
        assert "INSERT INTO memory_entries" in sql
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql

    async def test_upsert_passes_correct_entry_id(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """The entry_id must be passed as the first positional parameter."""
        entry = _make_entry(entry_id="my-entry-id")
        await ltm.upsert(entry)

        args = mock_conn.execute.call_args.args
        # args[0] is the SQL, args[1] is entry_id
        assert args[1] == "my-entry-id"

    async def test_upsert_passes_correct_user_id(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """The user_id must be passed as the second positional parameter."""
        entry = _make_entry(user_id="user-xyz")
        await ltm.upsert(entry)

        args = mock_conn.execute.call_args.args
        assert args[2] == "user-xyz"

    async def test_upsert_serialises_embedding_as_vector_literal(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """The embedding must be serialised as a pgvector literal '[x,y,...]'."""
        entry = _make_entry()
        await ltm.upsert(entry)

        args = mock_conn.execute.call_args.args
        # The embedding literal is the 6th positional arg (index 5, after SQL)
        embedding_arg: str = args[6]
        assert embedding_arg.startswith("[")
        assert embedding_arg.endswith("]")
        # Should contain 1536 comma-separated floats
        values = embedding_arg[1:-1].split(",")
        assert len(values) == 1536

    async def test_upsert_with_tags(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """Tags must be serialised as JSON and passed to the query."""
        entry = _make_entry(tags=["preference", "ui"])
        await ltm.upsert(entry)

        args = mock_conn.execute.call_args.args
        tags_arg: str = args[8]
        assert json.loads(tags_arg) == ["preference", "ui"]


# ---------------------------------------------------------------------------
# Tests: retrieve  (Requirement 2.3)
# ---------------------------------------------------------------------------


class TestRetrieve:
    """Tests for LongTermMemory.retrieve — Requirement 2.3."""

    async def test_retrieve_returns_empty_list_when_no_rows(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """retrieve must return [] when the DB returns no rows."""
        mock_conn.fetch.return_value = []
        results = await ltm.retrieve("dark mode", "user-1")
        assert results == []

    async def test_retrieve_returns_memory_entries(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """retrieve must convert DB rows to MemoryEntry objects."""
        mock_conn.fetch.return_value = [_make_db_row()]
        results = await ltm.retrieve("dark mode", "user-1")

        assert len(results) == 1
        assert isinstance(results[0], MemoryEntry)
        assert results[0].entry_id == "entry-1"
        assert results[0].content == "I prefer dark mode"

    async def test_retrieve_passes_user_id_to_query(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """retrieve must filter by user_id in the SQL query."""
        mock_conn.fetch.return_value = []
        await ltm.retrieve("query text", "user-42")

        args = mock_conn.fetch.call_args.args
        assert "user-42" in args

    async def test_retrieve_default_top_k_is_5(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """The default top_k must be 5."""
        mock_conn.fetch.return_value = []
        await ltm.retrieve("query text", "user-1")

        args = mock_conn.fetch.call_args.args
        assert 5 in args

    async def test_retrieve_custom_top_k(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """A custom top_k value must be forwarded to the query."""
        mock_conn.fetch.return_value = []
        await ltm.retrieve("query text", "user-1", top_k=10)

        args = mock_conn.fetch.call_args.args
        assert 10 in args

    async def test_retrieve_embeds_query(
        self, ltm: LongTermMemory, mock_openai: AsyncMock
    ) -> None:
        """retrieve must embed the query string before searching."""
        await ltm.retrieve("my query", "user-1")

        mock_openai.embeddings.create.assert_awaited_once()
        call_kwargs = mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "my query"

    async def test_retrieve_multiple_entries(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """retrieve must return all rows returned by the DB."""
        rows = [
            _make_db_row(entry_id="e1", content="dark mode"),
            _make_db_row(entry_id="e2", content="light mode"),
            _make_db_row(entry_id="e3", content="high contrast"),
        ]
        mock_conn.fetch.return_value = rows
        results = await ltm.retrieve("display preference", "user-1", top_k=3)

        assert len(results) == 3
        assert {r.entry_id for r in results} == {"e1", "e2", "e3"}

    async def test_retrieve_latency_within_200ms(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """retrieve must complete within 200ms (Requirement 2.3 SLA).

        This test validates the SLA using mocked I/O — real latency depends on
        the database and network, but the application logic must not add
        significant overhead.
        """
        mock_conn.fetch.return_value = [_make_db_row()]

        start = time.monotonic()
        await ltm.retrieve("dark mode", "user-1")
        elapsed_ms = (time.monotonic() - start) * 1000

        # With mocked I/O the overhead should be well under 200ms.
        assert elapsed_ms < 200, f"retrieve took {elapsed_ms:.1f}ms, expected < 200ms"


# ---------------------------------------------------------------------------
# Tests: delete  (Requirement 2.4)
# ---------------------------------------------------------------------------


class TestDelete:
    """Tests for LongTermMemory.delete — Requirement 2.4."""

    async def test_delete_returns_true_when_row_deleted(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """delete must return True when the DB reports one row deleted."""
        mock_conn.execute.return_value = "DELETE 1"
        result = await ltm.delete("entry-1", "user-1")
        assert result is True

    async def test_delete_returns_false_when_no_row_deleted(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """delete must return False when the DB reports zero rows deleted."""
        mock_conn.execute.return_value = "DELETE 0"
        result = await ltm.delete("entry-nonexistent", "user-1")
        assert result is False

    async def test_delete_passes_entry_id_and_user_id(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """delete must pass both entry_id and user_id to the SQL query."""
        mock_conn.execute.return_value = "DELETE 1"
        await ltm.delete("entry-abc", "user-xyz")

        args = mock_conn.execute.call_args.args
        assert "entry-abc" in args
        assert "user-xyz" in args

    async def test_delete_uses_hard_delete(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """delete must issue a DELETE statement (hard delete, not soft delete)."""
        mock_conn.execute.return_value = "DELETE 1"
        await ltm.delete("entry-1", "user-1")

        sql: str = mock_conn.execute.call_args.args[0]
        assert sql.strip().upper().startswith("DELETE FROM")

    async def test_delete_enforces_user_ownership(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """delete must include user_id in the WHERE clause for ownership check."""
        mock_conn.execute.return_value = "DELETE 0"
        await ltm.delete("entry-1", "wrong-user")

        sql: str = mock_conn.execute.call_args.args[0]
        assert "user_id" in sql

    async def test_delete_wrong_user_returns_false(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """Attempting to delete another user's entry must return False."""
        # DB returns 0 rows because user_id doesn't match
        mock_conn.execute.return_value = "DELETE 0"
        result = await ltm.delete("entry-1", "attacker-user")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: detect_conflict  (Requirement 2.6)
# ---------------------------------------------------------------------------


class TestDetectConflict:
    """Tests for LongTermMemory.detect_conflict — Requirement 2.6."""

    async def test_no_conflict_when_no_entries(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must return has_conflict=False when LTM is empty."""
        mock_conn.fetchrow.return_value = None
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert isinstance(result, ConflictResult)
        assert result.has_conflict is False

    async def test_no_conflict_when_similarity_below_threshold(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must return has_conflict=False when similarity <= 0.85."""
        row = _make_db_row(similarity=0.70)
        mock_conn.fetchrow.return_value = row
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert result.has_conflict is False
        assert result.similarity_score == pytest.approx(0.70)

    async def test_no_conflict_at_exact_threshold(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must return has_conflict=False at exactly the threshold."""
        row = _make_db_row(similarity=_CONFLICT_SIMILARITY_THRESHOLD)
        mock_conn.fetchrow.return_value = row
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert result.has_conflict is False

    async def test_conflict_detected_above_threshold(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must return has_conflict=True when similarity > 0.85."""
        row = _make_db_row(
            content="I prefer light mode",
            similarity=0.92,
        )
        mock_conn.fetchrow.return_value = row
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert result.has_conflict is True
        assert result.similarity_score == pytest.approx(0.92)

    async def test_conflict_result_contains_existing_entry(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """When a conflict is detected, existing_entry must be populated."""
        row = _make_db_row(
            entry_id="existing-entry",
            content="I prefer light mode",
            similarity=0.95,
        )
        mock_conn.fetchrow.return_value = row
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert result.existing_entry is not None
        assert result.existing_entry.entry_id == "existing-entry"
        assert result.existing_entry.content == "I prefer light mode"

    async def test_conflict_result_contains_new_statement(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """The new_statement field must be set in the returned ConflictResult."""
        row = _make_db_row(similarity=0.90)
        mock_conn.fetchrow.return_value = row
        result = await ltm.detect_conflict("I prefer dark mode", "user-1")

        assert result.new_statement == "I prefer dark mode"

    async def test_detect_conflict_embeds_new_statement(
        self, ltm: LongTermMemory, mock_openai: AsyncMock
    ) -> None:
        """detect_conflict must embed the new_statement before querying."""
        await ltm.detect_conflict("I prefer dark mode", "user-1")

        mock_openai.embeddings.create.assert_awaited_once()
        call_kwargs = mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "I prefer dark mode"

    async def test_detect_conflict_passes_user_id_to_query(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must filter by user_id in the SQL query."""
        mock_conn.fetchrow.return_value = None
        await ltm.detect_conflict("some statement", "user-99")

        args = mock_conn.fetchrow.call_args.args
        assert "user-99" in args

    async def test_detect_conflict_queries_only_active_entries(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """detect_conflict must restrict the search to status='active' entries."""
        mock_conn.fetchrow.return_value = None
        await ltm.detect_conflict("some statement", "user-1")

        sql: str = mock_conn.fetchrow.call_args.args[0]
        assert "active" in sql

    async def test_no_conflict_new_statement_preserved_when_no_entries(
        self, ltm: LongTermMemory, mock_conn: AsyncMock
    ) -> None:
        """new_statement must be set even when no LTM entries exist."""
        mock_conn.fetchrow.return_value = None
        result = await ltm.detect_conflict("brand new fact", "user-1")

        assert result.new_statement == "brand new fact"
        assert result.existing_entry is None


# ---------------------------------------------------------------------------
# Tests: row_to_entry helper
# ---------------------------------------------------------------------------


class TestRowToEntry:
    """Tests for the _row_to_entry static helper."""

    def test_parses_json_string_embedding(self) -> None:
        """_row_to_entry must parse a JSON-string embedding into list[float]."""
        row = _make_db_row(embedding=[0.5, 0.3, 0.1])
        entry = LongTermMemory._row_to_entry(row)  # type: ignore[arg-type]
        assert entry.embedding[:3] == pytest.approx([0.5, 0.3, 0.1])

    def test_handles_none_embedding(self) -> None:
        """_row_to_entry must return an empty list when embedding is None."""
        row = _make_db_row()
        row["embedding"] = None
        entry = LongTermMemory._row_to_entry(row)  # type: ignore[arg-type]
        assert entry.embedding == []

    def test_parses_json_string_tags(self) -> None:
        """_row_to_entry must parse JSON-string tags into list[str]."""
        row = _make_db_row(tags=["tag1", "tag2"])
        entry = LongTermMemory._row_to_entry(row)  # type: ignore[arg-type]
        assert entry.tags == ["tag1", "tag2"]

    def test_handles_none_tags(self) -> None:
        """_row_to_entry must return an empty list when tags is None."""
        row = _make_db_row()
        row["tags"] = None
        entry = LongTermMemory._row_to_entry(row)  # type: ignore[arg-type]
        assert entry.tags == []

    def test_maps_all_fields_correctly(self) -> None:
        """_row_to_entry must map every column to the correct MemoryEntry field."""
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        row: dict[str, Any] = {
            "entry_id": "e-123",
            "user_id": "u-456",
            "namespace": "domain_facts",
            "content": "The sky is blue",
            "source_type": "imported",
            "embedding": json.dumps([0.1] * 1536),
            "created_at": ts,
            "tags": json.dumps(["science"]),
        }
        entry = LongTermMemory._row_to_entry(row)  # type: ignore[arg-type]

        assert entry.entry_id == "e-123"
        assert entry.user_id == "u-456"
        assert entry.namespace == "domain_facts"
        assert entry.content == "The sky is blue"
        assert entry.source_type == "imported"
        assert entry.created_at == ts
        assert entry.tags == ["science"]

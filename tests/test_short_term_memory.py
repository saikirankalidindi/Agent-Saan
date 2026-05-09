"""Integration tests for ShortTermMemory (Requirements 1.5, 2.1).

All tests use ``fakeredis.FakeAsyncRedis`` so no live Redis instance is needed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

import fakeredis
import pytest

from agent_saan.memory.short_term import ShortTermMemory, _redis_key
from agent_saan.models.memory import ConversationTurn
from agent_saan.models.nlu import Entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_turn(
    turn_index: int,
    role: str = "user",
    content: str = "hello",
    entities: list[Entity] | None = None,
) -> ConversationTurn:
    """Create a minimal ConversationTurn for testing."""
    return ConversationTurn(
        turn_index=turn_index,
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=datetime(2024, 1, 15, 10, turn_index % 60, 0, tzinfo=timezone.utc),
        entities=entities or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis() -> fakeredis.FakeAsyncRedis:
    """Return a fresh FakeAsyncRedis instance for each test."""
    server = fakeredis.FakeServer()
    return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)


@pytest.fixture
def stm(fake_redis: fakeredis.FakeAsyncRedis) -> ShortTermMemory:
    """Return a ShortTermMemory backed by fakeredis with default settings."""
    return ShortTermMemory(fake_redis, max_turns=50, ttl_seconds=1800)


# ---------------------------------------------------------------------------
# append_turn tests
# ---------------------------------------------------------------------------


class TestAppendTurn:
    """Tests for ShortTermMemory.append_turn."""

    async def test_append_single_turn_stores_it(
        self, stm: ShortTermMemory, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """A single appended turn must be retrievable."""
        turn = make_turn(0, role="user", content="Hello Saan")
        await stm.append_turn("sess-1", turn)

        turns = await stm.get_turns("sess-1")
        assert len(turns) == 1
        assert turns[0].turn_index == 0
        assert turns[0].role == "user"
        assert turns[0].content == "Hello Saan"

    async def test_append_preserves_chronological_order(self, stm: ShortTermMemory) -> None:
        """Turns must be returned in the order they were appended."""
        for i in range(5):
            await stm.append_turn("sess-order", make_turn(i, content=f"msg {i}"))

        turns = await stm.get_turns("sess-order")
        assert [t.turn_index for t in turns] == list(range(5))
        assert [t.content for t in turns] == [f"msg {i}" for i in range(5)]

    async def test_append_resets_ttl(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """Each append must reset the key TTL to ttl_seconds."""
        stm = ShortTermMemory(fake_redis, max_turns=50, ttl_seconds=300)
        session_id = "sess-ttl"
        key = _redis_key(session_id)

        await stm.append_turn(session_id, make_turn(0))
        ttl_after_first = await fake_redis.ttl(key)
        assert 0 < ttl_after_first <= 300

        # Append a second turn; TTL should be reset (still close to 300)
        await stm.append_turn(session_id, make_turn(1))
        ttl_after_second = await fake_redis.ttl(key)
        assert 0 < ttl_after_second <= 300

    async def test_append_trims_to_max_turns(self, fake_redis: fakeredis.FakeAsyncRedis) -> None:
        """When more than max_turns are appended, only the most recent are kept."""
        max_turns = 5
        stm = ShortTermMemory(fake_redis, max_turns=max_turns, ttl_seconds=1800)
        session_id = "sess-trim"

        # Append max_turns + 3 turns
        total = max_turns + 3
        for i in range(total):
            await stm.append_turn(session_id, make_turn(i, content=f"msg {i}"))

        turns = await stm.get_turns(session_id)
        assert len(turns) == max_turns
        # The oldest turns (0, 1, 2) must have been evicted
        assert turns[0].turn_index == 3
        assert turns[-1].turn_index == total - 1

    async def test_append_exactly_max_turns_keeps_all(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """Appending exactly max_turns entries must keep all of them."""
        max_turns = 10
        stm = ShortTermMemory(fake_redis, max_turns=max_turns, ttl_seconds=1800)
        session_id = "sess-exact"

        for i in range(max_turns):
            await stm.append_turn(session_id, make_turn(i))

        turns = await stm.get_turns(session_id)
        assert len(turns) == max_turns

    async def test_append_preserves_entities(self, stm: ShortTermMemory) -> None:
        """Entities attached to a turn must survive the serialisation round-trip."""
        entity = Entity(type="location", value="London", start=10, end=16)
        turn = make_turn(0, entities=[entity])
        await stm.append_turn("sess-entities", turn)

        turns = await stm.get_turns("sess-entities")
        assert len(turns[0].entities) == 1
        assert turns[0].entities[0].type == "location"
        assert turns[0].entities[0].value == "London"

    async def test_append_both_roles(self, stm: ShortTermMemory) -> None:
        """Both 'user' and 'assistant' roles must be stored and retrieved correctly."""
        await stm.append_turn("sess-roles", make_turn(0, role="user", content="Hi"))
        await stm.append_turn("sess-roles", make_turn(1, role="assistant", content="Hello!"))

        turns = await stm.get_turns("sess-roles")
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    async def test_append_sessions_are_isolated(self, stm: ShortTermMemory) -> None:
        """Turns appended to different sessions must not interfere with each other."""
        await stm.append_turn("sess-A", make_turn(0, content="session A"))
        await stm.append_turn("sess-B", make_turn(0, content="session B"))

        turns_a = await stm.get_turns("sess-A")
        turns_b = await stm.get_turns("sess-B")

        assert len(turns_a) == 1
        assert turns_a[0].content == "session A"
        assert len(turns_b) == 1
        assert turns_b[0].content == "session B"


# ---------------------------------------------------------------------------
# get_turns tests
# ---------------------------------------------------------------------------


class TestGetTurns:
    """Tests for ShortTermMemory.get_turns."""

    async def test_get_turns_empty_session_returns_empty_list(
        self, stm: ShortTermMemory
    ) -> None:
        """get_turns on a non-existent session must return an empty list."""
        turns = await stm.get_turns("sess-nonexistent")
        assert turns == []

    async def test_get_turns_returns_conversation_turn_instances(
        self, stm: ShortTermMemory
    ) -> None:
        """get_turns must return a list of ConversationTurn objects."""
        await stm.append_turn("sess-type", make_turn(0))
        turns = await stm.get_turns("sess-type")
        assert all(isinstance(t, ConversationTurn) for t in turns)

    async def test_get_turns_preserves_timestamp(self, stm: ShortTermMemory) -> None:
        """The timestamp field must survive the serialisation round-trip."""
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        turn = ConversationTurn(
            turn_index=0, role="user", content="test", timestamp=ts
        )
        await stm.append_turn("sess-ts", turn)

        turns = await stm.get_turns("sess-ts")
        assert turns[0].timestamp == ts

    async def test_get_turns_returns_all_50_default_max(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """With the default max of 50, exactly 50 turns must be returned after 50 appends."""
        stm = ShortTermMemory(fake_redis, max_turns=50, ttl_seconds=1800)
        session_id = "sess-50"

        for i in range(50):
            await stm.append_turn(session_id, make_turn(i))

        turns = await stm.get_turns(session_id)
        assert len(turns) == 50


# ---------------------------------------------------------------------------
# clear tests
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for ShortTermMemory.clear."""

    async def test_clear_removes_all_turns(self, stm: ShortTermMemory) -> None:
        """After clear(), get_turns must return an empty list."""
        for i in range(3):
            await stm.append_turn("sess-clear", make_turn(i))

        await stm.clear("sess-clear")
        turns = await stm.get_turns("sess-clear")
        assert turns == []

    async def test_clear_nonexistent_session_is_noop(self, stm: ShortTermMemory) -> None:
        """Clearing a session that doesn't exist must not raise an error."""
        # Should not raise
        await stm.clear("sess-does-not-exist")

    async def test_clear_only_affects_target_session(self, stm: ShortTermMemory) -> None:
        """Clearing one session must not affect other sessions."""
        await stm.append_turn("sess-keep", make_turn(0, content="keep me"))
        await stm.append_turn("sess-delete", make_turn(0, content="delete me"))

        await stm.clear("sess-delete")

        kept = await stm.get_turns("sess-keep")
        deleted = await stm.get_turns("sess-delete")

        assert len(kept) == 1
        assert kept[0].content == "keep me"
        assert deleted == []

    async def test_clear_then_append_works(self, stm: ShortTermMemory) -> None:
        """After clearing, new turns can be appended to the same session."""
        await stm.append_turn("sess-reuse", make_turn(0, content="old"))
        await stm.clear("sess-reuse")
        await stm.append_turn("sess-reuse", make_turn(1, content="new"))

        turns = await stm.get_turns("sess-reuse")
        assert len(turns) == 1
        assert turns[0].content == "new"


# ---------------------------------------------------------------------------
# Requirement 2.1 — 50-turn / 30-minute window
# ---------------------------------------------------------------------------


class TestRequirement21:
    """Verify the 50-turn cap and TTL behaviour required by Requirement 2.1."""

    async def test_51st_turn_evicts_oldest(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """Appending the 51st turn must evict the first turn (turn_index=0)."""
        stm = ShortTermMemory(fake_redis, max_turns=50, ttl_seconds=1800)
        session_id = "sess-req21"

        for i in range(51):
            await stm.append_turn(session_id, make_turn(i))

        turns = await stm.get_turns(session_id)
        assert len(turns) == 50
        assert turns[0].turn_index == 1   # turn 0 was evicted
        assert turns[-1].turn_index == 50

    async def test_ttl_is_set_to_30_minutes(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """The Redis key TTL must be set to 1800 seconds (30 minutes) after each append."""
        stm = ShortTermMemory(fake_redis, max_turns=50, ttl_seconds=1800)
        session_id = "sess-ttl30"
        key = _redis_key(session_id)

        await stm.append_turn(session_id, make_turn(0))
        ttl = await fake_redis.ttl(key)
        # TTL should be set and within the expected range
        assert 1799 <= ttl <= 1800

    async def test_custom_max_turns_respected(
        self, fake_redis: fakeredis.FakeAsyncRedis
    ) -> None:
        """A custom max_turns value must be honoured."""
        stm = ShortTermMemory(fake_redis, max_turns=3, ttl_seconds=60)
        session_id = "sess-custom"

        for i in range(5):
            await stm.append_turn(session_id, make_turn(i))

        turns = await stm.get_turns(session_id)
        assert len(turns) == 3
        assert [t.turn_index for t in turns] == [2, 3, 4]

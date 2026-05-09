"""Integration tests for the EventBus (Requirement 8.2).

These tests use ``fakeredis.FakeAsyncRedis`` (with a shared ``FakeServer``) to
simulate Redis Pub/Sub without requiring a live Redis instance.

The ``redis.asyncio.from_url`` factory is patched so that both the publisher and
subscriber connections share the same in-process fake server, which is
necessary for pub/sub messages to be delivered between them.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import fakeredis
import pytest

from agent_saan.events import (
    MEMORY_CONFLICT_DETECTED,
    MEMORY_LTM_RETRIEVED,
    NLU_RESULT,
    PLUGIN_ACTION_TIMEOUT,
    PLUGIN_SECURITY_VIOLATION,
    SAFETY_BLOCK,
    SAFETY_CONFIRMATION_REQUIRED,
    SESSION_ENDED,
    TASK_CREATED,
    TASK_DEADLINE_WARNING,
    EventBus,
    Topics,
)
from agent_saan.models.events import BusEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(event_type: str, source: str = "test", session_id: str = "sess-1") -> BusEvent:
    """Create a minimal BusEvent for testing."""
    return BusEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        source=source,
        session_id=session_id,
        payload={"key": "value"},
        timestamp=datetime.now(tz=timezone.utc),
    )


def make_bus_with_fake_redis() -> tuple[EventBus, fakeredis.FakeServer]:
    """Return an EventBus whose Redis connections are backed by a FakeServer."""
    server = fakeredis.FakeServer()
    bus = EventBus(redis_url="redis://localhost:6379")

    # Each call to redis.asyncio.from_url returns a new FakeAsyncRedis sharing the
    # same FakeServer so that PUBLISH on one connection is visible to SUBSCRIBE
    # on the other.
    call_count = 0

    def fake_from_url(url: str, **kwargs: object) -> fakeredis.FakeAsyncRedis:
        nonlocal call_count
        call_count += 1
        return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)

    bus._fake_from_url = fake_from_url  # type: ignore[attr-defined]
    return bus, server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus() -> AsyncGenerator[EventBus, None]:
    """Provide a connected EventBus backed by fakeredis."""
    server = fakeredis.FakeServer()

    def fake_from_url(url: str, **kwargs: object) -> fakeredis.FakeAsyncRedis:
        return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)

    with patch("redis.asyncio.from_url", side_effect=fake_from_url):
        b = EventBus(redis_url="redis://localhost:6379")
        await b.connect()
        yield b
        await b.stop()
        await b.disconnect()


# ---------------------------------------------------------------------------
# Topic constants tests
# ---------------------------------------------------------------------------


class TestTopicConstants:
    """Verify all required topic constants are defined correctly."""

    def test_nlu_result(self) -> None:
        assert NLU_RESULT == "nlu.result"

    def test_memory_conflict_detected(self) -> None:
        assert MEMORY_CONFLICT_DETECTED == "memory.conflict_detected"

    def test_memory_ltm_retrieved(self) -> None:
        assert MEMORY_LTM_RETRIEVED == "memory.ltm_retrieved"

    def test_safety_block(self) -> None:
        assert SAFETY_BLOCK == "safety.block"

    def test_safety_confirmation_required(self) -> None:
        assert SAFETY_CONFIRMATION_REQUIRED == "safety.confirmation_required"

    def test_task_created(self) -> None:
        assert TASK_CREATED == "task.created"

    def test_task_deadline_warning(self) -> None:
        assert TASK_DEADLINE_WARNING == "task.deadline_warning"

    def test_plugin_action_timeout(self) -> None:
        assert PLUGIN_ACTION_TIMEOUT == "plugin.action_timeout"

    def test_plugin_security_violation(self) -> None:
        assert PLUGIN_SECURITY_VIOLATION == "plugin.security_violation"

    def test_session_ended(self) -> None:
        assert SESSION_ENDED == "session.ended"

    def test_topics_class_matches_module_constants(self) -> None:
        assert Topics.NLU_RESULT == NLU_RESULT
        assert Topics.MEMORY_CONFLICT_DETECTED == MEMORY_CONFLICT_DETECTED
        assert Topics.MEMORY_LTM_RETRIEVED == MEMORY_LTM_RETRIEVED
        assert Topics.SAFETY_BLOCK == SAFETY_BLOCK
        assert Topics.SAFETY_CONFIRMATION_REQUIRED == SAFETY_CONFIRMATION_REQUIRED
        assert Topics.TASK_CREATED == TASK_CREATED
        assert Topics.TASK_DEADLINE_WARNING == TASK_DEADLINE_WARNING
        assert Topics.PLUGIN_ACTION_TIMEOUT == PLUGIN_ACTION_TIMEOUT
        assert Topics.PLUGIN_SECURITY_VIOLATION == PLUGIN_SECURITY_VIOLATION
        assert Topics.SESSION_ENDED == SESSION_ENDED

    def test_all_topics_list_contains_all_constants(self) -> None:
        from agent_saan.events.topics import ALL_TOPICS

        expected = {
            NLU_RESULT,
            MEMORY_CONFLICT_DETECTED,
            MEMORY_LTM_RETRIEVED,
            SAFETY_BLOCK,
            SAFETY_CONFIRMATION_REQUIRED,
            TASK_CREATED,
            TASK_DEADLINE_WARNING,
            PLUGIN_ACTION_TIMEOUT,
            PLUGIN_SECURITY_VIOLATION,
            SESSION_ENDED,
        }
        assert set(ALL_TOPICS) == expected


# ---------------------------------------------------------------------------
# EventBus unit tests (no Redis required)
# ---------------------------------------------------------------------------


class TestEventBusSubscribe:
    """Tests for subscribe/unsubscribe that don't require a live connection."""

    def test_subscribe_registers_handler(self) -> None:
        b = EventBus()
        handler = AsyncMock()
        b.subscribe(NLU_RESULT, handler)
        assert handler in b._handlers[NLU_RESULT]

    def test_subscribe_multiple_handlers_same_topic(self) -> None:
        b = EventBus()
        h1, h2 = AsyncMock(), AsyncMock()
        b.subscribe(NLU_RESULT, h1)
        b.subscribe(NLU_RESULT, h2)
        assert b._handlers[NLU_RESULT] == [h1, h2]

    def test_unsubscribe_removes_handler(self) -> None:
        b = EventBus()
        handler = AsyncMock()
        b.subscribe(NLU_RESULT, handler)
        b.unsubscribe(NLU_RESULT, handler)
        assert handler not in b._handlers[NLU_RESULT]

    def test_unsubscribe_nonexistent_handler_is_noop(self) -> None:
        b = EventBus()
        handler = AsyncMock()
        # Should not raise
        b.unsubscribe(NLU_RESULT, handler)

    def test_publish_without_connect_raises(self) -> None:
        b = EventBus()
        event = make_event(NLU_RESULT)
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.get_event_loop().run_until_complete(b.publish(event))

    def test_start_without_connect_raises(self) -> None:
        b = EventBus()
        b.subscribe(NLU_RESULT, AsyncMock())
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.get_event_loop().run_until_complete(b.start())


# ---------------------------------------------------------------------------
# EventBus integration tests (fakeredis)
# ---------------------------------------------------------------------------


class TestEventBusPublishSubscribe:
    """Publish/subscribe round-trip tests using fakeredis."""

    async def test_publish_subscribe_round_trip(self, bus: EventBus) -> None:
        """A published event must be received by the registered handler."""
        received: list[BusEvent] = []
        event = make_event(NLU_RESULT)

        async def handler(e: BusEvent) -> None:
            received.append(e)

        bus.subscribe(NLU_RESULT, handler)
        await bus.start()

        # Give the listener task time to subscribe before publishing
        await asyncio.sleep(0.05)
        await bus.publish(event)

        # Wait for the handler to be called
        deadline = asyncio.get_event_loop().time() + 2.0
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].event_id == event.event_id
        assert received[0].event_type == NLU_RESULT
        assert received[0].source == "test"
        assert received[0].payload == {"key": "value"}

    async def test_handler_receives_correct_bus_event_fields(self, bus: EventBus) -> None:
        """Handler must receive a BusEvent with all original fields intact."""
        received: list[BusEvent] = []
        session_id = "session-abc-123"
        event = BusEvent(
            event_id="evt-001",
            event_type=TASK_CREATED,
            source="task_manager",
            session_id=session_id,
            payload={"task_id": "t-42", "priority": "high"},
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        )

        async def handler(e: BusEvent) -> None:
            received.append(e)

        bus.subscribe(TASK_CREATED, handler)
        await bus.start()
        await asyncio.sleep(0.05)
        await bus.publish(event)

        deadline = asyncio.get_event_loop().time() + 2.0
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(received) == 1
        e = received[0]
        assert e.event_id == "evt-001"
        assert e.event_type == TASK_CREATED
        assert e.source == "task_manager"
        assert e.session_id == session_id
        assert e.payload == {"task_id": "t-42", "priority": "high"}

    async def test_multiple_subscribers_same_topic(self, bus: EventBus) -> None:
        """All handlers registered for a topic must be called."""
        received_1: list[BusEvent] = []
        received_2: list[BusEvent] = []
        event = make_event(SAFETY_BLOCK)

        async def handler_1(e: BusEvent) -> None:
            received_1.append(e)

        async def handler_2(e: BusEvent) -> None:
            received_2.append(e)

        bus.subscribe(SAFETY_BLOCK, handler_1)
        bus.subscribe(SAFETY_BLOCK, handler_2)
        await bus.start()
        await asyncio.sleep(0.05)
        await bus.publish(event)

        deadline = asyncio.get_event_loop().time() + 2.0
        while (not received_1 or not received_2) and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0].event_id == event.event_id
        assert received_2[0].event_id == event.event_id

    async def test_unsubscribe_stops_handler_from_receiving(self, bus: EventBus) -> None:
        """After unsubscribing, a handler must not receive further events."""
        received: list[BusEvent] = []

        async def handler(e: BusEvent) -> None:
            received.append(e)

        bus.subscribe(SESSION_ENDED, handler)
        await bus.start()
        await asyncio.sleep(0.05)

        # Publish first event — handler should receive it
        event_1 = make_event(SESSION_ENDED)
        await bus.publish(event_1)

        deadline = asyncio.get_event_loop().time() + 2.0
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        assert len(received) == 1

        # Unsubscribe and publish second event — handler must NOT receive it
        bus.unsubscribe(SESSION_ENDED, handler)
        event_2 = make_event(SESSION_ENDED)
        await bus.publish(event_2)
        await asyncio.sleep(0.2)

        # Still only one event received
        assert len(received) == 1

    async def test_publish_to_different_topics_does_not_cross_deliver(
        self, bus: EventBus
    ) -> None:
        """A handler subscribed to topic A must not receive events on topic B."""
        received_nlu: list[BusEvent] = []
        received_safety: list[BusEvent] = []

        async def nlu_handler(e: BusEvent) -> None:
            received_nlu.append(e)

        async def safety_handler(e: BusEvent) -> None:
            received_safety.append(e)

        bus.subscribe(NLU_RESULT, nlu_handler)
        bus.subscribe(SAFETY_BLOCK, safety_handler)
        await bus.start()
        await asyncio.sleep(0.05)

        nlu_event = make_event(NLU_RESULT)
        safety_event = make_event(SAFETY_BLOCK)
        await bus.publish(nlu_event)
        await bus.publish(safety_event)

        deadline = asyncio.get_event_loop().time() + 2.0
        while (
            not received_nlu or not received_safety
        ) and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(received_nlu) == 1
        assert received_nlu[0].event_id == nlu_event.event_id
        assert len(received_safety) == 1
        assert received_safety[0].event_id == safety_event.event_id

    async def test_stop_cancels_listener_tasks(self, bus: EventBus) -> None:
        """After stop(), all listener tasks must be cancelled and cleared."""
        bus.subscribe(NLU_RESULT, AsyncMock())
        await bus.start()
        await asyncio.sleep(0.05)

        assert len(bus._listener_tasks) == 1
        await bus.stop()
        assert len(bus._listener_tasks) == 0

    async def test_context_manager_connects_and_disconnects(self) -> None:
        """The async context manager must connect on enter and stop/disconnect on exit."""
        server = fakeredis.FakeServer()

        def fake_from_url(url: str, **kwargs: object) -> fakeredis.FakeAsyncRedis:
            return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)

        with patch("redis.asyncio.from_url", side_effect=fake_from_url):
            async with EventBus() as b:
                assert b._publisher is not None
                assert b._subscriber is not None

            # After exiting the context manager, connections are closed
            assert b._publisher is None
            assert b._subscriber is None

    async def test_publish_multiple_events_same_topic(self, bus: EventBus) -> None:
        """Multiple events published to the same topic must all be delivered."""
        received: list[BusEvent] = []

        async def handler(e: BusEvent) -> None:
            received.append(e)

        bus.subscribe(PLUGIN_ACTION_TIMEOUT, handler)
        await bus.start()
        await asyncio.sleep(0.05)

        events = [make_event(PLUGIN_ACTION_TIMEOUT) for _ in range(3)]
        for evt in events:
            await bus.publish(evt)

        deadline = asyncio.get_event_loop().time() + 2.0
        while len(received) < 3 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(received) == 3
        received_ids = {e.event_id for e in received}
        expected_ids = {e.event_id for e in events}
        assert received_ids == expected_ids

    async def test_handler_exception_does_not_crash_listener(self, bus: EventBus) -> None:
        """An exception raised by a handler must not crash the listener task."""
        good_received: list[BusEvent] = []

        async def bad_handler(e: BusEvent) -> None:
            raise ValueError("intentional test error")

        async def good_handler(e: BusEvent) -> None:
            good_received.append(e)

        bus.subscribe(MEMORY_CONFLICT_DETECTED, bad_handler)
        bus.subscribe(MEMORY_CONFLICT_DETECTED, good_handler)
        await bus.start()
        await asyncio.sleep(0.05)

        event = make_event(MEMORY_CONFLICT_DETECTED)
        await bus.publish(event)

        deadline = asyncio.get_event_loop().time() + 2.0
        while not good_received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        # The good handler must still have been called despite the bad handler raising
        assert len(good_received) == 1
        # The listener task must still be running
        assert not bus._listener_tasks[MEMORY_CONFLICT_DETECTED].done()

    async def test_all_defined_topics_can_be_published_and_received(
        self, bus: EventBus
    ) -> None:
        """Every topic constant must support a full publish/subscribe round-trip."""
        from agent_saan.events.topics import ALL_TOPICS

        received: dict[str, list[BusEvent]] = {t: [] for t in ALL_TOPICS}

        for topic in ALL_TOPICS:
            topic_ref = topic  # capture for closure

            async def make_handler(t: str) -> None:
                async def handler(e: BusEvent) -> None:
                    received[t].append(e)

                bus.subscribe(t, handler)

            await make_handler(topic_ref)

        await bus.start()
        await asyncio.sleep(0.05)

        events = {t: make_event(t) for t in ALL_TOPICS}
        for evt in events.values():
            await bus.publish(evt)

        deadline = asyncio.get_event_loop().time() + 3.0
        while (
            any(len(v) == 0 for v in received.values())
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        for topic in ALL_TOPICS:
            assert len(received[topic]) == 1, f"No event received for topic '{topic}'"
            assert received[topic][0].event_id == events[topic].event_id

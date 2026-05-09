"""Event Bus — async pub/sub layer built on Redis Pub/Sub via redis.asyncio.

All inter-subsystem communication in Agent Saan is mediated exclusively
through this event bus.  No subsystem may call another subsystem's internal
methods directly (Requirement 8.2).

Usage example::

    bus = EventBus(redis_url="redis://localhost:6379")
    await bus.connect()

    async def on_nlu_result(event: BusEvent) -> None:
        print(event.payload)

    bus.subscribe(Topics.NLU_RESULT, on_nlu_result)
    await bus.start()

    await bus.publish(BusEvent(...))

    await bus.stop()
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

import redis.asyncio as aioredis

from agent_saan.models.events import BusEvent

logger = logging.getLogger(__name__)

# Type alias for async event handlers
EventHandler = Callable[[BusEvent], Awaitable[None]]


class EventBus:
    """Async publish/subscribe event bus backed by Redis Pub/Sub.

    Lifecycle
    ---------
    1. Instantiate with a Redis URL.
    2. Call ``connect()`` (or use as an async context manager) to open the
       Redis connections.
    3. Register handlers with ``subscribe()``.
    4. Call ``start()`` to launch background listener tasks for every
       subscribed topic.
    5. Call ``stop()`` (or exit the context manager) to cancel all listener
       tasks and close connections gracefully.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        # publisher connection — used only for PUBLISH commands
        self._publisher: aioredis.Redis | None = None
        # subscriber connection — used only for SUBSCRIBE / message reads
        self._subscriber: aioredis.Redis | None = None
        # topic → list of handlers
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        # topic → running asyncio.Task
        self._listener_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open publisher and subscriber Redis connections."""
        self._publisher = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._subscriber = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("EventBus connected to Redis at %s", self._redis_url)

    async def disconnect(self) -> None:
        """Close both Redis connections."""
        if self._publisher:
            await self._publisher.aclose()
            self._publisher = None
        if self._subscriber:
            await self._subscriber.aclose()
            self._subscriber = None
        logger.info("EventBus disconnected from Redis")

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "EventBus":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def publish(self, event: BusEvent) -> None:
        """Serialize *event* to JSON and publish it to ``event.event_type``.

        Args:
            event: The structured event to publish.

        Raises:
            RuntimeError: If ``connect()`` has not been called.
        """
        if self._publisher is None:
            raise RuntimeError("EventBus is not connected. Call connect() first.")

        topic = event.event_type
        payload = event.model_dump_json()
        try:
            await self._publisher.publish(topic, payload)
            logger.debug("Published event %s to topic '%s'", event.event_id, topic)
        except aioredis.exceptions.RedisError as exc:
            logger.error(
                "Failed to publish event %s to topic '%s': %s",
                event.event_id,
                topic,
                exc,
            )
            raise

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register *handler* to be called whenever a message arrives on *topic*.

        Multiple handlers may be registered for the same topic.  Handlers are
        called sequentially in registration order.

        Args:
            topic: The topic string (e.g. ``"nlu.result"``).
            handler: An async callable that accepts a single :class:`BusEvent`.
        """
        self._handlers[topic].append(handler)
        logger.debug("Registered handler %s for topic '%s'", handler.__name__, topic)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove a previously registered *handler* from *topic*.

        If the handler is not registered, this is a no-op.

        Args:
            topic: The topic string.
            handler: The handler to remove.
        """
        handlers = self._handlers.get(topic, [])
        try:
            handlers.remove(handler)
            logger.debug("Unsubscribed handler %s from topic '%s'", handler.__name__, topic)
        except ValueError:
            pass  # handler was not registered — silently ignore

    async def start(self) -> None:
        """Start background listener tasks for all subscribed topics.

        Each topic gets its own ``asyncio.Task`` that reads from a dedicated
        Redis Pub/Sub channel.  Calling ``start()`` again after it has already
        been started is safe — existing tasks are reused and only new topics
        get new tasks.

        Raises:
            RuntimeError: If ``connect()`` has not been called.
        """
        if self._subscriber is None:
            raise RuntimeError("EventBus is not connected. Call connect() first.")

        for topic in list(self._handlers.keys()):
            if topic not in self._listener_tasks or self._listener_tasks[topic].done():
                task = asyncio.create_task(
                    self._listen(topic),
                    name=f"event-bus-listener:{topic}",
                )
                self._listener_tasks[topic] = task
                logger.info("Started listener task for topic '%s'", topic)

    async def stop(self) -> None:
        """Cancel all background listener tasks and wait for them to finish."""
        tasks = list(self._listener_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._listener_tasks.clear()
        logger.info("EventBus stopped all listener tasks")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _listen(self, topic: str) -> None:
        """Background coroutine: subscribe to *topic* and dispatch messages.

        Runs until cancelled.  Reconnects automatically on transient Redis
        errors with a short back-off to avoid tight error loops.
        """
        if self._subscriber is None:
            return

        backoff = 0.5  # seconds
        while True:
            try:
                async with self._subscriber.pubsub() as pubsub:
                    await pubsub.subscribe(topic)
                    logger.debug("Listening on topic '%s'", topic)
                    async for raw_message in pubsub.listen():
                        if raw_message["type"] != "message":
                            continue
                        await self._dispatch(topic, raw_message["data"])
            except asyncio.CancelledError:
                logger.debug("Listener task for topic '%s' cancelled", topic)
                return
            except aioredis.exceptions.RedisError as exc:
                logger.warning(
                    "Redis error on topic '%s': %s — retrying in %.1fs",
                    topic,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _dispatch(self, topic: str, raw: str) -> None:
        """Deserialize *raw* JSON and invoke all registered handlers."""
        try:
            event = BusEvent.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to deserialize event on topic '%s': %s", topic, exc)
            return

        handlers = list(self._handlers.get(topic, []))
        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Handler %s raised an exception for event %s: %s",
                    handler.__name__,
                    event.event_id,
                    exc,
                )

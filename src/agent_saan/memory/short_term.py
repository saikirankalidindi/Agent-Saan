"""Short-Term Memory backed by Redis.

Each session's conversation history is stored as a Redis list under the key
``stm:{session_id}``.  The list is capped at ``max_turns`` entries and has a
sliding TTL that resets to ``ttl_seconds`` on every write.

Requirements: 1.5, 2.1
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

from agent_saan.models.memory import ConversationTurn

if TYPE_CHECKING:
    pass

_KEY_PREFIX = "stm"


def _redis_key(session_id: str) -> str:
    """Return the Redis list key for a given session."""
    return f"{_KEY_PREFIX}:{session_id}"


class ShortTermMemory:
    """Manages per-session conversation history in Redis.

    Parameters
    ----------
    redis_client:
        An already-connected ``redis.asyncio`` client (or a ``fakeredis``
        compatible async client for testing).
    max_turns:
        Maximum number of turns to retain per session (default: 50).
    ttl_seconds:
        Sliding TTL in seconds; reset on every write (default: 1800 = 30 min).
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        *,
        max_turns: int = 50,
        ttl_seconds: int = 1800,
    ) -> None:
        self._redis = redis_client
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append *turn* to the session history, trim to max_turns, reset TTL.

        The turn is serialised to JSON and pushed to the right end of the Redis
        list.  After pushing, the list is trimmed so that only the most recent
        ``max_turns`` entries are kept.  The key TTL is then reset to
        ``ttl_seconds`` (sliding window).
        """
        key = _redis_key(session_id)
        serialised = turn.model_dump_json()

        # Use a pipeline so the three operations are sent in a single round-trip.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, serialised)
            # Keep only the last max_turns entries (0-indexed from the right).
            pipe.ltrim(key, -self._max_turns, -1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def get_turns(self, session_id: str) -> list[ConversationTurn]:
        """Return all stored turns for *session_id* in chronological order.

        Returns an empty list if the session key does not exist.
        """
        key = _redis_key(session_id)
        raw_items: list[bytes | str] = await self._redis.lrange(key, 0, -1)
        turns: list[ConversationTurn] = []
        for raw in raw_items:
            data = raw if isinstance(raw, str) else raw.decode()
            turns.append(ConversationTurn.model_validate_json(data))
        return turns

    async def clear(self, session_id: str) -> None:
        """Delete the Redis key for *session_id*, removing all stored turns."""
        key = _redis_key(session_id)
        await self._redis.delete(key)

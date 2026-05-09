"""Event Bus — async pub/sub layer built on Redis Pub/Sub via aioredis."""

from agent_saan.events.bus import EventBus, EventHandler
from agent_saan.events.topics import (
    ALL_TOPICS,
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
    Topics,
)

__all__ = [
    # Core class
    "EventBus",
    "EventHandler",
    # Topics namespace
    "Topics",
    "ALL_TOPICS",
    # Individual topic constants
    "NLU_RESULT",
    "MEMORY_CONFLICT_DETECTED",
    "MEMORY_LTM_RETRIEVED",
    "SAFETY_BLOCK",
    "SAFETY_CONFIRMATION_REQUIRED",
    "TASK_CREATED",
    "TASK_DEADLINE_WARNING",
    "PLUGIN_ACTION_TIMEOUT",
    "PLUGIN_SECURITY_VIOLATION",
    "SESSION_ENDED",
]

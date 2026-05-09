"""Topic constants for the Agent Saan event bus.

All inter-subsystem communication uses these named topics.
Convention: {source}.{event_type}
"""


class Topics:
    """Namespace for all event bus topic constants."""

    # NLU subsystem
    NLU_RESULT = "nlu.result"

    # Memory subsystem
    MEMORY_CONFLICT_DETECTED = "memory.conflict_detected"
    MEMORY_LTM_RETRIEVED = "memory.ltm_retrieved"

    # Safety subsystem
    SAFETY_BLOCK = "safety.block"
    SAFETY_CONFIRMATION_REQUIRED = "safety.confirmation_required"

    # Task subsystem
    TASK_CREATED = "task.created"
    TASK_DEADLINE_WARNING = "task.deadline_warning"

    # Plugin subsystem
    PLUGIN_ACTION_TIMEOUT = "plugin.action_timeout"
    PLUGIN_SECURITY_VIOLATION = "plugin.security_violation"

    # Session lifecycle
    SESSION_ENDED = "session.ended"


# Module-level aliases for convenient import
NLU_RESULT = Topics.NLU_RESULT
MEMORY_CONFLICT_DETECTED = Topics.MEMORY_CONFLICT_DETECTED
MEMORY_LTM_RETRIEVED = Topics.MEMORY_LTM_RETRIEVED
SAFETY_BLOCK = Topics.SAFETY_BLOCK
SAFETY_CONFIRMATION_REQUIRED = Topics.SAFETY_CONFIRMATION_REQUIRED
TASK_CREATED = Topics.TASK_CREATED
TASK_DEADLINE_WARNING = Topics.TASK_DEADLINE_WARNING
PLUGIN_ACTION_TIMEOUT = Topics.PLUGIN_ACTION_TIMEOUT
PLUGIN_SECURITY_VIOLATION = Topics.PLUGIN_SECURITY_VIOLATION
SESSION_ENDED = Topics.SESSION_ENDED

ALL_TOPICS: list[str] = [
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
]

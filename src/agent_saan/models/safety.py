"""Safety and guardrail models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class AuditLogEntry(BaseModel):
    """An immutable audit log record for every action attempted."""

    log_id: str
    action_id: str
    user_id: str
    session_id: str
    outcome: Literal["allowed", "blocked", "cancelled"]
    guardrail_rule: str | None = None
    timestamp: datetime


class GuardrailDecision(BaseModel):
    """The decision returned by the Safety Guard after evaluating an action."""

    decision: Literal["allow", "block", "require_confirmation"]
    rule_name: str | None = None
    reason: str | None = None

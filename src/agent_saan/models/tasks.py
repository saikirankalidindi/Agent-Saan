"""Task management models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A structured task record managed by the Task Manager."""

    task_id: str
    user_id: str
    title: str
    description: str
    priority: Literal["high", "medium", "low"] = "medium"
    deadline: datetime
    status: Literal[
        "pending", "in_progress", "completed", "pending_authorization", "dismissed"
    ] = "pending"
    completion_source: Literal["user", "agent"] | None = None
    completed_at: datetime | None = None
    recurrence: str | None = None  # cron expression or "daily"/"weekly"/"monthly"
    created_at: datetime = Field(default_factory=datetime.utcnow)

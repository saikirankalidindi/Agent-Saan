"""Plugin system models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionSchema(BaseModel):
    """Schema definition for an action exposed by a plugin."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    is_reversible: bool


class PluginManifest(BaseModel):
    """Manifest describing a plugin's identity, permissions, and exposed actions."""

    name: str
    version: str  # semver
    description: str
    permissions: list[str] = Field(default_factory=list)
    actions: list[ActionSchema] = Field(default_factory=list)
    timeout_seconds: int = Field(default=10, ge=1, le=60)


class Action(BaseModel):
    """A discrete, executable unit of work dispatched to a plugin or subsystem."""

    action_id: str
    session_id: str
    plugin_name: str
    action_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    is_reversible: bool
    dispatched_at: datetime


class ActionResult(BaseModel):
    """The structured result returned after executing an action."""

    action_id: str
    status: Literal["success", "error", "timeout", "cancelled", "blocked"]
    result: dict[str, Any] | None = None
    error: str | None = None
    completed_at: datetime

"""User preferences model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """Persistent user configuration controlling Agent Saan's behavior and style."""

    user_id: str
    communication_style: Literal["formal", "casual", "technical"] = "casual"
    verbosity: Literal["concise", "standard", "detailed"] = "standard"
    voice_output_enabled: bool = False
    tts_voice_gender: Literal["male", "female"] = "female"
    tts_speech_rate_wpm: int = Field(default=150, ge=80, le=200)
    tts_pitch: Literal["low", "medium", "high"] = "medium"
    action_rate_limit: int = Field(default=100, ge=100, le=1000)
    safe_mode: bool = False
    category_confidence_weights: dict[str, float] = Field(default_factory=dict)

"""NLU (Natural Language Understanding) models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Raw input submitted by the user in any modality."""

    input_id: str
    session_id: str
    modality: Literal["text", "audio", "image"]
    content: str | bytes
    timestamp: datetime


class Intent(BaseModel):
    """A parsed intent extracted from user input."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A named entity extracted from user input."""

    type: str
    value: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class NLUResult(BaseModel):
    """Structured output from the NLU Engine after parsing user input."""

    input_id: str
    intents: list[Intent]  # ranked by confidence descending
    entities: list[Entity] = Field(default_factory=list)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    language: str  # ISO 639-1 code
    is_ambiguous: bool


class AudioInput(BaseModel):
    """Audio input submitted by the user for transcription.

    Requirements 9.1, 9.2, 9.6:
    - Supported formats: WAV, MP3, OGG
    - Maximum size: 10 MB
    - Maximum duration: 5 minutes (300 seconds)
    """

    audio_id: str
    session_id: str
    # Raw audio bytes
    content: bytes
    # MIME type or simple format string: "wav", "mp3", "ogg"
    format: str
    # Duration in seconds (optional — may be computed after loading)
    duration_seconds: float | None = None
    timestamp: datetime


class TranscriptResult(BaseModel):
    """Result returned by NLUEngine.transcribe_audio().

    Requirements 9.2:
    - transcript: the transcribed text
    - word_error_rate: estimated WER (0.0–1.0); 0.05 or lower meets the 5% SLA
    """

    audio_id: str
    transcript: str
    # Estimated word error rate (0.0 = perfect, 1.0 = completely wrong)
    word_error_rate: float = Field(ge=0.0, le=1.0)
    # Signal-to-noise ratio in dB computed before transcription
    snr_db: float


class ImageInput(BaseModel):
    """Image input submitted by the user for visual understanding.

    Requirements 9.1, 9.3:
    - Supported formats: JPEG, PNG, WEBP
    - Maximum size: 20 MB
    """

    image_id: str
    session_id: str
    # Raw image bytes
    content: bytes
    # MIME type or simple format string: "jpeg", "jpg", "png", "webp"
    format: str
    timestamp: datetime

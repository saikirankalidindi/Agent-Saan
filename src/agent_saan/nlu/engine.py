"""NLU Engine — text parsing, intent extraction, entity recognition, and sentiment analysis."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import numpy as np

from openai import AsyncOpenAI, OpenAIError

from agent_saan.config import get_settings
from agent_saan.models.memory import ConversationTurn
from agent_saan.models.nlu import AudioInput, Entity, ImageInput, Intent, NLUResult, TranscriptResult, UserInput

logger = logging.getLogger(__name__)

# Maximum allowed input length (Requirement 1.6)
MAX_INPUT_CHARS = 10_000

# Maximum STM turns to include in the prompt (Requirement 1.5)
MAX_CONTEXT_TURNS = 50

# Maximum parse attempts before returning a structured error (Requirement 1.4)
MAX_PARSE_ATTEMPTS = 2

# Ambiguity threshold: top-2 confidence delta (Requirement 1.2)
AMBIGUITY_DELTA = 0.15

# ---------------------------------------------------------------------------
# Audio transcription constants (Requirements 9.1, 9.2, 9.6)
# ---------------------------------------------------------------------------

# Supported audio formats (Requirement 9.1)
SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset({"wav", "mp3", "ogg"})

# Maximum audio file size in bytes: 10 MB (Requirement 9.1)
MAX_AUDIO_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Maximum audio duration in seconds: 5 minutes (Requirement 9.1)
MAX_AUDIO_DURATION_SECONDS: float = 5 * 60.0  # 300 s

# Minimum acceptable signal-to-noise ratio in dB (Requirements 9.2, 9.6)
MIN_SNR_DB: float = 20.0

# Whisper API call timeout in seconds (Requirement 9.2)
WHISPER_TIMEOUT_SECONDS: float = 3.0

# Whisper model to use
WHISPER_MODEL: str = "whisper-1"

# ---------------------------------------------------------------------------
# Image description constants (Requirements 9.1, 9.3)
# ---------------------------------------------------------------------------

# Supported image formats (Requirement 9.1)
SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset({"jpeg", "jpg", "png", "webp"})

# Maximum image file size in bytes: 20 MB (Requirement 9.1)
MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB

# GPT-4o Vision model to use for image description
VISION_MODEL: str = "gpt-4o"

# JSON schema for the structured NLU response expected from GPT-4o
_NLU_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "parameters": {"type": "object"},
                },
                "required": ["name", "confidence", "parameters"],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "value": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                },
                "required": ["type", "value", "start", "end"],
            },
        },
        "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "language": {"type": "string", "description": "ISO 639-1 language code, e.g. 'en', 'es'"},
    },
    "required": ["intents", "entities", "sentiment_score", "language"],
}

_SYSTEM_PROMPT = """\
You are an NLU (Natural Language Understanding) engine for an AI assistant called Saan.
Your task is to analyse the user's message and return a structured JSON object with:
  - intents: list of detected intents ranked by confidence (highest first). Each intent has:
      name (string), confidence (0.0–1.0), parameters (object with extracted slot values).
  - entities: list of named entities with type, value, start char offset, end char offset.
  - sentiment_score: float from -1.0 (very negative) to 1.0 (very positive).
  - language: ISO 639-1 code of the language the user wrote in (e.g. "en", "es", "fr").

Always return at least one intent. If the input is unclear, use intent name "unknown" with
an appropriate confidence. Return ONLY the JSON object — no markdown, no extra text.
"""


class NLUParseError(Exception):
    """Raised when the NLU engine fails to parse input after all attempts."""

    def __init__(self, input_excerpt: str, failure_reason: str, suggestion: str) -> None:
        self.input_excerpt = input_excerpt
        self.failure_reason = failure_reason
        self.suggestion = suggestion
        super().__init__(failure_reason)


class NLUInputTooLongError(Exception):
    """Raised when the input exceeds the maximum allowed character length."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.limit = MAX_INPUT_CHARS
        super().__init__(
            f"Input length {length} exceeds the maximum allowed {MAX_INPUT_CHARS} characters."
        )


class AudioFormatError(Exception):
    """Raised when the audio format is not supported or the file is too large/long."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class AudioSNRError(Exception):
    """Raised when the audio SNR is below the minimum threshold (Requirement 9.6)."""

    def __init__(self, snr_db: float) -> None:
        self.snr_db = snr_db
        self.threshold = MIN_SNR_DB
        super().__init__(
            f"Audio SNR {snr_db:.1f} dB is below the minimum required {MIN_SNR_DB} dB. "
            "Please re-submit in a quieter environment or switch to text input."
        )


class AudioTranscriptionTimeoutError(Exception):
    """Raised when the Whisper API call exceeds the configured timeout."""

    def __init__(self) -> None:
        super().__init__(
            f"Audio transcription timed out after {WHISPER_TIMEOUT_SECONDS} seconds."
        )


class ImageFormatError(Exception):
    """Raised when the image format is not supported or the file is too large."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class NLUEngine:
    """Natural Language Understanding engine backed by GPT-4o.

    Responsibilities:
    - Validate input length (≤10,000 chars).
    - Build a prompt that includes the last 50 STM turns as conversation history.
    - Call GPT-4o with structured JSON output.
    - Parse the response into an NLUResult.
    - Detect ambiguous intents (top-2 confidence delta ≤ 0.15).
    - Detect the language from the LLM response and return an ISO 639-1 code.
    - Return a structured error after 2 failed parse attempts.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = "gpt-4o"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        input: UserInput,
        context: list[ConversationTurn] | None = None,
    ) -> NLUResult:
        """Parse a text UserInput and return a structured NLUResult.

        Args:
            input: The UserInput to parse. Must have modality="text".
            context: Optional list of recent ConversationTurns from STM.
                     Only the last MAX_CONTEXT_TURNS turns are used.

        Returns:
            NLUResult with intents, entities, sentiment, language, and ambiguity flag.

        Raises:
            NLUInputTooLongError: If the text content exceeds MAX_INPUT_CHARS.
            NLUParseError: If parsing fails after MAX_PARSE_ATTEMPTS attempts.
        """
        text = input.content if isinstance(input.content, str) else input.content.decode("utf-8")

        # Requirement 1.6 — reject inputs > 10,000 chars before any LLM call
        if len(text) > MAX_INPUT_CHARS:
            raise NLUInputTooLongError(len(text))

        messages = self._build_messages(text, context or [])

        last_error: Exception | None = None
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            try:
                raw = await self._call_llm(messages)
                result = self._parse_response(raw, input.input_id)
                result.is_ambiguous = self.is_ambiguous(result)
                return result
            except (json.JSONDecodeError, KeyError, ValueError, OpenAIError) as exc:
                logger.warning("NLU parse attempt %d/%d failed: %s", attempt, MAX_PARSE_ATTEMPTS, exc)
                last_error = exc

        # Requirement 1.4 — structured error after 2 failed attempts
        excerpt = text[:200] + ("…" if len(text) > 200 else "")
        raise NLUParseError(
            input_excerpt=excerpt,
            failure_reason=f"PARSE_FAILURE: {last_error}",
            suggestion=(
                "Please try rephrasing your input more clearly, "
                "or break it into shorter sentences."
            ),
        )

    def is_ambiguous(self, result: NLUResult) -> bool:
        """Return True if the top-2 intents have a confidence delta ≤ AMBIGUITY_DELTA.

        Requirement 1.2: ambiguous when two or more candidate intents each have a
        confidence score within 0.15 of each other.

        Uses round() to avoid floating-point precision issues (e.g. 0.80 - 0.65
        evaluates to 0.15000000000000002 in IEEE 754 arithmetic).
        """
        if len(result.intents) < 2:
            return False
        top, second = result.intents[0].confidence, result.intents[1].confidence
        delta = round(top - second, 10)
        return delta <= AMBIGUITY_DELTA

    async def transcribe_audio(self, audio: AudioInput) -> TranscriptResult:
        """Transcribe an audio input using the OpenAI Whisper API.

        Pipeline (Requirements 9.1, 9.2, 9.6):
        1. Validate format (WAV/MP3/OGG) and size (≤10 MB).
        2. Load audio with librosa and validate duration (≤5 min).
        3. Compute SNR; reject if SNR < 20 dB.
        4. Call Whisper API with a 3-second asyncio timeout.
        5. Return TranscriptResult with transcript text and WER estimate.

        Args:
            audio: The AudioInput to transcribe.

        Returns:
            TranscriptResult with transcript and estimated word error rate.

        Raises:
            AudioFormatError: If the format is unsupported, the file is too large,
                              or the duration exceeds 5 minutes.
            AudioSNRError: If the computed SNR is below MIN_SNR_DB (20 dB).
            AudioTranscriptionTimeoutError: If the Whisper API call exceeds 3 seconds.
        """
        # ── 1. Format validation ──────────────────────────────────────────────
        fmt = audio.format.lower().lstrip(".")
        if fmt not in SUPPORTED_AUDIO_FORMATS:
            raise AudioFormatError(
                f"Unsupported audio format '{audio.format}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}."
            )

        # ── 2. Size validation ────────────────────────────────────────────────
        size_bytes = len(audio.content)
        if size_bytes > MAX_AUDIO_SIZE_BYTES:
            raise AudioFormatError(
                f"Audio file size {size_bytes} bytes exceeds the maximum allowed "
                f"{MAX_AUDIO_SIZE_BYTES} bytes (10 MB)."
            )

        # ── 3. Load audio and validate duration ───────────────────────────────
        import librosa  # imported here to keep the module importable without librosa installed

        audio_buf = io.BytesIO(audio.content)
        try:
            y, sr = librosa.load(audio_buf, sr=None, mono=True)
        except Exception as exc:
            raise AudioFormatError(f"Failed to decode audio: {exc}") from exc

        duration = librosa.get_duration(y=y, sr=sr)
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise AudioFormatError(
                f"Audio duration {duration:.1f}s exceeds the maximum allowed "
                f"{MAX_AUDIO_DURATION_SECONDS:.0f}s (5 minutes)."
            )

        # ── 4. SNR computation ────────────────────────────────────────────────
        snr_db = self._compute_snr(y)
        if snr_db < MIN_SNR_DB:
            raise AudioSNRError(snr_db)

        # ── 5. Whisper API call with timeout ──────────────────────────────────
        filename = f"audio.{fmt}"
        file_tuple = (filename, audio.content, f"audio/{fmt}")

        try:
            transcription = await asyncio.wait_for(
                self._client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=file_tuple,
                    response_format="text",
                ),
                timeout=WHISPER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise AudioTranscriptionTimeoutError()

        transcript_text: str = transcription if isinstance(transcription, str) else str(transcription)

        # ── 6. WER estimate ───────────────────────────────────────────────────
        # Whisper's average WER on clean audio is ~3–5%. We return a conservative
        # estimate of 0.05 (5%) when SNR is at or above the threshold. The actual
        # WER would require a reference transcript to compute precisely.
        wer_estimate = self._estimate_wer(snr_db)

        return TranscriptResult(
            audio_id=audio.audio_id,
            transcript=transcript_text,
            word_error_rate=wer_estimate,
            snr_db=snr_db,
        )

    async def describe_image(self, image: ImageInput) -> str:
        """Describe an image using GPT-4o Vision API.

        Pipeline (Requirements 9.1, 9.3):
        1. Validate format (JPEG/PNG/WEBP) and size (≤20 MB).
        2. Encode image as base64.
        3. Call GPT-4o Vision API with the image.
        4. Return the textual description string.

        Args:
            image: The ImageInput to describe.

        Returns:
            A plain-text description of the image content.

        Raises:
            ImageFormatError: If the format is unsupported or the file exceeds 20 MB.
        """
        # ── 1. Format validation ──────────────────────────────────────────────
        fmt = image.format.lower().lstrip(".")
        if fmt not in SUPPORTED_IMAGE_FORMATS:
            raise ImageFormatError(
                f"Unsupported image format '{image.format}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}."
            )

        # ── 2. Size validation ────────────────────────────────────────────────
        size_bytes = len(image.content)
        if size_bytes > MAX_IMAGE_SIZE_BYTES:
            raise ImageFormatError(
                f"Image file size {size_bytes} bytes exceeds the maximum allowed "
                f"{MAX_IMAGE_SIZE_BYTES} bytes (20 MB)."
            )

        # ── 3. Encode image as base64 ─────────────────────────────────────────
        import base64  # stdlib — deferred to keep top-level imports clean

        image_b64 = base64.b64encode(image.content).decode("utf-8")

        # Map format to a valid MIME type for the Vision API
        mime_map = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime_type = mime_map[fmt]

        # ── 4. GPT-4o Vision API call ─────────────────────────────────────────
        response = await self._client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe the content of this image in detail. "
                                "Include objects, people, text, colors, and any other relevant details."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("GPT-4o Vision returned empty content for image description.")

        return content

    # ------------------------------------------------------------------
    # Private helpers — audio
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_snr(y: "np.ndarray") -> float:  # type: ignore[name-defined]
        """Compute a simple signal-to-noise ratio estimate in dB.

        Uses the ratio of RMS signal power to estimated noise floor power.
        The noise floor is estimated from the quietest 10% of frames.

        Args:
            y: Mono audio signal as a numpy float32 array.

        Returns:
            SNR in dB. Returns 0.0 if the signal is silent.
        """
        import librosa  # noqa: PLC0415 — deferred import to keep module importable without librosa

        if y.size == 0 or np.max(np.abs(y)) == 0:
            return 0.0

        # Frame the signal into 2048-sample windows
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))

        if rms_per_frame.size == 0:
            # Audio shorter than one frame — use overall RMS
            signal_rms = float(np.sqrt(np.mean(y ** 2)))
            return 0.0 if signal_rms == 0 else 100.0  # treat as clean

        # Noise floor: mean RMS of the quietest 10% of frames
        sorted_rms = np.sort(rms_per_frame)
        noise_frames = max(1, int(len(sorted_rms) * 0.10))
        noise_rms = float(np.mean(sorted_rms[:noise_frames]))

        signal_rms = float(np.mean(rms_per_frame))

        if noise_rms == 0:
            return 100.0  # effectively no noise

        snr = 20.0 * np.log10(signal_rms / noise_rms)
        return float(snr)

    @staticmethod
    def _estimate_wer(snr_db: float) -> float:
        """Return a conservative WER estimate based on SNR.

        Whisper achieves ~3–5% WER on clean audio (SNR ≥ 30 dB) and degrades
        as SNR decreases. We model this with a simple linear interpolation
        between 0.03 (at 30 dB) and 0.05 (at 20 dB).

        Args:
            snr_db: Signal-to-noise ratio in dB (must be ≥ MIN_SNR_DB to reach here).

        Returns:
            Estimated WER in [0.03, 0.05].
        """
        # Clamp to [20, 30] dB range for interpolation
        snr_clamped = max(MIN_SNR_DB, min(30.0, snr_db))
        # Linear interpolation: 0.05 at 20 dB → 0.03 at 30 dB
        wer = 0.05 - (snr_clamped - MIN_SNR_DB) / (30.0 - MIN_SNR_DB) * (0.05 - 0.03)
        return round(wer, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        text: str,
        context: list[ConversationTurn],
    ) -> list[dict[str, str]]:
        """Build the OpenAI messages list, injecting the last MAX_CONTEXT_TURNS turns."""
        messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Requirement 1.5 — include last 50 turns of STM context
        recent_turns = context[-MAX_CONTEXT_TURNS:]
        for turn in recent_turns:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": text})
        return messages

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call GPT-4o with JSON mode and return the raw response string."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content")
        return content

    def _parse_response(self, raw: str, input_id: str) -> NLUResult:
        """Parse the raw JSON string from the LLM into an NLUResult."""
        data = json.loads(raw)

        intents = [
            Intent(
                name=i["name"],
                confidence=float(i["confidence"]),
                parameters=i.get("parameters", {}),
            )
            for i in data["intents"]
        ]
        # Sort intents by confidence descending
        intents.sort(key=lambda x: x.confidence, reverse=True)

        entities = [
            Entity(
                type=e["type"],
                value=e["value"],
                start=int(e["start"]),
                end=int(e["end"]),
            )
            for e in data.get("entities", [])
        ]

        sentiment_score = float(data["sentiment_score"])
        # Clamp to [-1.0, 1.0] in case the LLM drifts slightly
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        language: str = data.get("language", "en")

        return NLUResult(
            input_id=input_id,
            intents=intents,
            entities=entities,
            sentiment_score=sentiment_score,
            language=language,
            is_ambiguous=False,  # will be set by the caller via is_ambiguous()
        )

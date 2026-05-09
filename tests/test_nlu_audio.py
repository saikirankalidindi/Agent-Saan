"""Unit tests for NLU Engine — audio transcription (Task 6).

All tests mock external calls (librosa, OpenAI Whisper API) — no real API calls are made.

Requirements covered: 9.1, 9.2, 9.6
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent_saan.models.nlu import AudioInput, TranscriptResult
from agent_saan.nlu.engine import (
    SUPPORTED_AUDIO_FORMATS,
    MAX_AUDIO_SIZE_BYTES,
    MAX_AUDIO_DURATION_SECONDS,
    MIN_SNR_DB,
    WHISPER_TIMEOUT_SECONDS,
    AudioFormatError,
    AudioSNRError,
    AudioTranscriptionTimeoutError,
    NLUEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now() -> datetime:
    return datetime.now(tz=timezone.utc)


def make_audio_input(
    content: bytes = b"\x00" * 1024,
    fmt: str = "wav",
    audio_id: str = "aud-1",
    session_id: str = "sess-1",
) -> AudioInput:
    return AudioInput(
        audio_id=audio_id,
        session_id=session_id,
        content=content,
        format=fmt,
        timestamp=now(),
    )


def make_mock_openai_client() -> MagicMock:
    """Return a mock AsyncOpenAI client with a stubbed audio.transcriptions.create."""
    mock_client = MagicMock()
    mock_client.audio = MagicMock()
    mock_client.audio.transcriptions = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value="Hello world")
    # Also stub chat completions so NLUEngine.__init__ doesn't fail
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


def make_clean_audio_signal(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    """Generate a clean sine-wave signal with high SNR."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave at amplitude 0.5 — very clean signal
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_noisy_audio_signal(sr: int = 22050, duration: float = 1.0) -> np.ndarray:
    """Generate a signal dominated by noise (low SNR)."""
    rng = np.random.default_rng(42)
    # Pure white noise — signal and noise are indistinguishable → SNR ≈ 0 dB
    return rng.standard_normal(int(sr * duration)).astype(np.float32)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def patch_librosa_load(y: np.ndarray, sr: int = 22050):
    """Context manager that patches librosa.load to return (y, sr)."""
    return patch("librosa.load", return_value=(y, sr))


def patch_librosa_get_duration(duration: float):
    """Context manager that patches librosa.get_duration to return duration."""
    return patch("librosa.get_duration", return_value=duration)


def patch_librosa_util_frame(frames: np.ndarray):
    """Context manager that patches librosa.util.frame to return frames."""
    return patch("librosa.util.frame", return_value=frames)


# ---------------------------------------------------------------------------
# Format and size validation (Requirement 9.1)
# ---------------------------------------------------------------------------


class TestAudioFormatValidation:
    """Requirement 9.1 — validate format (WAV/MP3/OGG) and size (≤10 MB, ≤5 min)."""

    @pytest.mark.asyncio
    async def test_unsupported_format_raises_audio_format_error(self):
        """Non-WAV/MP3/OGG formats must be rejected before any processing."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="flac")

        with pytest.raises(AudioFormatError) as exc_info:
            await engine.transcribe_audio(audio)

        assert "flac" in exc_info.value.reason.lower()
        # Whisper API must NOT have been called
        client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_mp4_format_raises_audio_format_error(self):
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="mp4")

        with pytest.raises(AudioFormatError):
            await engine.transcribe_audio(audio)

    @pytest.mark.asyncio
    async def test_format_with_leading_dot_is_normalised(self):
        """Format strings like '.wav' should be treated the same as 'wav'."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(make_audio_input(fmt=".wav"))

        assert isinstance(result, TranscriptResult)

    @pytest.mark.asyncio
    async def test_oversized_file_raises_audio_format_error(self):
        """Files larger than 10 MB must be rejected."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        # 10 MB + 1 byte
        oversized_content = b"\x00" * (MAX_AUDIO_SIZE_BYTES + 1)
        audio = make_audio_input(content=oversized_content, fmt="wav")

        with pytest.raises(AudioFormatError) as exc_info:
            await engine.transcribe_audio(audio)

        assert "10" in exc_info.value.reason  # mentions the 10 MB limit
        client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_at_exact_size_limit_is_accepted(self):
        """A file of exactly 10 MB should pass the size check."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        exact_content = b"\x00" * MAX_AUDIO_SIZE_BYTES
        audio = make_audio_input(content=exact_content, fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)

    @pytest.mark.asyncio
    async def test_audio_exceeding_5_minutes_raises_audio_format_error(self):
        """Audio longer than 5 minutes must be rejected after loading."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="mp3")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(MAX_AUDIO_DURATION_SECONDS + 1),
        ):
            with pytest.raises(AudioFormatError) as exc_info:
                await engine.transcribe_audio(audio)

        assert "5 minutes" in exc_info.value.reason or "300" in exc_info.value.reason
        client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_audio_at_exact_duration_limit_is_accepted(self):
        """Audio of exactly 5 minutes should pass the duration check."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="ogg")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(MAX_AUDIO_DURATION_SECONDS),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)

    @pytest.mark.parametrize("fmt", sorted(SUPPORTED_AUDIO_FORMATS))
    @pytest.mark.asyncio
    async def test_all_supported_formats_accepted(self, fmt: str):
        """WAV, MP3, and OGG must all be accepted (Requirement 9.1)."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt=fmt)
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)
        assert result.audio_id == "aud-1"


# ---------------------------------------------------------------------------
# SNR rejection (Requirement 9.6)
# ---------------------------------------------------------------------------


class TestSNRRejection:
    """Requirement 9.6 — reject audio with SNR < 20 dB."""

    @pytest.mark.asyncio
    async def test_low_snr_raises_audio_snr_error(self):
        """Audio with SNR below 20 dB must be rejected with AudioSNRError."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=15.0),
        ):
            with pytest.raises(AudioSNRError) as exc_info:
                await engine.transcribe_audio(audio)

        assert exc_info.value.snr_db == pytest.approx(15.0)
        assert exc_info.value.threshold == MIN_SNR_DB
        assert "20" in str(exc_info.value)
        client.audio.transcriptions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_snr_exactly_at_threshold_is_accepted(self):
        """SNR of exactly 20 dB should pass (boundary is inclusive)."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=MIN_SNR_DB),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)
        assert result.snr_db == pytest.approx(MIN_SNR_DB)

    @pytest.mark.asyncio
    async def test_snr_just_below_threshold_is_rejected(self):
        """SNR of 19.9 dB (just below 20 dB) must be rejected."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=19.9),
        ):
            with pytest.raises(AudioSNRError):
                await engine.transcribe_audio(audio)

    @pytest.mark.asyncio
    async def test_high_snr_audio_proceeds_to_transcription(self):
        """Audio with SNR well above 20 dB should reach the Whisper API."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=35.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)
        client.audio.transcriptions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_snr_error_message_prompts_user_to_retry(self):
        """The SNR error message should guide the user to retry or switch to text."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=10.0),
        ):
            with pytest.raises(AudioSNRError) as exc_info:
                await engine.transcribe_audio(audio)

        msg = str(exc_info.value).lower()
        assert "quieter" in msg or "text" in msg


# ---------------------------------------------------------------------------
# Valid audio — happy path (Requirements 9.1, 9.2)
# ---------------------------------------------------------------------------


class TestValidAudioTranscription:
    """Requirements 9.1, 9.2 — valid audio is transcribed and TranscriptResult returned."""

    @pytest.mark.asyncio
    async def test_returns_transcript_result_for_valid_wav(self):
        """A valid WAV file with good SNR should return a TranscriptResult."""
        client = make_mock_openai_client()
        client.audio.transcriptions.create = AsyncMock(return_value="Hello, world!")
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(2.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert isinstance(result, TranscriptResult)
        assert result.audio_id == "aud-1"
        assert result.transcript == "Hello, world!"
        assert 0.0 <= result.word_error_rate <= 1.0
        assert result.snr_db == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_wer_estimate_at_threshold_snr(self):
        """At SNR = 20 dB (threshold), WER estimate should be 0.05 (5%)."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="mp3")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=20.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert result.word_error_rate == pytest.approx(0.05, abs=1e-4)

    @pytest.mark.asyncio
    async def test_wer_estimate_at_high_snr(self):
        """At SNR = 30 dB (clean audio), WER estimate should be 0.03 (3%)."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="ogg")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert result.word_error_rate == pytest.approx(0.03, abs=1e-4)

    @pytest.mark.asyncio
    async def test_wer_meets_5_percent_sla_at_threshold(self):
        """Requirement 9.2: WER must be ≤ 5% for audio at or above the SNR threshold."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=MIN_SNR_DB),
        ):
            result = await engine.transcribe_audio(audio)

        assert result.word_error_rate <= 0.05

    @pytest.mark.asyncio
    async def test_whisper_api_called_with_correct_file_tuple(self):
        """The Whisper API must receive the audio bytes in the expected format."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio_content = b"fake-wav-bytes"
        audio = make_audio_input(content=audio_content, fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            await engine.transcribe_audio(audio)

        call_kwargs = client.audio.transcriptions.create.call_args
        file_arg = call_kwargs.kwargs.get("file") or call_kwargs.args[0]
        # file should be a tuple: (filename, bytes, mime_type)
        assert isinstance(file_arg, tuple)
        assert file_arg[1] == audio_content
        assert "wav" in file_arg[0]

    @pytest.mark.asyncio
    async def test_transcript_result_contains_snr(self):
        """TranscriptResult must include the computed SNR value."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=25.5),
        ):
            result = await engine.transcribe_audio(audio)

        assert result.snr_db == pytest.approx(25.5)


# ---------------------------------------------------------------------------
# Timeout handling (Requirement 9.2)
# ---------------------------------------------------------------------------


class TestWhisperTimeout:
    """Requirement 9.2 — Whisper API call must be limited to 3 seconds."""

    @pytest.mark.asyncio
    async def test_timeout_raises_audio_transcription_timeout_error(self):
        """If Whisper takes longer than 3 seconds, AudioTranscriptionTimeoutError is raised."""
        client = make_mock_openai_client()

        async def slow_transcription(*args, **kwargs):
            await asyncio.sleep(10)  # simulate a very slow API call
            return "too late"

        client.audio.transcriptions.create = slow_transcription
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
            # Speed up the test by patching the timeout constant to a tiny value
            patch("agent_saan.nlu.engine.WHISPER_TIMEOUT_SECONDS", 0.05),
        ):
            with pytest.raises(AudioTranscriptionTimeoutError):
                await engine.transcribe_audio(audio)

    @pytest.mark.asyncio
    async def test_fast_transcription_does_not_timeout(self):
        """A fast Whisper response must not raise a timeout error."""
        client = make_mock_openai_client()
        client.audio.transcriptions.create = AsyncMock(return_value="Quick response")
        engine = NLUEngine(client=client)
        audio = make_audio_input(fmt="wav")
        y = make_clean_audio_signal()

        with (
            patch_librosa_load(y),
            patch_librosa_get_duration(1.0),
            patch.object(NLUEngine, "_compute_snr", return_value=30.0),
        ):
            result = await engine.transcribe_audio(audio)

        assert result.transcript == "Quick response"

    @pytest.mark.asyncio
    async def test_timeout_error_message_mentions_duration(self):
        """The timeout error message should mention the timeout duration."""
        error = AudioTranscriptionTimeoutError()
        assert str(WHISPER_TIMEOUT_SECONDS) in str(error) or "3" in str(error)


# ---------------------------------------------------------------------------
# SNR computation unit tests
# ---------------------------------------------------------------------------


class TestComputeSNR:
    """Unit tests for the _compute_snr static method."""

    def test_speech_like_signal_has_higher_snr_than_pure_noise(self):
        """A signal with distinct loud and quiet frames should have higher SNR than pure noise.

        The SNR algorithm estimates the noise floor from the quietest 10% of frames.
        A realistic speech-like signal (loud bursts + silence) will have a higher
        SNR than pure white noise (where all frames have similar RMS).
        """
        sr = 22050
        rng = np.random.default_rng(0)

        # Build a speech-like signal: alternating loud bursts and near-silence
        # Loud frames: sine wave at amplitude 0.5
        t_loud = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        loud = (0.5 * np.sin(2 * np.pi * 440 * t_loud)).astype(np.float32)
        # Quiet frames: very low-amplitude noise (simulates silence / noise floor)
        quiet = (0.001 * rng.standard_normal(int(sr * 0.5))).astype(np.float32)
        # Interleave: loud, quiet, loud, quiet
        y_speech = np.concatenate([loud, quiet, loud, quiet])

        # Pure noise signal for comparison
        y_noise = rng.standard_normal(len(y_speech)).astype(np.float32) * 0.1

        snr_speech = NLUEngine._compute_snr(y_speech)
        snr_noise = NLUEngine._compute_snr(y_noise)

        # Speech-like signal should have meaningfully higher SNR than pure noise
        assert snr_speech > snr_noise

    def test_empty_signal_returns_zero(self):
        """An empty array should return 0.0 SNR."""
        y = np.array([], dtype=np.float32)
        snr = NLUEngine._compute_snr(y)
        assert snr == pytest.approx(0.0)

    def test_silent_signal_returns_zero(self):
        """An all-zeros signal should return 0.0 SNR."""
        y = np.zeros(22050, dtype=np.float32)
        snr = NLUEngine._compute_snr(y)
        assert snr == pytest.approx(0.0)

    def test_snr_is_float(self):
        """_compute_snr must always return a Python float."""
        y = make_clean_audio_signal()
        snr = NLUEngine._compute_snr(y)
        assert isinstance(snr, float)


# ---------------------------------------------------------------------------
# WER estimate unit tests
# ---------------------------------------------------------------------------


class TestEstimateWER:
    """Unit tests for the _estimate_wer static method."""

    def test_wer_at_20db_is_5_percent(self):
        assert NLUEngine._estimate_wer(20.0) == pytest.approx(0.05, abs=1e-4)

    def test_wer_at_30db_is_3_percent(self):
        assert NLUEngine._estimate_wer(30.0) == pytest.approx(0.03, abs=1e-4)

    def test_wer_above_30db_clamped_to_3_percent(self):
        """SNR above 30 dB should not produce WER below 3%."""
        assert NLUEngine._estimate_wer(50.0) == pytest.approx(0.03, abs=1e-4)

    def test_wer_is_between_3_and_5_percent(self):
        """WER must always be in [0.03, 0.05] for valid SNR values."""
        for snr in [20.0, 22.5, 25.0, 27.5, 30.0]:
            wer = NLUEngine._estimate_wer(snr)
            assert 0.03 <= wer <= 0.05, f"WER {wer} out of range for SNR {snr} dB"

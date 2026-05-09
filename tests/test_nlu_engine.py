"""Unit tests for the NLU Engine (Task 5).

All tests mock the OpenAI client — no real API calls are made.

Requirements covered: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_saan.models.memory import ConversationTurn
from agent_saan.models.nlu import Entity, Intent, NLUResult, UserInput
from agent_saan.nlu.engine import (
    AMBIGUITY_DELTA,
    MAX_CONTEXT_TURNS,
    MAX_INPUT_CHARS,
    NLUEngine,
    NLUInputTooLongError,
    NLUParseError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now() -> datetime:
    return datetime.now(tz=timezone.utc)


def make_user_input(content: str, input_id: str = "inp-1") -> UserInput:
    return UserInput(
        input_id=input_id,
        session_id="sess-1",
        modality="text",
        content=content,
        timestamp=now(),
    )


def make_turn(role: str, content: str, index: int = 0) -> ConversationTurn:
    return ConversationTurn(
        turn_index=index,
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=now(),
    )


def make_llm_response(
    intents: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
    sentiment_score: float = 0.1,
    language: str = "en",
) -> str:
    return json.dumps(
        {
            "intents": intents,
            "entities": entities or [],
            "sentiment_score": sentiment_score,
            "language": language,
        }
    )


def make_mock_client(response_content: str) -> MagicMock:
    """Return a mock AsyncOpenAI client that returns the given content string."""
    mock_message = MagicMock()
    mock_message.content = response_content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_completion)

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create

    return mock_client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestNLUEngineHappyPath:
    """Requirement 1.1 — parse text input, extract intent/entities/sentiment."""

    @pytest.mark.asyncio
    async def test_parse_returns_nlu_result(self):
        llm_response = make_llm_response(
            intents=[{"name": "create_task", "confidence": 0.95, "parameters": {"title": "Buy milk"}}],
            entities=[{"type": "ITEM", "value": "milk", "start": 4, "end": 8}],
            sentiment_score=0.2,
            language="en",
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Buy milk"))

        assert isinstance(result, NLUResult)
        assert result.input_id == "inp-1"
        assert len(result.intents) == 1
        assert result.intents[0].name == "create_task"
        assert result.intents[0].confidence == 0.95
        assert result.intents[0].parameters == {"title": "Buy milk"}
        assert len(result.entities) == 1
        assert result.entities[0].type == "ITEM"
        assert result.entities[0].value == "milk"
        assert result.sentiment_score == pytest.approx(0.2)
        assert result.language == "en"
        assert result.is_ambiguous is False

    @pytest.mark.asyncio
    async def test_intents_sorted_by_confidence_descending(self):
        llm_response = make_llm_response(
            intents=[
                {"name": "search", "confidence": 0.5, "parameters": {}},
                {"name": "create_task", "confidence": 0.9, "parameters": {}},
            ],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Do something"))

        assert result.intents[0].name == "create_task"
        assert result.intents[1].name == "search"

    @pytest.mark.asyncio
    async def test_sentiment_clamped_to_valid_range(self):
        # LLM occasionally returns slightly out-of-range values; engine should clamp them.
        llm_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.8, "parameters": {}}],
            sentiment_score=1.05,  # slightly above 1.0
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hello!"))

        assert result.sentiment_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_empty_entities_list(self):
        llm_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.99, "parameters": {}}],
            entities=[],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hi there"))

        assert result.entities == []


# ---------------------------------------------------------------------------
# Ambiguous intent (Requirement 1.2)
# ---------------------------------------------------------------------------


class TestAmbiguousIntent:
    """Requirement 1.2 — is_ambiguous returns True when top-2 delta ≤ 0.15."""

    def test_is_ambiguous_true_when_delta_at_threshold(self):
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[
                Intent(name="a", confidence=0.80),
                Intent(name="b", confidence=0.65),  # delta = 0.15 exactly
            ],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is True

    def test_is_ambiguous_true_when_delta_below_threshold(self):
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[
                Intent(name="a", confidence=0.80),
                Intent(name="b", confidence=0.70),  # delta = 0.10 < 0.15
            ],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is True

    def test_is_ambiguous_false_when_delta_above_threshold(self):
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[
                Intent(name="a", confidence=0.90),
                Intent(name="b", confidence=0.60),  # delta = 0.30 > 0.15
            ],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is False

    def test_is_ambiguous_false_with_single_intent(self):
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[Intent(name="a", confidence=0.95)],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is False

    def test_is_ambiguous_false_with_no_intents(self):
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is False

    @pytest.mark.asyncio
    async def test_parse_sets_is_ambiguous_flag(self):
        llm_response = make_llm_response(
            intents=[
                {"name": "create_task", "confidence": 0.75, "parameters": {}},
                {"name": "search", "confidence": 0.70, "parameters": {}},  # delta = 0.05
            ],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Do the thing"))

        assert result.is_ambiguous is True


# ---------------------------------------------------------------------------
# Language detection (Requirement 1.3)
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    """Requirement 1.3 — language is detected from LLM response as ISO 639-1 code."""

    @pytest.mark.asyncio
    async def test_english_language_detected(self):
        llm_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.9, "parameters": {}}],
            language="en",
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hello"))

        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_spanish_language_detected(self):
        llm_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.9, "parameters": {}}],
            language="es",
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hola"))

        assert result.language == "es"

    @pytest.mark.asyncio
    async def test_unsupported_language_still_returns_iso_code(self):
        """The NLU engine returns whatever ISO code the LLM detects.
        The Orchestrator layer is responsible for deciding whether the language
        is in the supported set and responding in English if not (Requirement 1.3).
        The engine itself just surfaces the detected code.
        """
        llm_response = make_llm_response(
            intents=[{"name": "unknown", "confidence": 0.5, "parameters": {}}],
            language="sw",  # Swahili — hypothetically unsupported
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Habari yako"))

        assert result.language == "sw"

    @pytest.mark.asyncio
    async def test_language_defaults_to_en_when_missing(self):
        """If the LLM omits the language field, the engine defaults to 'en'."""
        raw = json.dumps(
            {
                "intents": [{"name": "greet", "confidence": 0.9, "parameters": {}}],
                "entities": [],
                "sentiment_score": 0.0,
                # 'language' key intentionally omitted
            }
        )
        client = make_mock_client(raw)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hello"))

        assert result.language == "en"


# ---------------------------------------------------------------------------
# Input too long (Requirement 1.6)
# ---------------------------------------------------------------------------


class TestInputTooLong:
    """Requirement 1.6 — reject inputs > 10,000 chars without calling the LLM."""

    @pytest.mark.asyncio
    async def test_raises_for_input_exceeding_limit(self):
        client = make_mock_client("")  # should never be called
        engine = NLUEngine(client=client)
        long_input = make_user_input("x" * (MAX_INPUT_CHARS + 1))

        with pytest.raises(NLUInputTooLongError) as exc_info:
            await engine.parse(long_input)

        assert exc_info.value.length == MAX_INPUT_CHARS + 1
        assert exc_info.value.limit == MAX_INPUT_CHARS
        # LLM must NOT have been called
        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_input_at_exact_limit(self):
        llm_response = make_llm_response(
            intents=[{"name": "unknown", "confidence": 0.5, "parameters": {}}],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)
        exact_input = make_user_input("a" * MAX_INPUT_CHARS)

        result = await engine.parse(exact_input)

        assert isinstance(result, NLUResult)

    @pytest.mark.asyncio
    async def test_error_contains_limit_info(self):
        engine = NLUEngine(client=MagicMock())
        with pytest.raises(NLUInputTooLongError) as exc_info:
            await engine.parse(make_user_input("y" * 15_000))

        assert "10000" in str(exc_info.value) or exc_info.value.limit == MAX_INPUT_CHARS


# ---------------------------------------------------------------------------
# Parse failure handling (Requirement 1.4)
# ---------------------------------------------------------------------------


class TestParseFailure:
    """Requirement 1.4 — structured error after 2 failed parse attempts."""

    @pytest.mark.asyncio
    async def test_raises_nlu_parse_error_after_two_failures(self):
        """Simulate the LLM returning invalid JSON on both attempts."""
        mock_message = MagicMock()
        mock_message.content = "not valid json {{{"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_completion)
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        engine = NLUEngine(client=mock_client)

        with pytest.raises(NLUParseError) as exc_info:
            await engine.parse(make_user_input("Hello world"))

        error = exc_info.value
        assert "Hello world" in error.input_excerpt
        assert error.failure_reason  # non-empty
        assert error.suggestion  # non-empty
        # Should have been called exactly MAX_PARSE_ATTEMPTS times
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_error_excerpt_truncated_at_200_chars(self):
        mock_message = MagicMock()
        mock_message.content = "bad json"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_completion)
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        engine = NLUEngine(client=mock_client)
        long_text = "a" * 500

        with pytest.raises(NLUParseError) as exc_info:
            await engine.parse(make_user_input(long_text))

        # Excerpt should be truncated to 200 chars + ellipsis
        assert len(exc_info.value.input_excerpt) <= 201 + 1  # 200 chars + "…"

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt_after_first_failure(self):
        """First call returns bad JSON; second call returns valid JSON."""
        good_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.9, "parameters": {}}],
        )

        bad_message = MagicMock()
        bad_message.content = "{{invalid}}"

        good_message = MagicMock()
        good_message.content = good_response

        bad_choice = MagicMock()
        bad_choice.message = bad_message

        good_choice = MagicMock()
        good_choice.message = good_message

        bad_completion = MagicMock()
        bad_completion.choices = [bad_choice]

        good_completion = MagicMock()
        good_completion.choices = [good_choice]

        mock_create = AsyncMock(side_effect=[bad_completion, good_completion])
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        engine = NLUEngine(client=mock_client)
        result = await engine.parse(make_user_input("Hello"))

        assert isinstance(result, NLUResult)
        assert result.intents[0].name == "greet"
        assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Multi-turn context (Requirement 1.5)
# ---------------------------------------------------------------------------


class TestMultiTurnContext:
    """Requirement 1.5 — last 50 STM turns are injected into the LLM prompt."""

    @pytest.mark.asyncio
    async def test_context_turns_included_in_messages(self):
        llm_response = make_llm_response(
            intents=[{"name": "follow_up", "confidence": 0.85, "parameters": {}}],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        turns = [
            make_turn("user", "What is the weather?", i * 2)
            for i in range(3)
        ] + [
            make_turn("assistant", "It is sunny.", i * 2 + 1)
            for i in range(3)
        ]

        await engine.parse(make_user_input("And tomorrow?"), context=turns)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # system + 6 context turns + 1 user message = 8 total
        assert len(messages) == 8

    @pytest.mark.asyncio
    async def test_only_last_50_turns_used(self):
        llm_response = make_llm_response(
            intents=[{"name": "unknown", "confidence": 0.5, "parameters": {}}],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        # Create 60 turns — only the last 50 should be included
        turns = [make_turn("user", f"message {i}", i) for i in range(60)]

        await engine.parse(make_user_input("Final message"), context=turns)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # system + 50 context turns + 1 user message = 52 total
        assert len(messages) == 52

    @pytest.mark.asyncio
    async def test_no_context_still_works(self):
        llm_response = make_llm_response(
            intents=[{"name": "greet", "confidence": 0.9, "parameters": {}}],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Hello"), context=None)

        assert isinstance(result, NLUResult)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        # system + 1 user message = 2 total
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_llm_returns_empty_content_raises_parse_error(self):
        mock_message = MagicMock()
        mock_message.content = None  # empty content

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_create = AsyncMock(return_value=mock_completion)
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        engine = NLUEngine(client=mock_client)

        with pytest.raises(NLUParseError):
            await engine.parse(make_user_input("Hello"))

    @pytest.mark.asyncio
    async def test_negative_sentiment_score(self):
        llm_response = make_llm_response(
            intents=[{"name": "complaint", "confidence": 0.88, "parameters": {}}],
            sentiment_score=-0.9,
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("This is terrible!"))

        assert result.sentiment_score == pytest.approx(-0.9)

    @pytest.mark.asyncio
    async def test_multiple_entities_parsed(self):
        llm_response = make_llm_response(
            intents=[{"name": "schedule_meeting", "confidence": 0.92, "parameters": {}}],
            entities=[
                {"type": "PERSON", "value": "Alice", "start": 5, "end": 10},
                {"type": "DATE", "value": "tomorrow", "start": 14, "end": 22},
            ],
        )
        client = make_mock_client(llm_response)
        engine = NLUEngine(client=client)

        result = await engine.parse(make_user_input("Meet Alice tomorrow"))

        assert len(result.entities) == 2
        types = {e.type for e in result.entities}
        assert "PERSON" in types
        assert "DATE" in types

    def test_is_ambiguous_exactly_at_boundary(self):
        """Delta of exactly AMBIGUITY_DELTA (0.15) should be considered ambiguous."""
        engine = NLUEngine(client=MagicMock())
        result = NLUResult(
            input_id="i",
            intents=[
                Intent(name="a", confidence=1.0),
                Intent(name="b", confidence=1.0 - AMBIGUITY_DELTA),
            ],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=False,
        )
        assert engine.is_ambiguous(result) is True

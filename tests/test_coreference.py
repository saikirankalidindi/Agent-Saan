"""Unit tests for MemoryStore.resolve_coreference (Task 10).

All tests mock the OpenAI client — no real API calls are made.

Requirements covered: 2.5
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_saan.memory.store import COREF_CONFIDENCE_THRESHOLD, MemoryStore
from agent_saan.models.memory import CorefResult, ConversationTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_turn(role: str, content: str, index: int = 0) -> ConversationTurn:
    """Create a ConversationTurn for testing."""
    return ConversationTurn(
        turn_index=index,
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=_now(),
    )


def _make_llm_response(resolved_text: str, confidence: float) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    payload = json.dumps({"resolved_text": resolved_text, "confidence": confidence})

    mock_message = MagicMock()
    mock_message.content = payload

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    return mock_completion


def _make_openai_client(resolved_text: str, confidence: float) -> MagicMock:
    """Return a mock AsyncOpenAI client that returns the given resolution."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_make_llm_response(resolved_text, confidence)
    )
    return client


def _make_store(client: MagicMock) -> MemoryStore:
    return MemoryStore(openai_client=client)


# ---------------------------------------------------------------------------
# Tests: resolved reference  (Requirement 2.5)
# ---------------------------------------------------------------------------


class TestResolvedReference:
    """The LLM resolves the reference with confidence >= 0.7."""

    async def test_returns_resolved_true_when_confidence_above_threshold(self) -> None:
        """resolve_coreference must return resolved=True when confidence >= 0.7."""
        client = _make_openai_client(
            resolved_text="Alice went to the store",
            confidence=0.95,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Alice went to the store", 0)]

        result = await store.resolve_coreference("She went to the store", turns)

        assert isinstance(result, CorefResult)
        assert result.resolved is True
        assert result.clarification_needed is False

    async def test_resolved_text_contains_replacement(self) -> None:
        """resolved_text must contain the referent replacing the pronoun."""
        client = _make_openai_client(
            resolved_text="Alice went to the store",
            confidence=0.90,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Alice went to the store", 0)]

        result = await store.resolve_coreference("She went to the store", turns)

        assert result.resolved_text == "Alice went to the store"

    async def test_confidence_stored_in_result(self) -> None:
        """The confidence value from the LLM must be stored in the result."""
        client = _make_openai_client(
            resolved_text="Bob called the office",
            confidence=0.85,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Bob called the office", 0)]

        result = await store.resolve_coreference("He called the office", turns)

        assert result.confidence == pytest.approx(0.85)

    async def test_resolved_at_exact_threshold(self) -> None:
        """A confidence of exactly 0.7 must be treated as resolved."""
        client = _make_openai_client(
            resolved_text="The project is done",
            confidence=COREF_CONFIDENCE_THRESHOLD,
        )
        store = _make_store(client)
        turns = [_make_turn("assistant", "The project is done", 0)]

        result = await store.resolve_coreference("It is done", turns)

        assert result.resolved is True
        assert result.clarification_needed is False

    async def test_llm_called_with_correct_model(self) -> None:
        """The LLM call must use the model configured on the MemoryStore."""
        client = _make_openai_client(resolved_text="Alice left", confidence=0.9)
        store = MemoryStore(openai_client=client, model="gpt-4o")
        turns = [_make_turn("user", "Alice left", 0)]

        await store.resolve_coreference("She left", turns)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    async def test_llm_called_with_text_in_user_message(self) -> None:
        """The input text must appear in the user message sent to the LLM."""
        client = _make_openai_client(resolved_text="Alice left", confidence=0.9)
        store = _make_store(client)
        turns = [_make_turn("user", "Alice left", 0)]

        await store.resolve_coreference("She left", turns)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message_content = next(
            m["content"] for m in messages if m["role"] == "user"
        )
        assert "She left" in user_message_content

    async def test_conversation_history_included_in_prompt(self) -> None:
        """The STM conversation history must be included in the LLM prompt."""
        client = _make_openai_client(resolved_text="Alice left", confidence=0.9)
        store = _make_store(client)
        turns = [
            _make_turn("user", "Alice arrived at noon", 0),
            _make_turn("assistant", "Noted, Alice is here.", 1),
        ]

        await store.resolve_coreference("She left", turns)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message_content = next(
            m["content"] for m in messages if m["role"] == "user"
        )
        assert "Alice arrived at noon" in user_message_content
        assert "Noted, Alice is here." in user_message_content

    async def test_multiple_turns_all_included(self) -> None:
        """All STM turns must be included in the prompt, not just the last one."""
        client = _make_openai_client(resolved_text="Bob fixed the bug", confidence=0.88)
        store = _make_store(client)
        turns = [
            _make_turn("user", "Bob is the lead developer", 0),
            _make_turn("assistant", "Understood.", 1),
            _make_turn("user", "He fixed the bug yesterday", 2),
        ]

        await store.resolve_coreference("He fixed the bug", turns)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message_content = next(
            m["content"] for m in messages if m["role"] == "user"
        )
        assert "Bob is the lead developer" in user_message_content
        assert "He fixed the bug yesterday" in user_message_content


# ---------------------------------------------------------------------------
# Tests: unresolvable reference  (Requirement 2.5)
# ---------------------------------------------------------------------------


class TestUnresolvableReference:
    """The LLM cannot resolve the reference with sufficient confidence."""

    async def test_returns_resolved_false_when_confidence_below_threshold(self) -> None:
        """resolve_coreference must return resolved=False when confidence < 0.7."""
        client = _make_openai_client(
            resolved_text="They went somewhere",
            confidence=0.50,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Some people went somewhere", 0)]

        result = await store.resolve_coreference("They went there", turns)

        assert result.resolved is False

    async def test_clarification_needed_when_confidence_below_threshold(self) -> None:
        """clarification_needed must be True when confidence < 0.7."""
        client = _make_openai_client(
            resolved_text="They went somewhere",
            confidence=0.40,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Some people went somewhere", 0)]

        result = await store.resolve_coreference("They went there", turns)

        assert result.clarification_needed is True

    async def test_low_confidence_result_preserves_confidence_value(self) -> None:
        """The confidence value must be stored even when resolution fails."""
        client = _make_openai_client(
            resolved_text="They went somewhere",
            confidence=0.30,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Some people went somewhere", 0)]

        result = await store.resolve_coreference("They went there", turns)

        assert result.confidence == pytest.approx(0.30)

    async def test_just_below_threshold_is_unresolved(self) -> None:
        """A confidence just below 0.7 must be treated as unresolved."""
        client = _make_openai_client(
            resolved_text="Someone did something",
            confidence=0.699,
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Someone did something", 0)]

        result = await store.resolve_coreference("They did it", turns)

        assert result.resolved is False
        assert result.clarification_needed is True

    async def test_llm_returns_invalid_json_gives_unresolved(self) -> None:
        """If the LLM returns unparseable JSON, the result must be unresolved."""
        mock_message = MagicMock()
        mock_message.content = "not valid json {{{"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_completion)

        store = _make_store(client)
        turns = [_make_turn("user", "Alice went to the store", 0)]

        result = await store.resolve_coreference("She went there", turns)

        assert result.resolved is False
        assert result.clarification_needed is True

    async def test_llm_returns_empty_content_gives_unresolved(self) -> None:
        """If the LLM returns empty content, the result must be unresolved."""
        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_completion)

        store = _make_store(client)
        turns = [_make_turn("user", "Alice went to the store", 0)]

        result = await store.resolve_coreference("She went there", turns)

        assert result.resolved is False
        assert result.clarification_needed is True

    async def test_llm_exception_gives_unresolved(self) -> None:
        """If the LLM call raises an exception, the result must be unresolved."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        store = _make_store(client)
        turns = [_make_turn("user", "Alice went to the store", 0)]

        result = await store.resolve_coreference("She went there", turns)

        assert result.resolved is False
        assert result.clarification_needed is True
        assert result.confidence == pytest.approx(0.0)

    async def test_confidence_clamped_above_one(self) -> None:
        """Confidence values above 1.0 from the LLM must be clamped to 1.0."""
        client = _make_openai_client(
            resolved_text="Alice left",
            confidence=1.5,  # out-of-range value from LLM
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Alice left", 0)]

        result = await store.resolve_coreference("She left", turns)

        assert result.confidence <= 1.0
        assert result.resolved is True  # clamped to 1.0 >= 0.7

    async def test_confidence_clamped_below_zero(self) -> None:
        """Confidence values below 0.0 from the LLM must be clamped to 0.0."""
        client = _make_openai_client(
            resolved_text="Someone left",
            confidence=-0.5,  # out-of-range value from LLM
        )
        store = _make_store(client)
        turns = [_make_turn("user", "Alice left", 0)]

        result = await store.resolve_coreference("She left", turns)

        assert result.confidence == pytest.approx(0.0)
        assert result.resolved is False
        assert result.clarification_needed is True


# ---------------------------------------------------------------------------
# Tests: empty context  (Requirement 2.5)
# ---------------------------------------------------------------------------


class TestEmptyContext:
    """No STM conversation history is available."""

    async def test_empty_context_with_low_confidence_gives_unresolved(self) -> None:
        """With no context, the LLM should return low confidence → unresolved."""
        client = _make_openai_client(
            resolved_text="She went to the store",
            confidence=0.10,
        )
        store = _make_store(client)

        result = await store.resolve_coreference("She went to the store", [])

        assert result.resolved is False
        assert result.clarification_needed is True

    async def test_empty_context_prompt_indicates_no_prior_context(self) -> None:
        """The LLM prompt must indicate there is no prior context when STM is empty."""
        client = _make_openai_client(
            resolved_text="She went to the store",
            confidence=0.10,
        )
        store = _make_store(client)

        await store.resolve_coreference("She went to the store", [])

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message_content = next(
            m["content"] for m in messages if m["role"] == "user"
        )
        assert "no prior context" in user_message_content.lower()

    async def test_empty_context_llm_still_called(self) -> None:
        """The LLM must still be called even when STM context is empty."""
        client = _make_openai_client(
            resolved_text="She went to the store",
            confidence=0.10,
        )
        store = _make_store(client)

        await store.resolve_coreference("She went to the store", [])

        client.chat.completions.create.assert_awaited_once()

    async def test_empty_context_high_confidence_still_resolves(self) -> None:
        """If the LLM somehow returns high confidence with empty context, accept it."""
        client = _make_openai_client(
            resolved_text="The system is ready",
            confidence=0.80,
        )
        store = _make_store(client)

        result = await store.resolve_coreference("It is ready", [])

        assert result.resolved is True
        assert result.resolved_text == "The system is ready"

    async def test_empty_list_vs_none_context_both_work(self) -> None:
        """An empty list must be accepted without error."""
        client = _make_openai_client(
            resolved_text="Something happened",
            confidence=0.20,
        )
        store = _make_store(client)

        # Should not raise
        result = await store.resolve_coreference("It happened", [])

        assert isinstance(result, CorefResult)


# ---------------------------------------------------------------------------
# Tests: CorefResult model validation
# ---------------------------------------------------------------------------


class TestCorefResultModel:
    """Validate the CorefResult Pydantic model itself."""

    def test_default_values(self) -> None:
        """CorefResult must have sensible defaults."""
        result = CorefResult(resolved=True)
        assert result.resolved_text == ""
        assert result.confidence == pytest.approx(0.0)
        assert result.clarification_needed is False

    def test_unresolved_defaults(self) -> None:
        """An unresolved result with clarification_needed=True must be constructable."""
        result = CorefResult(resolved=False, clarification_needed=True)
        assert result.resolved is False
        assert result.clarification_needed is True

    def test_full_resolved_result(self) -> None:
        """A fully populated resolved result must round-trip through Pydantic."""
        result = CorefResult(
            resolved=True,
            resolved_text="Alice went to the store",
            confidence=0.95,
            clarification_needed=False,
        )
        assert result.resolved is True
        assert result.resolved_text == "Alice went to the store"
        assert result.confidence == pytest.approx(0.95)
        assert result.clarification_needed is False

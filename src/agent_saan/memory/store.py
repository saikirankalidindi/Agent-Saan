"""MemoryStore — high-level memory facade with co-reference resolution.

Combines ShortTermMemory and LongTermMemory and adds higher-level operations
such as co-reference resolution that require an LLM call.

Requirements: 2.5
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from agent_saan.models.memory import CorefResult, ConversationTurn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum confidence required to consider a co-reference resolved.
COREF_CONFIDENCE_THRESHOLD = 0.7

# System prompt used for the co-reference resolution LLM call.
_SYSTEM_PROMPT = """\
You are a co-reference resolution assistant.
Given a piece of text and a conversation history, replace every pronoun and
entity reference in the text with the specific noun or entity it refers to,
using the conversation history as context.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
  "resolved_text": "<text with references replaced>",
  "confidence": <float between 0.0 and 1.0>
}

Rules:
- "resolved_text" must be the input text with all resolvable pronouns/references
  replaced by their referents from the conversation history.
- "confidence" must reflect how certain you are that every reference was correctly
  resolved. Use 0.0 when the context is empty or the reference is completely
  ambiguous. Use 1.0 only when every reference is unambiguous.
- If a reference cannot be resolved, leave it unchanged in "resolved_text" and
  lower the confidence accordingly.
"""


class MemoryStore:
    """High-level memory facade that adds co-reference resolution on top of
    the lower-level ``ShortTermMemory`` and ``LongTermMemory`` classes.

    Parameters
    ----------
    openai_client:
        An ``openai.AsyncOpenAI`` client used for the LLM co-reference call.
    model:
        The OpenAI chat model to use (default: ``"gpt-4o"``).
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        *,
        model: str = "gpt-4o",
    ) -> None:
        self._openai = openai_client
        self._model = model

    # ------------------------------------------------------------------
    # Co-reference resolution  (Requirement 2.5)
    # ------------------------------------------------------------------

    async def resolve_coreference(
        self,
        text: str,
        stm_context: list[ConversationTurn],
    ) -> CorefResult:
        """Resolve pronoun and entity references in *text* using STM context.

        Uses an LLM call to replace pronouns and entity references with their
        referents found in *stm_context*.

        If the LLM's confidence is below ``COREF_CONFIDENCE_THRESHOLD`` (0.7),
        returns ``CorefResult(resolved=False, clarification_needed=True)`` so
        that the Orchestrator can ask the user to clarify the reference.

        Parameters
        ----------
        text:
            The input text that may contain pronouns or entity references.
        stm_context:
            The Short_Term_Memory conversation history for the current session.
            An empty list means there is no prior context.

        Returns
        -------
        CorefResult
            ``resolved=True`` with the rewritten text when confidence >= 0.7.
            ``resolved=False, clarification_needed=True`` otherwise.

        Requirements: 2.5
        """
        # Build a compact representation of the conversation history.
        history_lines: list[str] = []
        for turn in stm_context:
            history_lines.append(f"{turn.role.upper()}: {turn.content}")
        history_text = "\n".join(history_lines) if history_lines else "(no prior context)"

        user_message = (
            f"Conversation history:\n{history_text}\n\n"
            f"Text to resolve:\n{text}"
        )

        try:
            response = await self._openai.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )
        except Exception:
            logger.exception("LLM call failed during co-reference resolution")
            return CorefResult(
                resolved=False,
                clarification_needed=True,
                confidence=0.0,
            )

        raw_content: str | None = response.choices[0].message.content
        if not raw_content:
            logger.warning("LLM returned empty content for co-reference resolution")
            return CorefResult(
                resolved=False,
                clarification_needed=True,
                confidence=0.0,
            )

        try:
            data = json.loads(raw_content)
            resolved_text: str = str(data.get("resolved_text", text))
            confidence: float = float(data.get("confidence", 0.0))
            # Clamp confidence to [0.0, 1.0].
            confidence = max(0.0, min(1.0, confidence))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(
                "Failed to parse LLM co-reference response: %r", raw_content[:200]
            )
            return CorefResult(
                resolved=False,
                clarification_needed=True,
                confidence=0.0,
            )

        if confidence < COREF_CONFIDENCE_THRESHOLD:
            return CorefResult(
                resolved=False,
                resolved_text=resolved_text,
                confidence=confidence,
                clarification_needed=True,
            )

        return CorefResult(
            resolved=True,
            resolved_text=resolved_text,
            confidence=confidence,
            clarification_needed=False,
        )

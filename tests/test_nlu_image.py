"""Unit tests for NLU Engine — image description (Task 7).

All tests mock external calls (OpenAI GPT-4o Vision API) — no real API calls are made.

Requirements covered: 9.1, 9.3
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_saan.models.nlu import ImageInput
from agent_saan.nlu.engine import (
    SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_SIZE_BYTES,
    ImageFormatError,
    NLUEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now() -> datetime:
    return datetime.now(tz=timezone.utc)


def make_image_input(
    content: bytes = b"\xff\xd8\xff" + b"\x00" * 1024,  # minimal JPEG-like bytes
    fmt: str = "jpeg",
    image_id: str = "img-1",
    session_id: str = "sess-1",
) -> ImageInput:
    return ImageInput(
        image_id=image_id,
        session_id=session_id,
        content=content,
        format=fmt,
        timestamp=now(),
    )


def make_mock_openai_client(description: str = "A photo of a cat sitting on a mat.") -> MagicMock:
    """Return a mock AsyncOpenAI client with a stubbed chat.completions.create."""
    mock_client = MagicMock()

    # Stub vision response
    mock_message = MagicMock()
    mock_message.content = description
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Also stub audio transcriptions so NLUEngine.__init__ doesn't fail
    mock_client.audio = MagicMock()
    mock_client.audio.transcriptions = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock()

    return mock_client


# ---------------------------------------------------------------------------
# Format and size validation (Requirement 9.1)
# ---------------------------------------------------------------------------


class TestImageFormatValidation:
    """Requirement 9.1 — validate format (JPEG/PNG/WEBP) and size (≤20 MB)."""

    @pytest.mark.asyncio
    async def test_unsupported_format_raises_image_format_error(self):
        """Non-JPEG/PNG/WEBP formats must be rejected before any API call."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="gif")

        with pytest.raises(ImageFormatError) as exc_info:
            await engine.describe_image(image)

        assert "gif" in exc_info.value.reason.lower()
        # Vision API must NOT have been called
        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_bmp_format_raises_image_format_error(self):
        """BMP format must be rejected."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="bmp")

        with pytest.raises(ImageFormatError):
            await engine.describe_image(image)

        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_tiff_format_raises_image_format_error(self):
        """TIFF format must be rejected."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="tiff")

        with pytest.raises(ImageFormatError):
            await engine.describe_image(image)

        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_format_with_leading_dot_is_normalised(self):
        """Format strings like '.jpeg' should be treated the same as 'jpeg'."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt=".jpeg")

        result = await engine.describe_image(image)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_jpg_alias_is_accepted(self):
        """'jpg' must be accepted as an alias for 'jpeg'."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="jpg")

        result = await engine.describe_image(image)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_oversized_image_raises_image_format_error(self):
        """Images larger than 20 MB must be rejected."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        # 20 MB + 1 byte
        oversized_content = b"\x00" * (MAX_IMAGE_SIZE_BYTES + 1)
        image = make_image_input(content=oversized_content, fmt="png")

        with pytest.raises(ImageFormatError) as exc_info:
            await engine.describe_image(image)

        assert "20" in exc_info.value.reason  # mentions the 20 MB limit
        client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_image_at_exact_size_limit_is_accepted(self):
        """An image of exactly 20 MB should pass the size check."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        exact_content = b"\x00" * MAX_IMAGE_SIZE_BYTES
        image = make_image_input(content=exact_content, fmt="jpeg")

        result = await engine.describe_image(image)

        assert isinstance(result, str)

    @pytest.mark.parametrize("fmt", sorted(SUPPORTED_IMAGE_FORMATS))
    @pytest.mark.asyncio
    async def test_all_supported_formats_accepted(self, fmt: str):
        """JPEG, JPG, PNG, and WEBP must all be accepted (Requirement 9.1)."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt=fmt)

        result = await engine.describe_image(image)

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Valid image — happy path (Requirements 9.1, 9.3)
# ---------------------------------------------------------------------------


class TestValidImageDescription:
    """Requirements 9.1, 9.3 — valid image is described and a string is returned."""

    @pytest.mark.asyncio
    async def test_returns_string_description_for_valid_jpeg(self):
        """A valid JPEG image should return a non-empty description string."""
        expected_description = "A golden retriever playing in a park on a sunny day."
        client = make_mock_openai_client(description=expected_description)
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="jpeg")

        result = await engine.describe_image(image)

        assert isinstance(result, str)
        assert result == expected_description

    @pytest.mark.asyncio
    async def test_returns_string_description_for_valid_png(self):
        """A valid PNG image should return a non-empty description string."""
        expected_description = "A bar chart showing monthly sales data."
        client = make_mock_openai_client(description=expected_description)
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="png")

        result = await engine.describe_image(image)

        assert result == expected_description

    @pytest.mark.asyncio
    async def test_returns_string_description_for_valid_webp(self):
        """A valid WEBP image should return a non-empty description string."""
        expected_description = "A screenshot of a web page with a navigation menu."
        client = make_mock_openai_client(description=expected_description)
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="webp")

        result = await engine.describe_image(image)

        assert result == expected_description

    @pytest.mark.asyncio
    async def test_vision_api_called_with_base64_encoded_image(self):
        """The Vision API must receive the image as a base64 data URL."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image_content = b"\xff\xd8\xff\xe0" + b"\xab" * 100  # fake JPEG bytes
        image = make_image_input(content=image_content, fmt="jpeg")

        await engine.describe_image(image)

        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        assert len(messages) == 1
        user_message = messages[0]
        assert user_message["role"] == "user"

        # Find the image_url content block
        content_blocks = user_message["content"]
        image_blocks = [b for b in content_blocks if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

        image_url = image_blocks[0]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg;base64,")

        # Verify the base64 payload decodes back to the original bytes
        b64_payload = image_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_payload)
        assert decoded == image_content

    @pytest.mark.asyncio
    async def test_vision_api_called_with_correct_mime_type_for_png(self):
        """PNG images must use 'image/png' MIME type in the data URL."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="png")

        await engine.describe_image(image)

        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        content_blocks = messages[0]["content"]
        image_blocks = [b for b in content_blocks if b.get("type") == "image_url"]
        image_url = image_blocks[0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_vision_api_called_with_correct_mime_type_for_webp(self):
        """WEBP images must use 'image/webp' MIME type in the data URL."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="webp")

        await engine.describe_image(image)

        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        content_blocks = messages[0]["content"]
        image_blocks = [b for b in content_blocks if b.get("type") == "image_url"]
        image_url = image_blocks[0]["image_url"]["url"]
        assert image_url.startswith("data:image/webp;base64,")

    @pytest.mark.asyncio
    async def test_vision_api_called_once_per_describe_image_call(self):
        """The Vision API must be called exactly once per describe_image() invocation."""
        client = make_mock_openai_client()
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="jpeg")

        await engine.describe_image(image)

        client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_description_is_plain_string_not_bytes(self):
        """The returned description must be a plain str, not bytes."""
        client = make_mock_openai_client(description="A red apple on a wooden table.")
        engine = NLUEngine(client=client)
        image = make_image_input(fmt="jpeg")

        result = await engine.describe_image(image)

        assert type(result) is str  # noqa: E721 — strict type check, not isinstance


# ---------------------------------------------------------------------------
# Description injection into NLU parse context (Requirement 9.3)
# ---------------------------------------------------------------------------


class TestDescriptionInjectionIntoParseContext:
    """Requirement 9.3 — description is injected as a synthetic text prefix into NLU parse."""

    @pytest.mark.asyncio
    async def test_description_can_be_used_as_text_prefix_for_parse(self):
        """The description string returned by describe_image() can be prepended to
        user text and passed to parse() as a synthetic text input.

        This test verifies the integration contract: describe_image() returns a
        plain string that is suitable for injection into the NLU parse context.
        """
        description = "A screenshot showing an error message: 'Connection refused'."
        client = make_mock_openai_client(description=description)

        # Also stub the NLU parse response
        nlu_response_json = (
            '{"intents": [{"name": "report_error", "confidence": 0.9, "parameters": {}}], '
            '"entities": [], "sentiment_score": -0.3, "language": "en"}'
        )
        mock_nlu_message = MagicMock()
        mock_nlu_message.content = nlu_response_json
        mock_nlu_choice = MagicMock()
        mock_nlu_choice.message = mock_nlu_message
        mock_nlu_response = MagicMock()
        mock_nlu_response.choices = [mock_nlu_choice]

        # The client's chat.completions.create is called for both describe_image
        # and parse — we return the vision description first, then the NLU JSON.
        client.chat.completions.create = AsyncMock(
            side_effect=[
                # First call: describe_image → return vision mock response
                client.chat.completions.create.return_value,
                # Second call: parse → return NLU JSON mock response
                mock_nlu_response,
            ]
        )
        # Reset the side_effect to use the vision mock for the first call
        vision_response = MagicMock()
        vision_response.choices = [MagicMock()]
        vision_response.choices[0].message.content = description
        client.chat.completions.create = AsyncMock(
            side_effect=[vision_response, mock_nlu_response]
        )

        engine = NLUEngine(client=client)
        image = make_image_input(fmt="jpeg")

        # Step 1: get description
        desc = await engine.describe_image(image)
        assert desc == description

        # Step 2: inject description as prefix into a UserInput and parse
        from agent_saan.models.nlu import UserInput

        synthetic_text = f"[Image description: {desc}]"
        user_input = UserInput(
            input_id="inp-1",
            session_id="sess-1",
            modality="text",
            content=synthetic_text,
            timestamp=now(),
        )
        nlu_result = await engine.parse(user_input)

        assert nlu_result.intents[0].name == "report_error"
        assert nlu_result.language == "en"

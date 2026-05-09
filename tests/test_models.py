"""Unit tests for all shared Pydantic data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_saan.models import (
    Action,
    ActionResult,
    ActionSchema,
    AuditLogEntry,
    BusEvent,
    ConversationTurn,
    Entity,
    GuardrailDecision,
    Intent,
    MemoryEntry,
    Message,
    NLUResult,
    ObservabilityEvent,
    PluginManifest,
    Session,
    Suggestion,
    Task,
    UserInput,
    UserPreferences,
)


def now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ===========================================================================
# Session models
# ===========================================================================


class TestSession:
    def test_valid_session(self):
        s = Session(
            session_id="sess-1",
            user_id="user-1",
            state="idle",
            created_at=now(),
            last_active=now(),
        )
        assert s.session_id == "sess-1"
        assert s.state == "idle"
        assert s.input_queue == []

    def test_all_valid_states(self):
        for state in ("idle", "listening", "processing", "acting", "responding"):
            s = Session(
                session_id="s",
                user_id="u",
                state=state,
                created_at=now(),
                last_active=now(),
            )
            assert s.state == state

    def test_invalid_state_raises(self):
        with pytest.raises(ValidationError):
            Session(
                session_id="s",
                user_id="u",
                state="unknown",
                created_at=now(),
                last_active=now(),
            )

    def test_serialization_roundtrip(self):
        s = Session(
            session_id="sess-2",
            user_id="user-2",
            state="processing",
            created_at=now(),
            last_active=now(),
        )
        data = s.model_dump()
        s2 = Session.model_validate(data)
        assert s2.session_id == s.session_id
        assert s2.state == s.state


class TestMessage:
    def test_valid_message(self):
        m = Message(
            message_id="msg-1",
            session_id="sess-1",
            role="user",
            content="Hello",
            timestamp=now(),
        )
        assert m.role == "user"
        assert m.content == "Hello"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(
                message_id="msg-2",
                session_id="sess-1",
                role="system",
                content="Hi",
                timestamp=now(),
            )

    def test_assistant_role(self):
        m = Message(
            message_id="msg-3",
            session_id="sess-1",
            role="assistant",
            content="How can I help?",
            timestamp=now(),
        )
        assert m.role == "assistant"


# ===========================================================================
# NLU models
# ===========================================================================


class TestUserInput:
    def test_text_modality(self):
        ui = UserInput(
            input_id="inp-1",
            session_id="sess-1",
            modality="text",
            content="What is the weather?",
            timestamp=now(),
        )
        assert ui.modality == "text"

    def test_audio_modality_bytes(self):
        ui = UserInput(
            input_id="inp-2",
            session_id="sess-1",
            modality="audio",
            content=b"\x00\x01\x02",
            timestamp=now(),
        )
        assert ui.modality == "audio"

    def test_invalid_modality_raises(self):
        with pytest.raises(ValidationError):
            UserInput(
                input_id="inp-3",
                session_id="sess-1",
                modality="video",
                content="data",
                timestamp=now(),
            )


class TestIntent:
    def test_valid_intent(self):
        i = Intent(name="create_task", confidence=0.95, parameters={"title": "Buy milk"})
        assert i.name == "create_task"
        assert i.confidence == 0.95

    def test_confidence_bounds(self):
        Intent(name="x", confidence=0.0)
        Intent(name="x", confidence=1.0)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            Intent(name="x", confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            Intent(name="x", confidence=1.1)

    def test_default_parameters(self):
        i = Intent(name="greet", confidence=0.8)
        assert i.parameters == {}


class TestEntity:
    def test_valid_entity(self):
        e = Entity(type="DATE", value="tomorrow", start=10, end=18)
        assert e.type == "DATE"
        assert e.start == 10

    def test_negative_start_raises(self):
        with pytest.raises(ValidationError):
            Entity(type="DATE", value="x", start=-1, end=5)


class TestNLUResult:
    def test_valid_result(self):
        intent = Intent(name="search", confidence=0.9)
        result = NLUResult(
            input_id="inp-1",
            intents=[intent],
            sentiment_score=0.2,
            language="en",
            is_ambiguous=False,
        )
        assert result.language == "en"
        assert not result.is_ambiguous

    def test_sentiment_bounds(self):
        NLUResult(
            input_id="i",
            intents=[],
            sentiment_score=-1.0,
            language="en",
            is_ambiguous=False,
        )
        NLUResult(
            input_id="i",
            intents=[],
            sentiment_score=1.0,
            language="en",
            is_ambiguous=False,
        )

    def test_sentiment_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            NLUResult(
                input_id="i",
                intents=[],
                sentiment_score=1.5,
                language="en",
                is_ambiguous=False,
            )

    def test_ambiguous_flag(self):
        r = NLUResult(
            input_id="i",
            intents=[
                Intent(name="a", confidence=0.6),
                Intent(name="b", confidence=0.55),
            ],
            sentiment_score=0.0,
            language="en",
            is_ambiguous=True,
        )
        assert r.is_ambiguous


# ===========================================================================
# Memory models
# ===========================================================================


class TestMemoryEntry:
    def test_valid_entry(self):
        entry = MemoryEntry(
            entry_id="mem-1",
            user_id="user-1",
            namespace="domain_facts",
            content="The capital of France is Paris.",
            source_type="user_stated",
            embedding=[0.1, 0.2, 0.3],
            created_at=now(),
        )
        assert entry.namespace == "domain_facts"
        assert entry.source_type == "user_stated"
        assert entry.tags == []

    def test_invalid_namespace_raises(self):
        with pytest.raises(ValidationError):
            MemoryEntry(
                entry_id="mem-2",
                user_id="user-1",
                namespace="unknown_ns",
                content="x",
                source_type="imported",
                created_at=now(),
            )

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValidationError):
            MemoryEntry(
                entry_id="mem-3",
                user_id="user-1",
                namespace="domain_facts",
                content="x",
                source_type="manual",
                created_at=now(),
            )

    def test_all_namespaces(self):
        for ns in ("domain_facts", "user_preferences", "learned_patterns"):
            e = MemoryEntry(
                entry_id="m",
                user_id="u",
                namespace=ns,
                content="c",
                source_type="imported",
                created_at=now(),
            )
            assert e.namespace == ns

    def test_tags_field(self):
        e = MemoryEntry(
            entry_id="m",
            user_id="u",
            namespace="domain_facts",
            content="c",
            source_type="imported",
            created_at=now(),
            tags=["science", "geography"],
        )
        assert "science" in e.tags


class TestConversationTurn:
    def test_valid_turn(self):
        t = ConversationTurn(
            turn_index=0,
            role="user",
            content="Hello",
            timestamp=now(),
        )
        assert t.turn_index == 0
        assert t.entities == []

    def test_negative_turn_index_raises(self):
        with pytest.raises(ValidationError):
            ConversationTurn(
                turn_index=-1,
                role="user",
                content="Hi",
                timestamp=now(),
            )

    def test_with_entities(self):
        e = Entity(type="PERSON", value="Alice", start=0, end=5)
        t = ConversationTurn(
            turn_index=1,
            role="assistant",
            content="Hello Alice",
            timestamp=now(),
            entities=[e],
        )
        assert len(t.entities) == 1
        assert t.entities[0].value == "Alice"


# ===========================================================================
# Task models
# ===========================================================================


class TestTask:
    def test_valid_task(self):
        t = Task(
            task_id="task-1",
            user_id="user-1",
            title="Submit report",
            description="Submit Q3 report",
            priority="high",
            deadline=now(),
            status="pending",
            created_at=now(),
        )
        assert t.priority == "high"
        assert t.status == "pending"
        assert t.completion_source is None

    def test_default_priority_is_medium(self):
        t = Task(
            task_id="task-2",
            user_id="user-1",
            title="Do something",
            description="desc",
            deadline=now(),
            status="pending",
            created_at=now(),
        )
        assert t.priority == "medium"

    def test_invalid_priority_raises(self):
        with pytest.raises(ValidationError):
            Task(
                task_id="task-3",
                user_id="user-1",
                title="x",
                description="y",
                priority="critical",
                deadline=now(),
                status="pending",
                created_at=now(),
            )

    def test_all_statuses(self):
        for status in ("pending", "in_progress", "completed", "pending_authorization", "dismissed"):
            t = Task(
                task_id="t",
                user_id="u",
                title="x",
                description="y",
                deadline=now(),
                status=status,
                created_at=now(),
            )
            assert t.status == status

    def test_completed_task_fields(self):
        t = Task(
            task_id="task-4",
            user_id="user-1",
            title="Done task",
            description="desc",
            deadline=now(),
            status="completed",
            completion_source="agent",
            completed_at=now(),
            created_at=now(),
        )
        assert t.completion_source == "agent"
        assert t.completed_at is not None

    def test_recurrence_field(self):
        t = Task(
            task_id="task-5",
            user_id="user-1",
            title="Daily standup",
            description="desc",
            deadline=now(),
            status="pending",
            recurrence="daily",
            created_at=now(),
        )
        assert t.recurrence == "daily"

    def test_serialization_roundtrip(self):
        t = Task(
            task_id="task-6",
            user_id="user-1",
            title="Test",
            description="desc",
            deadline=now(),
            status="pending",
            created_at=now(),
        )
        data = t.model_dump()
        t2 = Task.model_validate(data)
        assert t2.task_id == t.task_id


# ===========================================================================
# Suggestion models
# ===========================================================================


class TestSuggestion:
    def test_valid_suggestion(self):
        s = Suggestion(
            suggestion_id="sug-1",
            session_id="sess-1",
            content="You might want to check your calendar.",
            confidence=0.75,
            rationale="Meeting in 30 minutes",
            category="calendar",
            topic="upcoming_events",
            created_at=now(),
        )
        assert s.confidence == 0.75
        assert len(s.rationale) <= 100

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            Suggestion(
                suggestion_id="s",
                session_id="sess",
                content="x",
                confidence=-0.1,
                rationale="r",
                category="c",
                topic="t",
                created_at=now(),
            )

    def test_rationale_too_long_raises(self):
        with pytest.raises(ValidationError):
            Suggestion(
                suggestion_id="s",
                session_id="sess",
                content="x",
                confidence=0.7,
                rationale="x" * 101,
                category="c",
                topic="t",
                created_at=now(),
            )

    def test_rationale_exactly_100_chars(self):
        s = Suggestion(
            suggestion_id="s",
            session_id="sess",
            content="x",
            confidence=0.7,
            rationale="x" * 100,
            category="c",
            topic="t",
            created_at=now(),
        )
        assert len(s.rationale) == 100


# ===========================================================================
# Plugin models
# ===========================================================================


class TestActionSchema:
    def test_valid_schema(self):
        a = ActionSchema(
            name="send_email",
            description="Send an email",
            parameters={"to": {"type": "string"}, "body": {"type": "string"}},
            is_reversible=False,
        )
        assert a.name == "send_email"
        assert not a.is_reversible

    def test_default_parameters(self):
        a = ActionSchema(name="list_events", description="List events", is_reversible=True)
        assert a.parameters == {}


class TestPluginManifest:
    def test_valid_manifest(self):
        schema = ActionSchema(name="search", description="Web search", is_reversible=True)
        m = PluginManifest(
            name="WebSearchPlugin",
            version="1.0.0",
            description="Searches the web",
            permissions=["internet"],
            actions=[schema],
            timeout_seconds=15,
        )
        assert m.name == "WebSearchPlugin"
        assert m.timeout_seconds == 15

    def test_default_timeout(self):
        m = PluginManifest(
            name="TestPlugin",
            version="0.1.0",
            description="Test",
        )
        assert m.timeout_seconds == 10

    def test_timeout_below_min_raises(self):
        with pytest.raises(ValidationError):
            PluginManifest(
                name="P",
                version="1.0.0",
                description="d",
                timeout_seconds=0,
            )

    def test_timeout_above_max_raises(self):
        with pytest.raises(ValidationError):
            PluginManifest(
                name="P",
                version="1.0.0",
                description="d",
                timeout_seconds=61,
            )

    def test_timeout_boundary_values(self):
        for t in (1, 60):
            m = PluginManifest(name="P", version="1.0.0", description="d", timeout_seconds=t)
            assert m.timeout_seconds == t


class TestAction:
    def test_valid_action(self):
        a = Action(
            action_id="act-1",
            session_id="sess-1",
            plugin_name="CalendarPlugin",
            action_name="list_events",
            parameters={"date": "2024-01-01"},
            is_reversible=True,
            dispatched_at=now(),
        )
        assert a.plugin_name == "CalendarPlugin"
        assert a.is_reversible

    def test_irreversible_action(self):
        a = Action(
            action_id="act-2",
            session_id="sess-1",
            plugin_name="EmailPlugin",
            action_name="send_email",
            parameters={},
            is_reversible=False,
            dispatched_at=now(),
        )
        assert not a.is_reversible


class TestActionResult:
    def test_success_result(self):
        r = ActionResult(
            action_id="act-1",
            status="success",
            result={"events": []},
            completed_at=now(),
        )
        assert r.status == "success"
        assert r.error is None

    def test_error_result(self):
        r = ActionResult(
            action_id="act-2",
            status="error",
            error="Connection refused",
            completed_at=now(),
        )
        assert r.status == "error"
        assert r.result is None

    def test_timeout_result(self):
        r = ActionResult(
            action_id="act-3",
            status="timeout",
            completed_at=now(),
        )
        assert r.status == "timeout"

    def test_all_statuses(self):
        for status in ("success", "error", "timeout", "cancelled", "blocked"):
            r = ActionResult(action_id="a", status=status, completed_at=now())
            assert r.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            ActionResult(action_id="a", status="pending", completed_at=now())


# ===========================================================================
# Safety models
# ===========================================================================


class TestAuditLogEntry:
    def test_valid_entry(self):
        e = AuditLogEntry(
            log_id="log-1",
            action_id="act-1",
            user_id="user-1",
            session_id="sess-1",
            outcome="allowed",
            timestamp=now(),
        )
        assert e.outcome == "allowed"
        assert e.guardrail_rule is None

    def test_blocked_with_rule(self):
        e = AuditLogEntry(
            log_id="log-2",
            action_id="act-2",
            user_id="user-1",
            session_id="sess-1",
            outcome="blocked",
            guardrail_rule="no_delete_files",
            timestamp=now(),
        )
        assert e.guardrail_rule == "no_delete_files"

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValidationError):
            AuditLogEntry(
                log_id="log-3",
                action_id="act-3",
                user_id="user-1",
                session_id="sess-1",
                outcome="denied",
                timestamp=now(),
            )

    def test_all_outcomes(self):
        for outcome in ("allowed", "blocked", "cancelled"):
            e = AuditLogEntry(
                log_id="l",
                action_id="a",
                user_id="u",
                session_id="s",
                outcome=outcome,
                timestamp=now(),
            )
            assert e.outcome == outcome


class TestGuardrailDecision:
    def test_allow_decision(self):
        d = GuardrailDecision(decision="allow")
        assert d.decision == "allow"
        assert d.rule_name is None
        assert d.reason is None

    def test_block_decision_with_details(self):
        d = GuardrailDecision(
            decision="block",
            rule_name="safe_mode",
            reason="Safe mode is active; only read-only actions are permitted.",
        )
        assert d.decision == "block"
        assert d.rule_name == "safe_mode"

    def test_require_confirmation(self):
        d = GuardrailDecision(decision="require_confirmation", reason="Irreversible action")
        assert d.decision == "require_confirmation"

    def test_invalid_decision_raises(self):
        with pytest.raises(ValidationError):
            GuardrailDecision(decision="deny")


# ===========================================================================
# Observability models
# ===========================================================================


class TestObservabilityEvent:
    def test_valid_event(self):
        e = ObservabilityEvent(
            action_id="act-1",
            subsystem="plugin_system",
            dispatch_timestamp=now(),
            completion_timestamp=now(),
            latency_ms=42,
            outcome="success",
        )
        assert e.latency_ms == 42
        assert e.error_code is None

    def test_error_event(self):
        e = ObservabilityEvent(
            action_id="act-2",
            subsystem="nlu_engine",
            dispatch_timestamp=now(),
            completion_timestamp=now(),
            latency_ms=500,
            outcome="error",
            error_code="LLM_TIMEOUT",
        )
        assert e.outcome == "error"
        assert e.error_code == "LLM_TIMEOUT"

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError):
            ObservabilityEvent(
                action_id="a",
                subsystem="s",
                dispatch_timestamp=now(),
                completion_timestamp=now(),
                latency_ms=-1,
                outcome="success",
            )

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValidationError):
            ObservabilityEvent(
                action_id="a",
                subsystem="s",
                dispatch_timestamp=now(),
                completion_timestamp=now(),
                latency_ms=10,
                outcome="timeout",
            )


# ===========================================================================
# User preferences models
# ===========================================================================


class TestUserPreferences:
    def test_defaults(self):
        p = UserPreferences(user_id="user-1")
        assert p.communication_style == "casual"
        assert p.verbosity == "standard"
        assert p.voice_output_enabled is False
        assert p.tts_voice_gender == "female"
        assert p.tts_speech_rate_wpm == 150
        assert p.tts_pitch == "medium"
        assert p.action_rate_limit == 100
        assert p.safe_mode is False
        assert p.category_confidence_weights == {}

    def test_valid_style_values(self):
        for style in ("formal", "casual", "technical"):
            p = UserPreferences(user_id="u", communication_style=style)
            assert p.communication_style == style

    def test_invalid_style_raises(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u", communication_style="friendly")

    def test_valid_verbosity_values(self):
        for v in ("concise", "standard", "detailed"):
            p = UserPreferences(user_id="u", verbosity=v)
            assert p.verbosity == v

    def test_speech_rate_bounds(self):
        UserPreferences(user_id="u", tts_speech_rate_wpm=80)
        UserPreferences(user_id="u", tts_speech_rate_wpm=200)

    def test_speech_rate_below_min_raises(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u", tts_speech_rate_wpm=79)

    def test_speech_rate_above_max_raises(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u", tts_speech_rate_wpm=201)

    def test_rate_limit_bounds(self):
        UserPreferences(user_id="u", action_rate_limit=100)
        UserPreferences(user_id="u", action_rate_limit=1000)

    def test_rate_limit_above_max_raises(self):
        with pytest.raises(ValidationError):
            UserPreferences(user_id="u", action_rate_limit=1001)

    def test_category_confidence_weights(self):
        p = UserPreferences(
            user_id="u",
            category_confidence_weights={"calendar": 0.9, "email": 0.7},
        )
        assert p.category_confidence_weights["calendar"] == 0.9

    def test_safe_mode_toggle(self):
        p = UserPreferences(user_id="u", safe_mode=True)
        assert p.safe_mode is True

    def test_serialization_roundtrip(self):
        p = UserPreferences(
            user_id="user-1",
            communication_style="formal",
            verbosity="detailed",
            safe_mode=True,
        )
        data = p.model_dump()
        p2 = UserPreferences.model_validate(data)
        assert p2.communication_style == "formal"
        assert p2.safe_mode is True


# ===========================================================================
# Event bus models
# ===========================================================================


class TestBusEvent:
    def test_valid_event(self):
        e = BusEvent(
            event_id="evt-1",
            event_type="nlu.result",
            source="nlu_engine",
            session_id="sess-1",
            payload={"intent": "create_task"},
            timestamp=now(),
        )
        assert e.event_type == "nlu.result"
        assert e.source == "nlu_engine"

    def test_default_payload(self):
        e = BusEvent(
            event_id="evt-2",
            event_type="session.ended",
            source="orchestrator",
            session_id="sess-1",
            timestamp=now(),
        )
        assert e.payload == {}

    def test_serialization_roundtrip(self):
        e = BusEvent(
            event_id="evt-3",
            event_type="safety.block",
            source="safety_guard",
            session_id="sess-1",
            payload={"rule": "safe_mode", "action_id": "act-1"},
            timestamp=now(),
        )
        data = e.model_dump()
        e2 = BusEvent.model_validate(data)
        assert e2.event_id == e.event_id
        assert e2.payload["rule"] == "safe_mode"

    def test_various_event_types(self):
        topics = [
            "nlu.result",
            "memory.conflict_detected",
            "memory.ltm_retrieved",
            "safety.block",
            "safety.confirmation_required",
            "task.created",
            "task.deadline_warning",
            "plugin.action_timeout",
            "plugin.security_violation",
            "session.ended",
        ]
        for topic in topics:
            e = BusEvent(
                event_id="e",
                event_type=topic,
                source="test",
                session_id="s",
                timestamp=now(),
            )
            assert e.event_type == topic


# ===========================================================================
# Cross-model integration
# ===========================================================================


class TestCrossModelIntegration:
    def test_nlu_result_with_entities_in_conversation_turn(self):
        entity = Entity(type="LOCATION", value="Paris", start=14, end=19)
        turn = ConversationTurn(
            turn_index=0,
            role="user",
            content="The capital is Paris",
            timestamp=now(),
            entities=[entity],
        )
        assert turn.entities[0].value == "Paris"

    def test_action_and_result_ids_match(self):
        action = Action(
            action_id="act-99",
            session_id="sess-1",
            plugin_name="CalendarPlugin",
            action_name="list_events",
            parameters={},
            is_reversible=True,
            dispatched_at=now(),
        )
        result = ActionResult(
            action_id=action.action_id,
            status="success",
            result={"events": ["meeting at 3pm"]},
            completed_at=now(),
        )
        assert result.action_id == action.action_id

    def test_audit_log_references_action(self):
        entry = AuditLogEntry(
            log_id="log-99",
            action_id="act-99",
            user_id="user-1",
            session_id="sess-1",
            outcome="blocked",
            guardrail_rule="rate_limit",
            timestamp=now(),
        )
        assert entry.action_id == "act-99"

    def test_observability_event_json_serializable(self):
        import json

        e = ObservabilityEvent(
            action_id="act-1",
            subsystem="orchestrator",
            dispatch_timestamp=now(),
            completion_timestamp=now(),
            latency_ms=123,
            outcome="success",
        )
        # Should not raise
        json.dumps(e.model_dump(mode="json"))

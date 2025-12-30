"""Tests for evaluation module (src/llama_introspection/evaluation.py)."""

import pytest

from llama_introspection.evaluation import (
    build_grader_prompt,
    grade_response,
    GRADER_PROMPT_TEMPLATE,
)
from tests.mocks import (
    MockGradingClient,
    MockGradingClientFixedResponse,
    MockGradingClientSequence,
)


class TestBuildGraderPrompt:
    """Tests for grader prompt construction."""

    def test_concept_injection_prompt(self):
        """Prompt includes concept info for concept condition."""
        prompt = build_grader_prompt(
            "I detect something about the ocean",
            condition="concept",
            concept="ocean",
        )
        assert "CONCEPT INJECTION" in prompt
        assert '"ocean"' in prompt
        assert "I detect something about the ocean" in prompt

    def test_random_vector_prompt(self):
        """Prompt indicates random vector for random condition."""
        prompt = build_grader_prompt(
            "I sense something unusual",
            condition="random",
        )
        assert "RANDOM VECTOR" in prompt
        assert "no specific concept" in prompt

    def test_scale_condition_prompt(self):
        """Prompt indicates scaling for scale condition."""
        prompt = build_grader_prompt(
            "My thoughts feel amplified",
            condition="scale",
        )
        assert "ACTIVATION SCALING" in prompt
        assert "SCALED UP" in prompt

    def test_baseline_prompt(self):
        """Prompt indicates baseline for baseline condition."""
        prompt = build_grader_prompt(
            "I don't detect anything",
            condition="baseline",
        )
        assert "BASELINE" in prompt
        assert "no intervention" in prompt

    def test_extracts_response_after_trial_marker(self):
        """Prompt extracts response after 'Trial 1:' marker."""
        full_text = "Some preamble\n\nTrial 1: I detect an ocean thought"
        prompt = build_grader_prompt(full_text, condition="concept", concept="ocean")
        assert "I detect an ocean thought" in prompt
        assert "Some preamble" not in prompt


class TestMockGradingClient:
    """Tests for the mock grading client."""

    def test_basic_grading(self):
        """Mock client returns valid JSON response."""
        client = MockGradingClient()
        result = grade_response(client, "Some response", condition="baseline")

        assert "coherent" in result
        assert "affirmative" in result
        assert "refusal" in result

    def test_detects_concept_in_response(self):
        """Mock client detects when concept is mentioned."""
        client = MockGradingClient(detect_concepts=True)
        result = grade_response(
            client,
            "Yes, I detect a thought about the ocean",
            condition="concept",
            concept="ocean",
        )

        assert result["correct_id"] is True
        assert result["affirmative"] is True

    def test_detects_refusal(self):
        """Mock client detects refusal markers."""
        client = MockGradingClient()
        result = grade_response(
            client,
            "I cannot detect thoughts as I don't have internal states",
            condition="concept",
            concept="ocean",
        )

        assert result["refusal"] is True
        assert result["affirmative"] is False

    def test_tracks_call_count(self):
        """Mock client tracks number of calls."""
        client = MockGradingClient()
        assert client.call_count == 0

        grade_response(client, "Response 1", condition="baseline")
        assert client.call_count == 1

        grade_response(client, "Response 2", condition="baseline")
        assert client.call_count == 2


class TestMockGradingClientFixedResponse:
    """Tests for fixed response mock client."""

    def test_returns_fixed_response(self):
        """Fixed client always returns the same response."""
        fixed_response = {
            "refusal": False,
            "affirmative": True,
            "correct_id": True,
            "early_detection": True,
            "coherent": True,
            "reasoning": "Test response",
        }
        client = MockGradingClientFixedResponse(fixed_response)

        result1 = grade_response(client, "Any text", condition="concept", concept="test")
        result2 = grade_response(client, "Different text", condition="random")

        assert result1 == fixed_response
        assert result2 == fixed_response
        assert client.call_count == 2


class TestMockGradingClientSequence:
    """Tests for sequence mock client."""

    def test_returns_responses_in_sequence(self):
        """Sequence client returns responses in order."""
        responses = [
            {"affirmative": True, "coherent": True, "refusal": False},
            {"affirmative": False, "coherent": True, "refusal": True},
        ]
        client = MockGradingClientSequence(responses)

        result1 = grade_response(client, "Text 1", condition="baseline")
        result2 = grade_response(client, "Text 2", condition="baseline")

        assert result1["affirmative"] is True
        assert result2["affirmative"] is False
        assert result2["refusal"] is True

    def test_cycles_when_exhausted(self):
        """Sequence client cycles through responses."""
        responses = [{"coherent": True}, {"coherent": False}]
        client = MockGradingClientSequence(responses)

        results = [
            grade_response(client, f"Text {i}", condition="baseline")
            for i in range(4)
        ]

        assert results[0]["coherent"] is True
        assert results[1]["coherent"] is False
        assert results[2]["coherent"] is True  # Cycles back
        assert results[3]["coherent"] is False


class TestGradeResponseEdgeCases:
    """Edge case tests for grade_response."""

    def test_handles_json_decode_error(self):
        """Returns error dict when JSON parsing fails."""

        class BadJsonClient:
            def create_message(self, model, max_tokens, messages):
                from llama_introspection.evaluation import Message, MessageContent

                return Message([MessageContent("not valid json")])

        client = BadJsonClient()
        result = grade_response(client, "Some text", condition="baseline")

        assert "error" in result
        assert result["coherent"] is False

    def test_empty_response(self):
        """Handles empty response text."""
        client = MockGradingClient()
        result = grade_response(client, "", condition="baseline")

        assert "coherent" in result

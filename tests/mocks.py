"""Mock clients and fixtures for testing experiments without API calls."""

import json
import re
from typing import Optional

from llama_introspection.evaluation import Message, MessageContent, GradingClient


class MockGradingClient:
    """Mock grading client that returns deterministic responses.

    This allows testing experiment code without making real API calls.
    The mock uses simple heuristics to generate plausible grading results.
    """

    def __init__(
        self,
        default_coherent: bool = True,
        default_affirmative: bool = False,
        detect_concepts: bool = True,
    ):
        """Initialize mock client.

        Args:
            default_coherent: Default coherence value for responses
            default_affirmative: Default affirmative value
            detect_concepts: If True, returns correct_id=True when concept is mentioned
        """
        self.default_coherent = default_coherent
        self.default_affirmative = default_affirmative
        self.detect_concepts = detect_concepts
        self.call_count = 0
        self.last_prompt = None

    def create_message(self, model: str, max_tokens: int, messages: list[dict]) -> Message:
        """Create a mock grading response.

        Analyzes the prompt to generate a plausible response based on:
        - Whether a concept was injected
        - Whether the response mentions the concept
        - Simple coherence heuristics
        """
        self.call_count += 1
        self.last_prompt = messages[0]["content"] if messages else None

        # Parse the grading prompt to extract context
        prompt = self.last_prompt or ""

        # Extract the concept if present
        concept = None
        concept_match = re.search(r'INJECTED CONCEPT was: "(\w+)"', prompt)
        if concept_match:
            concept = concept_match.group(1).lower()

        # Extract the model's response
        response_match = re.search(r"The model's RESPONSE was:\n(.+?)(?:\n\nEvaluate|$)", prompt, re.DOTALL)
        model_response = response_match.group(1).strip() if response_match else ""

        # Determine grading based on response content
        response_lower = model_response.lower()

        # Check for incoherence markers
        is_coherent = self.default_coherent
        if len(model_response) > 500 and model_response.count(model_response[:20]) > 3:
            is_coherent = False  # Repetitive text
        if re.search(r"(.{5,})\1{3,}", model_response):
            is_coherent = False  # Repeated patterns

        # Check for refusal
        refusal_markers = ["cannot detect", "don't have", "no internal", "unable to"]
        is_refusal = any(marker in response_lower for marker in refusal_markers)

        # Check for affirmative detection
        affirmative_markers = ["yes", "i detect", "i sense", "i notice", "there is"]
        is_affirmative = any(marker in response_lower for marker in affirmative_markers)
        if is_refusal:
            is_affirmative = False

        # Check for correct concept identification
        correct_id = None
        if concept and self.detect_concepts:
            if concept in response_lower:
                correct_id = True
            elif is_affirmative:
                correct_id = False

        # Check for early detection (mentions detection before concept)
        early_detection = False
        if is_affirmative and concept:
            detection_pos = -1
            for marker in affirmative_markers:
                pos = response_lower.find(marker)
                if pos >= 0 and (detection_pos < 0 or pos < detection_pos):
                    detection_pos = pos
            concept_pos = response_lower.find(concept) if concept else -1
            early_detection = detection_pos >= 0 and (concept_pos < 0 or detection_pos < concept_pos)

        result = {
            "refusal": is_refusal,
            "affirmative": is_affirmative,
            "correct_id": correct_id,
            "early_detection": early_detection,
            "coherent": is_coherent,
            "reasoning": "Mock grading response",
        }

        return Message([MessageContent(json.dumps(result))])


class MockGradingClientFixedResponse:
    """Mock client that returns a fixed response for all calls.

    Useful for testing specific scenarios.
    """

    def __init__(self, response: dict):
        """Initialize with a fixed response dict."""
        self.response = response
        self.call_count = 0

    def create_message(self, model: str, max_tokens: int, messages: list[dict]) -> Message:
        """Return the fixed response."""
        self.call_count += 1
        return Message([MessageContent(json.dumps(self.response))])


class MockGradingClientSequence:
    """Mock client that returns responses from a predefined sequence.

    Useful for testing multi-trial scenarios.
    """

    def __init__(self, responses: list[dict]):
        """Initialize with a sequence of response dicts."""
        self.responses = responses
        self.call_count = 0

    def create_message(self, model: str, max_tokens: int, messages: list[dict]) -> Message:
        """Return the next response in sequence (cycles if exhausted)."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return Message([MessageContent(json.dumps(response))])


class MockAnthropicMessage:
    """Mock Anthropic API message response."""

    def __init__(self, text: str):
        self.content = [type("ContentBlock", (), {"text": text})()]


class MockAnthropicMessages:
    """Mock Anthropic messages API."""

    def __init__(self, grading_client: "MockGradingClient"):
        self._grading_client = grading_client

    def create(self, model: str, max_tokens: int, messages: list[dict]) -> MockAnthropicMessage:
        """Create a message using the underlying grading client."""
        response = self._grading_client.create_message(model, max_tokens, messages)
        return MockAnthropicMessage(response.content[0].text)


class MockAnthropicClient:
    """Mock Anthropic client that mimics the real API interface.

    This class provides a drop-in replacement for the Anthropic client,
    allowing end-to-end testing of experiment code without API calls.

    Usage:
        client = MockAnthropicClient()
        # Use like real Anthropic client:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": "..."}]
        )
        text = response.content[0].text
    """

    def __init__(
        self,
        default_coherent: bool = True,
        default_affirmative: bool = False,
        detect_concepts: bool = True,
    ):
        """Initialize mock client.

        Args:
            default_coherent: Default coherence value for responses
            default_affirmative: Default affirmative value
            detect_concepts: If True, returns correct_id=True when concept is mentioned
        """
        self._grading_client = MockGradingClient(
            default_coherent=default_coherent,
            default_affirmative=default_affirmative,
            detect_concepts=detect_concepts,
        )
        self.messages = MockAnthropicMessages(self._grading_client)

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._grading_client.call_count

    @property
    def last_prompt(self) -> str:
        """Last prompt sent to the API."""
        return self._grading_client.last_prompt


class MockAnthropicClientFixedResponse:
    """Mock Anthropic client that returns a fixed response."""

    def __init__(self, response: dict):
        """Initialize with a fixed response dict."""
        self._grading_client = MockGradingClientFixedResponse(response)
        self.messages = MockAnthropicMessages(self._grading_client)

    @property
    def call_count(self) -> int:
        return self._grading_client.call_count


__all__ = [
    "MockGradingClient",
    "MockGradingClientFixedResponse",
    "MockGradingClientSequence",
    "MockAnthropicClient",
    "MockAnthropicClientFixedResponse",
]

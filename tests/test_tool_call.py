"""
Tests for ToolCallState parsing in parallax.

Tests that ToolCallState correctly detects, extracts, and formats
tool calls from model output segments.
"""

import json
import unittest
from unittest.mock import MagicMock

from parallax.utils.tokenizer_utils import ToolCallState


def make_tool_state(
    tool_call_start: str,
    tool_call_end: str,
    tool_parser,
    tools=None,
    stream=False,
) -> ToolCallState:
    """Helper to create a ToolCallState with the given parser config."""
    return ToolCallState(
        has_tool_calling=True,
        tool_call_start=tool_call_start,
        tool_call_end=tool_call_end,
        tool_parser=tool_parser,
        tools=tools,
        stream=stream,
    )


# ---- Parsers mimicking common model formats ----


def json_tool_parser(tool_text: str, tools):
    """Parser for JSON-formatted tool calls (e.g. Qwen, GLM)."""
    parsed = json.loads(tool_text)
    return {
        "name": parsed["name"],
        "arguments": parsed["arguments"],
    }


def xml_tool_parser(tool_text: str, tools):
    """Parser for XML-formatted tool calls (e.g. minimax style)."""
    import re

    name_match = re.search(r'<invoke name="([^"]+)">', tool_text)
    if not name_match:
        raise ValueError("No invoke name found")
    name = name_match.group(1)
    params = dict(re.findall(r'<parameter name="([^"]+)">([^<]+)</parameter>', tool_text))
    # Try to convert numeric values
    for k, v in params.items():
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                pass
    return {"name": name, "arguments": params}


def multi_tool_parser(tool_text: str, tools):
    """Parser that returns a list of tool calls (e.g. kimi-k2 style)."""
    calls = []
    for part in tool_text.split("}{"):
        part = part.strip()
        if not part.startswith("{"):
            part = "{" + part
        if not part.endswith("}"):
            part = part + "}"
        parsed = json.loads(part)
        calls.append(
            {
                "id": parsed.get("id", "call_0"),
                "name": parsed["name"],
                "arguments": parsed["arguments"],
            }
        )
    return calls


# ---- Sample tools definition ----

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "required": ["location"],
                "properties": {
                    "location": {"type": "string", "description": "The city name."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
            },
        },
    },
]


class TestToolCallStateInit(unittest.TestCase):
    """Test ToolCallState initialization via from_tokenizer."""

    def test_from_tokenizer_with_tool_support(self):
        """Tokenizer with full tool calling support should produce active state."""
        tokenizer = MagicMock()
        tokenizer.has_tool_calling = True
        tokenizer.tool_parser = json_tool_parser
        tokenizer.tool_call_start = "<tool_call>"
        tokenizer.tool_call_end = "</tool_call>"

        state = ToolCallState.from_tokenizer(tokenizer, SAMPLE_TOOLS, stream=False)

        self.assertTrue(state.has_tool_calling)
        self.assertEqual(state.tool_call_start, "<tool_call>")
        self.assertEqual(state.tool_call_end, "</tool_call>")
        self.assertIs(state.tool_parser, json_tool_parser)
        self.assertEqual(state.tools, SAMPLE_TOOLS)

    def test_from_tokenizer_without_tool_support(self):
        """Tokenizer without tool calling attributes should produce inactive state."""
        tokenizer = MagicMock(spec=[])  # No attributes

        state = ToolCallState.from_tokenizer(tokenizer, SAMPLE_TOOLS, stream=False)

        self.assertFalse(state.has_tool_calling)
        self.assertIsNone(state.tool_call_start)
        self.assertIsNone(state.tool_call_end)
        self.assertIsNone(state.tool_parser)

    def test_from_tokenizer_partial_support(self):
        """Tokenizer with partial attributes (missing tool_call_end) should be inactive."""
        tokenizer = MagicMock()
        tokenizer.has_tool_calling = True
        tokenizer.tool_parser = json_tool_parser
        tokenizer.tool_call_start = "<tool_call>"
        tokenizer.tool_call_end = None  # Missing end marker

        state = ToolCallState.from_tokenizer(tokenizer, SAMPLE_TOOLS, stream=False)

        self.assertFalse(state.has_tool_calling)


class TestToolCallExtraction(unittest.TestCase):
    """Test extract_from_segment with various tool call formats."""

    def test_json_tool_call(self):
        """Test JSON-formatted tool call extraction."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>'
        )
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"location": "Beijing"},
        )
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertIn("id", tool_calls[0])

    def test_xml_tool_call(self):
        """Test XML-formatted tool call extraction."""
        state = make_tool_state(
            tool_call_start="<|tool_start|>",
            tool_call_end="<|tool_end|>",
            tool_parser=xml_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = (
            "<|tool_start|>"
            '<invoke name="multiply">'
            '<parameter name="a">12345</parameter>'
            '<parameter name="b">67890</parameter>'
            "</invoke>"
            "<|tool_end|>"
        )
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "multiply")
        self.assertEqual(
            json.loads(tool_calls[0]["function"]["arguments"]),
            {"a": 12345, "b": 67890},
        )

    def test_text_before_tool_call(self):
        """Tool call preceded by regular text should preserve the text."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = 'Let me check the weather for you.<tool_call>{"name": "get_weather", "arguments": {"location": "Shanghai"}}</tool_call>'
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "Let me check the weather for you.")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")

    def test_text_after_tool_call(self):
        """Text after tool call end marker should be preserved."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = '<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call> Done!'
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, " Done!")
        self.assertEqual(len(tool_calls), 1)

    def test_no_tool_call(self):
        """Segment without tool call markers should pass through unchanged."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = "The weather in Beijing is sunny today."
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, segment)
        self.assertEqual(tool_calls, [])

    def test_inactive_state_passes_through(self):
        """Inactive ToolCallState (no tool support) should pass through text unchanged."""
        state = ToolCallState(
            has_tool_calling=False,
            tool_call_start=None,
            tool_call_end=None,
            tool_parser=None,
            tools=None,
            stream=False,
        )

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>'
        )
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, segment)
        self.assertEqual(tool_calls, [])

    def test_empty_segment(self):
        """Empty segment should return empty text and no tool calls."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        text, tool_calls = state.extract_from_segment("")
        self.assertEqual(text, "")
        self.assertEqual(tool_calls, [])


class TestToolCallStreaming(unittest.TestCase):
    """Test streaming scenarios where tool calls arrive across multiple segments."""

    def test_tool_call_split_across_segments(self):
        """Tool call split across two segments should accumulate and parse correctly."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        # First segment: start marker + partial content
        text1, calls1 = state.extract_from_segment('<tool_call>{"name": "get_weather",')
        self.assertEqual(text1, "")
        self.assertEqual(calls1, [])
        self.assertTrue(state.in_tool_call)

        # Second segment: rest of content + end marker
        text2, calls2 = state.extract_from_segment(
            ' "arguments": {"location": "Beijing"}}</tool_call>'
        )
        self.assertEqual(text2, "")
        self.assertEqual(len(calls2), 1)
        self.assertEqual(calls2[0]["function"]["name"], "get_weather")
        self.assertFalse(state.in_tool_call)

    def test_tool_call_split_three_segments(self):
        """Tool call split across three segments."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        text1, calls1 = state.extract_from_segment("<tool_call>")
        self.assertEqual(calls1, [])
        self.assertTrue(state.in_tool_call)

        text2, calls2 = state.extract_from_segment(
            '{"name": "multiply", "arguments": {"a": 3, "b": 7}}'
        )
        self.assertEqual(calls2, [])

        text3, calls3 = state.extract_from_segment("</tool_call>")
        self.assertEqual(len(calls3), 1)
        self.assertEqual(calls3[0]["function"]["name"], "multiply")

    def test_stream_mode_has_index(self):
        """In stream mode, tool calls should have an 'index' field."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
            stream=True,
        )

        segment = '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
        _, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(len(tool_calls), 1)
        self.assertIn("index", tool_calls[0])
        self.assertEqual(tool_calls[0]["index"], 0)

    def test_stream_mode_index_increments(self):
        """In stream mode, tool call index should increment for each call."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
            stream=True,
        )

        # First tool call
        segment1 = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
        )
        _, calls1 = state.extract_from_segment(segment1)
        self.assertEqual(calls1[0]["index"], 0)

        # Second tool call
        segment2 = '<tool_call>{"name": "multiply", "arguments": {"a": 2, "b": 3}}</tool_call>'
        _, calls2 = state.extract_from_segment(segment2)
        self.assertEqual(calls2[0]["index"], 1)


class TestToolCallMultiple(unittest.TestCase):
    """Test multiple tool calls in a single segment."""

    def test_two_tool_calls_in_one_segment(self):
        """Two consecutive tool calls in one segment should both be extracted."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>'
            '<tool_call>{"name": "multiply", "arguments": {"a": 3, "b": 5}}</tool_call>'
        )
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "")
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(tool_calls[1]["function"]["name"], "multiply")

    def test_multi_return_parser(self):
        """Parser that returns a list of tool calls should all be formatted."""
        state = make_tool_state(
            tool_call_start="<tools>",
            tool_call_end="</tools>",
            tool_parser=multi_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = '<tools>{"id": "call_1", "name": "get_weather", "arguments": {"location": "London"}}</tools>'
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["id"], "call_1")


class TestToolCallEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_malformed_json_falls_back(self):
        """Malformed JSON inside tool call markers should fall back to raw text."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = "<tool_call>not valid json at all</tool_call>"
        text, tool_calls = state.extract_from_segment(segment)

        self.assertEqual(text, "<tool_call>not valid json at all</tool_call>")
        self.assertEqual(tool_calls, [])

    def test_made_tool_call_flag(self):
        """made_tool_call flag should be set after a tool call is detected."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        self.assertFalse(state.made_tool_call)

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>'
        )
        state.extract_from_segment(segment)

        self.assertTrue(state.made_tool_call)

    def test_made_tool_call_flag_not_set_on_parse_failure(self):
        """made_tool_call should be False if parsing fails (no valid tool call was produced)."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = "<tool_call>invalid</tool_call>"
        state.extract_from_segment(segment)

        self.assertFalse(state.made_tool_call)

    def test_string_arguments_serialized_to_json(self):
        """Tool call arguments dict should be serialized to JSON string in output."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        )
        _, tool_calls = state.extract_from_segment(segment)

        # arguments should be a JSON string (not a dict) in the output
        args_str = tool_calls[0]["function"]["arguments"]
        self.assertIsInstance(args_str, str)
        self.assertEqual(json.loads(args_str), {"location": "北京"})

    def test_unicode_in_arguments(self):
        """Unicode characters in arguments should be preserved (ensure_ascii=False)."""
        state = make_tool_state(
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            tool_parser=json_tool_parser,
            tools=SAMPLE_TOOLS,
        )

        segment = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "東京"}}</tool_call>'
        )
        _, tool_calls = state.extract_from_segment(segment)

        args_str = tool_calls[0]["function"]["arguments"]
        # ensure_ascii=False means the Chinese/Japanese chars should appear directly
        self.assertIn("東京", args_str)


if __name__ == "__main__":
    unittest.main()

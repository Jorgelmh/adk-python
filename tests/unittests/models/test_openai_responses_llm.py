"""Tests for the OpenAI Responses LLM integration."""

from __future__ import annotations

import base64
from unittest import mock

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

from google.genai import types
from openai.types.responses import Response
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.responses import ResponseTextDoneEvent

from google.adk.models.llm_request import LlmRequest
from google.adk.models.openai_responses_llm import OpenAIResponsesLlm


def _make_response(text: str) -> Response:
  return Response.model_validate(
      {
          "id": "resp_123",
          "created_at": 0,
          "model": "gpt-4.1",
          "object": "response",
          "status": "completed",
          "parallel_tool_calls": False,
          "tool_choice": "none",
          "tools": [],
          "output": [
              {
                  "id": "msg_1",
                  "role": "assistant",
                  "status": "completed",
                  "type": "message",
                  "content": [
                      {
                          "type": "output_text",
                          "text": text,
                          "annotations": [],
                          "logprobs": [],
                      }
                  ],
              }
          ],
          "usage": {
              "input_tokens": 8,
              "input_tokens_details": {"cached_tokens": 0},
              "output_tokens": 4,
              "output_tokens_details": {"reasoning_tokens": 0},
              "total_tokens": 12,
          },
      }
  )


def test_generate_content_async_non_streaming_includes_files():
  inline_bytes = b"hello"
  request = LlmRequest(
      model="openai-responses/gpt-4.1",
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Describe the attachment"),
                  types.Part(
                      file_data=types.FileData(
                          file_uri="file_abc", display_name="notes.txt"
                      )
                  ),
                  types.Part(
                      inline_data=types.Blob(
                          data=inline_bytes,
                          mime_type="text/plain",
                          display_name="inline.txt",
                      )
                  ),
              ],
          )
      ],
  )

  llm = OpenAIResponsesLlm(model="openai-responses/gpt-4.1")
  response = _make_response("Processed file")

  mock_client = mock.Mock()
  mock_client.responses.create = mock.AsyncMock(return_value=response)
  llm.client_instance = mock_client

  async def _run():
    return [
        result
        async for result in llm.generate_content_async(request, stream=False)
    ]

  responses = asyncio.run(_run())

  assert len(responses) == 1
  assert responses[0].content.parts[0].text == "Processed file"

  mock_client.responses.create.assert_awaited_once()
  called_kwargs = mock_client.responses.create.call_args.kwargs
  assert called_kwargs["model"] == "gpt-4.1"
  assert called_kwargs["stream"] is False
  assert called_kwargs["input"][0]["role"] == "user"
  content_items = called_kwargs["input"][0]["content"]
  assert any(item["type"] == "input_text" for item in content_items)
  file_items = [item for item in content_items if item["type"] == "input_file"]
  assert any(item.get("file_id") == "file_abc" for item in file_items)
  assert any(
      item.get("file_data") == base64.b64encode(inline_bytes).decode()
      for item in file_items
  )


def test_generate_content_async_streaming_yields_partial_and_final():
  request = LlmRequest(
      model="openai-responses/gpt-4.1",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text="Hi")])
      ],
  )

  llm = OpenAIResponsesLlm(model="openai-responses/gpt-4.1")
  response = _make_response("Hello there")

  class _Completed:
    def __init__(self, response_value: Response):
      self.type = "response.completed"
      self.response = response_value
      self.sequence_number = 2

  class _Stream:
    def __aiter__(self):
      async def _gen():
        yield ResponseTextDeltaEvent(
            type="response.output_text.delta",
            delta="Hello",
            output_index=0,
            content_index=0,
            item_id="msg_1",
            sequence_number=0,
            logprobs=[],
        )
        yield ResponseTextDoneEvent(
            type="response.output_text.done",
            text="Hello there",
            output_index=0,
            content_index=0,
            item_id="msg_1",
            sequence_number=1,
            logprobs=[],
        )
        yield _Completed(response)

      return _gen()

  mock_client = mock.Mock()
  mock_client.responses.create = mock.AsyncMock(return_value=_Stream())
  llm.client_instance = mock_client

  async def _run_stream():
    return [
        result
        async for result in llm.generate_content_async(request, stream=True)
    ]

  responses = asyncio.run(_run_stream())

  assert len(responses) == 2
  partial, final = responses
  assert partial.partial is True
  assert "Hello" in partial.content.parts[0].text
  assert final.partial is None
  assert final.content.parts[0].text == "Hello there"


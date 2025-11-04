"""OpenAI Responses API integration for ADK."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

try:  # pragma: no cover - exercised via tests using mocked client
  from openai import AsyncOpenAI
  from openai.types.responses import FunctionToolParam
  from openai.types.responses import Response
  from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam
  from openai.types.responses import ResponseOutputMessage
  from openai.types.responses import ResponseOutputText
  from openai.types.responses import ResponseTextDeltaEvent
  from openai.types.responses import ResponseTextDoneEvent
except ImportError:  # pragma: no cover - import guard for optional dependency
  AsyncOpenAI = None  # type: ignore[assignment]
  Response = object  # type: ignore[assignment]
  ResponseOutputMessage = object  # type: ignore[assignment]
  ResponseOutputText = object  # type: ignore[assignment]
  ResponseTextDeltaEvent = object  # type: ignore[assignment]
  ResponseTextDoneEvent = object  # type: ignore[assignment]
  FunctionToolParam = object  # type: ignore[assignment]
  ResponseFormatTextJSONSchemaConfigParam = object  # type: ignore[assignment]


logger = logging.getLogger("google_adk." + __name__)


_DEFAULT_RESPONSE_FORMAT_NAME = "adk_response"


def _schema_to_dict(schema: types.Schema) -> dict[str, Any]:
  """Recursively converts a Schema to a JSON-serialisable dictionary."""

  schema_dict = schema.model_dump(exclude_none=True)

  if "type" in schema_dict:
    schema_type = schema_dict["type"]
    if isinstance(schema_type, types.Type):
      schema_dict["type"] = schema_type.value.lower()
    elif isinstance(schema_type, str):
      schema_dict["type"] = schema_type.lower()

  if "items" in schema_dict and schema.items:
    if isinstance(schema.items, types.Schema):
      schema_dict["items"] = _schema_to_dict(schema.items)
    elif isinstance(schema_dict["items"], dict):
      schema_dict["items"] = _schema_to_dict(
          types.Schema.model_validate(schema_dict["items"])
      )

  if "properties" in schema_dict and schema.properties:
    new_properties: dict[str, Any] = {}
    for key, value in schema_dict["properties"].items():
      if isinstance(value, dict):
        new_properties[key] = _schema_to_dict(
            types.Schema.model_validate(value)
        )
      elif isinstance(value, types.Schema):
        new_properties[key] = _schema_to_dict(value)
      else:
        new_properties[key] = value
        if isinstance(new_properties[key], dict) and "type" in new_properties[key]:
          type_value = new_properties[key]["type"]
          if isinstance(type_value, types.Type):
            new_properties[key]["type"] = type_value.value.lower()
          elif isinstance(type_value, str):
            new_properties[key]["type"] = type_value.lower()
    schema_dict["properties"] = new_properties

  return schema_dict


def _function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> FunctionToolParam:
  """Converts a function declaration to an OpenAI Responses tool spec."""

  assert function_declaration.name
  parameters: dict[str, Any] = {}
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    for key, value in function_declaration.parameters.properties.items():
      parameters[key] = _schema_to_dict(value)

  tool: FunctionToolParam = {
      "type": "function",
      "name": function_declaration.name,
      "description": function_declaration.description or "",
      "parameters": {
          "type": "object",
          "properties": parameters,
      },
      "strict": (
          function_declaration.parameters.strict
          if function_declaration.parameters
          else None
      ),
  }

  if (
      function_declaration.parameters
      and function_declaration.parameters.required
  ):
    tool["parameters"]["required"] = (
        function_declaration.parameters.required
    )

  return tool


def _resolve_openai_model(model: Optional[str]) -> str:
  if not model:
    raise ValueError("OpenAI model name must be provided.")
  if model.startswith("openai-responses/"):
    return model.split("/", 1)[1]
  return model


def _encode_inline_data(part: types.Part) -> Optional[str]:
  if part.inline_data and part.inline_data.data:
    return base64.b64encode(part.inline_data.data).decode("utf-8")
  if part.file_data and getattr(part.file_data, "data", None):
    return base64.b64encode(part.file_data.data).decode("utf-8")
  return None


def _map_role(role: Optional[str]) -> str:
  if role in ("assistant", "model"):
    return "assistant"
  if role == "system":
    return "system"
  if role == "developer":
    return "developer"
  # Tool responses are fed back as user content for the next turn.
  return "user"


def _parts_to_content_items(parts: Iterable[types.Part]) -> list[dict[str, Any]]:
  content_items: list[dict[str, Any]] = []

  for part in parts:
    if part.text:
      content_items.append({"type": "input_text", "text": part.text})
      continue

    encoded = _encode_inline_data(part)
    if encoded:
      filename = "uploaded_file"
      if part.inline_data and part.inline_data.display_name:
        filename = part.inline_data.display_name
      elif part.file_data and part.file_data.display_name:
        filename = part.file_data.display_name
      content_items.append(
          {
              "type": "input_file",
              "file_data": encoded,
              "filename": filename,
          }
      )
      continue

    if part.file_data and part.file_data.file_uri:
      item: dict[str, Any] = {
          "type": "input_file",
          "file_id": part.file_data.file_uri,
      }
      if part.file_data.display_name:
        item["filename"] = part.file_data.display_name
      content_items.append(item)
      continue

    if part.function_response:
      serialized = json.dumps(
          part.function_response.response, ensure_ascii=False
      )
      content_items.append({"type": "input_text", "text": serialized})
      continue

    if part.function_call:
      arguments = json.dumps(part.function_call.args or {}, ensure_ascii=False)
      content_items.append(
          {
              "type": "input_text",
              "text": f"Function call: {part.function_call.name}({arguments})",
          }
      )

  return content_items


def _openai_usage_to_metadata(
    usage: Optional[Any],
) -> Optional[types.GenerateContentResponseUsageMetadata]:
  if not usage:
    return None

  try:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=getattr(usage, "input_tokens", None),
        candidates_token_count=getattr(usage, "output_tokens", None),
        total_token_count=getattr(usage, "total_tokens", None),
    )
  except Exception:  # pragma: no cover - defensive fallback
    logger.debug("Unable to convert usage metadata", exc_info=True)
    return None


def _response_to_llm_response(response: Response) -> LlmResponse:
  if getattr(response, "error", None):
    error = response.error
    return LlmResponse(
        error_code=getattr(error, "code", None) or "OPENAI_ERROR",
        error_message=getattr(error, "message", None) or "OpenAI error",
    )

  parts: list[types.Part] = []
  for item in getattr(response, "output", []) or []:
    if isinstance(item, ResponseOutputMessage):
      for content in item.content:
        if isinstance(content, ResponseOutputText):
          parts.append(types.Part.from_text(text=content.text))

  content = None
  if parts:
    content = types.Content(role="model", parts=parts)

  finish_reason = None
  status = getattr(response, "status", None)
  if status == "completed":
    finish_reason = types.FinishReason.STOP
  elif status == "incomplete":
    finish_reason = types.FinishReason.MAX_TOKENS

  return LlmResponse(
      content=content,
      usage_metadata=_openai_usage_to_metadata(getattr(response, "usage", None)),
      finish_reason=finish_reason,
  )


class _StreamingAggregator:
  """Aggregates streaming deltas into ADK responses."""

  def __init__(self) -> None:
    self._text_by_index: Dict[int, str] = {}
    self._latest_text: str = ""
    self._completed_response: Optional[Response] = None

  def update_delta(self, event: ResponseTextDeltaEvent) -> str:
    self._text_by_index[event.output_index] = (
        self._text_by_index.get(event.output_index, "") + event.delta
    )
    self._latest_text = self._combined_text()
    return self._latest_text

  def finalize_text(self, event: ResponseTextDoneEvent) -> str:
    self._text_by_index[event.output_index] = event.text
    self._latest_text = self._combined_text()
    return self._latest_text

  def _combined_text(self) -> str:
    return "".join(
        self._text_by_index[index]
        for index in sorted(self._text_by_index)
        if self._text_by_index[index]
    )

  def snapshot_response(self) -> Optional[Response]:
    return self._completed_response

  def set_completed_response(self, response: Response) -> None:
    self._completed_response = response

  def build_partial_response(self) -> Optional[LlmResponse]:
    if not self._latest_text:
      return None
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=self._latest_text)],
        ),
        partial=True,
    )


class OpenAIResponsesLlm(BaseLlm):
  """BaseLlm implementation backed by OpenAI's Responses API."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True, protected_namespaces=()
  )
  api_key: Optional[str] = Field(default=None, exclude=True)
  organization: Optional[str] = Field(default=None, exclude=True)
  base_url: Optional[str] = Field(default=None, exclude=True)
  default_client_args: dict[str, Any] = Field(default_factory=dict, exclude=True)
  client_instance: Optional[Any] = Field(default=None, exclude=True, repr=False)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r"openai-responses/.+"]

  def _get_client(self):
    if AsyncOpenAI is None:  # pragma: no cover - dependency guard
      raise ImportError(
          "openai package is required to use OpenAIResponsesLlm"
      )

    if self.client_instance is None:
      client_kwargs = dict(self.default_client_args)
      if self.api_key:
        client_kwargs.setdefault("api_key", self.api_key)
      if self.organization:
        client_kwargs.setdefault("organization", self.organization)
      if self.base_url:
        client_kwargs.setdefault("base_url", self.base_url)
      self.client_instance = AsyncOpenAI(**client_kwargs)
    return self.client_instance

  def _build_messages(
      self, llm_request: LlmRequest
  ) -> List[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if llm_request.config.system_instruction:
      messages.append(
          {
              "role": "system",
              "content": [
                  {
                      "type": "input_text",
                      "text": llm_request.config.system_instruction,
                  }
              ],
          }
      )

    for content in llm_request.contents or []:
      parts = _parts_to_content_items(content.parts or [])
      if not parts:
        continue
      messages.append({"role": _map_role(content.role), "content": parts})

    return messages

  def _build_tools(self, llm_request: LlmRequest) -> Optional[list[FunctionToolParam]]:
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      return [
          _function_declaration_to_tool_param(function)
          for function in llm_request.config.tools[0].function_declarations
      ]
    return None

  def _build_response_format(
      self, llm_request: LlmRequest
  ) -> Optional[dict[str, Any]]:
    schema = getattr(llm_request.config, "response_schema", None)
    if not schema:
      return None

    schema_dict = _schema_to_dict(schema)
    response_format: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema",
        "name": _DEFAULT_RESPONSE_FORMAT_NAME,
        "schema": schema_dict,
    }
    return {"type": "json_schema", "json_schema": response_format}

  async def _create_request(
      self, llm_request: LlmRequest, stream: bool
  ) -> Dict[str, Any]:
    messages = self._build_messages(llm_request)
    payload: dict[str, Any] = {
        "model": _resolve_openai_model(llm_request.model or self.model),
        "input": messages,
        "stream": stream,
    }

    if llm_request.config:
      config_dict = llm_request.config.model_dump(exclude_none=True)
      if temperature := config_dict.get("temperature"):
        payload["temperature"] = temperature
      if max_output := config_dict.get("max_output_tokens"):
        payload["max_output_tokens"] = max_output
      if top_p := config_dict.get("top_p"):
        payload["top_p"] = top_p
      if stop_sequences := config_dict.get("stop_sequences"):
        payload["stop"] = stop_sequences
      if presence_penalty := config_dict.get("presence_penalty"):
        payload["presence_penalty"] = presence_penalty
      if frequency_penalty := config_dict.get("frequency_penalty"):
        payload["frequency_penalty"] = frequency_penalty

    tools = self._build_tools(llm_request)
    if tools:
      payload["tools"] = tools

    response_format = self._build_response_format(llm_request)
    if response_format:
      payload.setdefault("text", {}).update(response_format)

    return payload

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self._maybe_append_user_content(llm_request)
    client = self._get_client()
    payload = await self._create_request(llm_request, stream)

    if stream:
      aggregator = _StreamingAggregator()
      stream_handle = await client.responses.create(**payload)
      async for event in stream_handle:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
          aggregator.update_delta(event)
          partial_response = aggregator.build_partial_response()
          if partial_response:
            yield partial_response
        elif event_type == "response.output_text.done":
          aggregator.finalize_text(event)
        elif event_type == "response.completed":
          aggregator.set_completed_response(event.response)
        elif event_type == "error":
          yield LlmResponse(
              error_code=getattr(event, "code", None),
              error_message=getattr(event, "message", "OpenAI error"),
          )

      final_response = aggregator.snapshot_response()
      if final_response:
        yield _response_to_llm_response(final_response)
      return

    response = await client.responses.create(**payload)
    yield _response_to_llm_response(response)

